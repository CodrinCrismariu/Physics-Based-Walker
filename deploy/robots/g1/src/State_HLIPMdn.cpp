/**
 * State_HLIPMdn.cpp — CNNTransformerMDN distillation student deploy state.
 *
 * Depth subscription uses the raw CycloneDDS C API (dds_create_reader /
 * dds_take) with the IDL-generated descriptor in DepthImage_.hpp.
 * The descriptor contains the correct XTypes type hash, so CycloneDDS will
 * match this reader with the Python unitree_sdk2py ChannelPublisher writer.
 */

#include "State_HLIPMdn.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "param.h"

#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

// =============================================================================
// Stub observations
// =============================================================================

namespace isaaclab
{

REGISTER_OBSERVATION(depth_image)
{
    (void)env; (void)params;
    return std::vector<float>(DEPTH_PIXELS, DEPTH_MAX);
}

REGISTER_OBSERVATION(hlip_mdn_velocity_commands)
{
    (void)env; (void)params;
    return {0.f, 0.f, 0.f};
}

} // namespace isaaclab

// =============================================================================
// State_HLIPMdn
// =============================================================================

State_HLIPMdn::State_HLIPMdn(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env_->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    this->registered_checks.emplace_back(std::make_pair(
        [this]()->bool{ return abort_to_passive_; },
        FSMStringMap.right.at("Passive")
    ));
    this->registered_checks.emplace_back(std::make_pair(
        [&]()->bool{ return isaaclab::mdp::bad_orientation(env_.get(), 1.0); },
        FSMStringMap.right.at("Passive")
    ));
}

void State_HLIPMdn::enter()
{
    abort_to_passive_ = false;
    {
        std::lock_guard<std::mutex> lk(depth_mutex_);
        depth_received_ = false;
        depth_last_ts_  = 0;
        depth_buf_.fill(DEPTH_MAX);
    }
    // Clear observation history buffers.
    for (int t = 0; t < NUM_TERMS; ++t)
        obs_history_[t].clear();

    // ── 0. Initialize per-run logging ─────────────────────────────────────────
    init_logging();

    // ── 1. Start depth reader and wait for first frame ────────────────────────
    spdlog::info("[HLIPMdn] Starting depth subscriber on '{}'...", DEPTH_TOPIC);
    start_depth_reader();

    static constexpr int DEPTH_TIMEOUT_S = 10;
    bool got_frame = false;
    {
        std::unique_lock<std::mutex> lk(depth_mutex_);
        got_frame = depth_ready_cv_.wait_for(
            lk, std::chrono::seconds(DEPTH_TIMEOUT_S),
            [this]{ return depth_received_; });
    }

    if (!got_frame) {
        spdlog::error(
            "[HLIPMdn] No depth frame on '{}' within {}s. "
            "Is the depth publisher running?",
            DEPTH_TOPIC, DEPTH_TIMEOUT_S);
        stop_depth_reader();
        abort_to_passive_ = true;
        return;
    }

    // ── 2. Depth sanity stats ─────────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(depth_mutex_);
        float dmin = *std::min_element(depth_buf_.begin(), depth_buf_.end());
        float dmax = *std::max_element(depth_buf_.begin(), depth_buf_.end());
        float dsum = 0.f;
        for (float v : depth_buf_) dsum += v;
        spdlog::info(
            "[HLIPMdn] Depth OK — min={:.3f}m  max={:.3f}m  mean={:.3f}m  ({}x{} px)",
            dmin, dmax, dsum / float(DEPTH_PIXELS), DEPTH_WIDTH, DEPTH_HEIGHT);
    }

    // ── 3. Set PD gains ───────────────────────────────────────────────────────
    for (int i = 0; i < (int)env_->robot->data.joint_stiffness.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = env_->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env_->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    // ── 4. Start policy thread ────────────────────────────────────────────────
    env_->robot->update();
    env_->reset();

    policy_thread_running_ = true;
    policy_thread_ = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const double step_dt_s = env_->step_dt;           // should be 0.02
        const auto dt = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(step_dt_s));

        spdlog::info("[HLIPMdn] Policy thread started — step_dt={:.4f}s ({:.1f} Hz target)",
                     step_dt_s, 1.0 / step_dt_s);

        auto sleep_till = clock::now() + dt;

        // Hz tracking
        hz_step_count_ = 0;
        hz_window_start_ = clock::now();

        while (policy_thread_running_) {
            auto step_start = clock::now();

            env_->robot->update();

            std::unordered_map<std::string, std::vector<float>> obs_map;
            obs_map["student_vec"]       = build_student_vec();
            obs_map["head_camera_depth"] = get_depth_obs();

            auto raw_action = env_->alg->act(obs_map);

            // Cache raw network output for last_action obs BEFORE applying scale
            {
                std::lock_guard<std::mutex> lk(last_action_mutex_);
                last_raw_action_ = raw_action;
            }

            // Log all network inputs + action
            log_step(obs_map["student_vec"], obs_map["head_camera_depth"], raw_action);

            env_->action_manager->process_action(raw_action);
            env_->episode_length += 1;

            // Hz monitoring — report every 50 steps (~1 second)
            hz_step_count_++;
            if (hz_step_count_ >= 50) {
                auto now = clock::now();
                double elapsed = std::chrono::duration<double>(now - hz_window_start_).count();
                double hz = hz_step_count_ / elapsed;
                double step_ms = std::chrono::duration<double, std::milli>(
                    now - step_start).count();
                spdlog::info("[HLIPMdn] Policy rate: {:.1f} Hz  (target {:.0f} Hz, "
                             "last step {:.2f}ms, budget {:.1f}ms)",
                             hz, 1.0 / step_dt_s, step_ms, step_dt_s * 1000.0);
                hz_step_count_ = 0;
                hz_window_start_ = now;
            }

            // Sleep until next scheduled tick
            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;

            // Drift guard: if we're more than 2 steps behind schedule,
            // reset to avoid burst catch-up (drop missed ticks instead).
            auto now = clock::now();
            if (now > sleep_till + dt) {
                spdlog::warn("[HLIPMdn] Timing drift detected — resetting schedule");
                sleep_till = now + dt;
            }
        }
    });
}

void State_HLIPMdn::run()
{
    auto action = env_->action_manager->processed_actions();

    for (int i = 0; i < (int)env_->robot->data.joint_ids_map.size(); ++i)
        lowcmd->msg_.motor_cmd()[env_->robot->data.joint_ids_map[i]].q() = action[i];

    // ── Waist upright compensation (mirrors UprightWaistActionCfg) ────────────
    {
        using G1 = unitree::BaseArticulation<LowState_t::SharedPtr>;
        G1* robot = dynamic_cast<G1*>(env_->robot.get());

        const auto& q = robot->lowstate->msg_.imu_state().quaternion();
        float qw = q[0], qx = q[1], qy = q[2], qz = q[3];

        float sinr_cosp = 2.f * (qw * qx + qy * qz);
        float cosr_cosp = 1.f - 2.f * (qx * qx + qy * qy);
        float base_pitch  = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.f * (qw * qy - qz * qx);
        float base_roll = std::abs(sinp) >= 1.f
                         ? std::copysign(static_cast<float>(M_PI / 2.0), sinp)
                         : std::asin(sinp);

        const auto& dpos = env_->robot->data.default_joint_pos;
        lowcmd->msg_.motor_cmd()[13].q() = ((dpos.size() > 13) ? dpos[13] : 0.f) - base_pitch;
        lowcmd->msg_.motor_cmd()[14].q() = ((dpos.size() > 14) ? dpos[14] : 0.f) - base_roll;
    }
}

// =============================================================================
// Observation helpers
// =============================================================================

std::vector<float> State_HLIPMdn::build_raw_obs_frame()
{
    using G1 = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1* robot = dynamic_cast<G1*>(env_->robot.get());
    const auto& motors = robot->lowstate->msg_.motor_state();
    const auto& imu    = robot->lowstate->msg_.imu_state();
    const auto& data   = env_->robot->data;
    const int N = (int)data.joint_ids_map.size(); // 29

    std::vector<float> obs;
    obs.reserve(RAW_VEC_DIM);

    // [0:3] base_ang_vel — IMU gyroscope (rad/s), scale 1.0
    for (int i = 0; i < 3; ++i) obs.push_back(imu.gyroscope()[i]);

    // [3:6] projected_gravity — gravity vector expressed in robot base frame
    {
        const auto& q = imu.quaternion();
        Eigen::Quaternionf quat(q[0], q[1], q[2], q[3]);
        Eigen::Vector3f g_base = quat.toRotationMatrix().transpose() *
                                 Eigen::Vector3f(0.f, 0.f, -1.f);
        for (int i = 0; i < 3; ++i) obs.push_back(g_base[i]);
    }

    // [6:9] velocity_commands: [vx, 0, 0] * 2.0
    {
        constexpr float VX_MAX = 0.5f;  // capped below training max (0.6) for safety
        float vx = FSMState::lowstate->joystick.ly() * VX_MAX;
        vx = std::clamp(vx, -0.1f, 0.5f);
        obs.push_back(vx * 2.0f);
        obs.push_back(0.0f);
        obs.push_back(0.0f);
    }

    // [9:38] joint_pos — q - default_joint_pos, scale 1.0
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].q() - data.default_joint_pos[i]);

    // [38:67] joint_vel — dq * 0.05
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].dq() * 0.05f);

    // [67:96] last_action — raw network output (before scale/offset), scale 1.0
    {
        std::lock_guard<std::mutex> lk(last_action_mutex_);
        for (int i = 0; i < N; ++i)
            obs.push_back(i < (int)last_raw_action_.size() ? last_raw_action_[i] : 0.f);
    }

    return obs;
}

std::vector<float> State_HLIPMdn::build_student_vec()
{
    // 1. Get current raw 96-d frame.
    auto raw = build_raw_obs_frame();

    // 2. Split into per-term slices and push into history deques.
    //    Mirrors simulation CircularBuffer(max_len=5).append() per term.
    int offset = 0;
    for (int t = 0; t < NUM_TERMS; ++t) {
        int dim = TERM_DIMS[t];
        std::vector<float> term_obs(raw.begin() + offset, raw.begin() + offset + dim);
        offset += dim;

        auto& hist = obs_history_[t];
        if (hist.empty()) {
            // Backfill: simulation fills all slots with the first observation
            // (CircularBuffer.append with num_pushes==0 backfills all slots).
            for (int h = 0; h < OBS_HISTORY_LENGTH; ++h)
                hist.push_back(term_obs);
        } else {
            hist.push_back(term_obs);
            if ((int)hist.size() > OBS_HISTORY_LENGTH)
                hist.pop_front();
        }
    }

    // 3. Flatten with term-major ordering (matches flatten_history_dim=True):
    //    [term0_t0, term0_t1, ..., term0_t4, term1_t0, ..., term1_t4, ...]
    //    where t0=oldest, t4=newest.
    std::vector<float> out;
    out.reserve(HIST_VEC_DIM);
    for (int t = 0; t < NUM_TERMS; ++t) {
        for (int h = 0; h < OBS_HISTORY_LENGTH; ++h) {
            const auto& frame = obs_history_[t][h];
            out.insert(out.end(), frame.begin(), frame.end());
        }
    }

    return out;
}


std::vector<float> State_HLIPMdn::get_depth_obs()
{
    std::lock_guard<std::mutex> lk(depth_mutex_);
    return std::vector<float>(depth_buf_.begin(), depth_buf_.end());
}

// =============================================================================
// Raw CycloneDDS C API depth reader
// =============================================================================

void State_HLIPMdn::start_depth_reader()
{
    depth_running_ = true;

    depth_thread_ = std::thread([this] {
        // Create a participant on domain 0.
        // ChannelFactory::Init(0) was already called in main(), so both
        // the Unitree SDK participant and this one live on the same domain.
        dds_participant_ = dds_create_participant(0, nullptr, nullptr);
        if (dds_participant_ < 0) {
            spdlog::error("[HLIPMdn] dds_create_participant failed: {}", dds_participant_);
            depth_running_ = false;
            return;
        }

        dds_entity_t topic = dds_create_topic(
            dds_participant_, &DepthImage__desc, DEPTH_TOPIC, nullptr, nullptr);
        if (topic < 0) {
            spdlog::error("[HLIPMdn] dds_create_topic failed: {}", topic);
            dds_delete(dds_participant_);
            depth_running_ = false;
            return;
        }

        dds_qos_t* qos = dds_create_qos();
        dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, 0);
        dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 1);
        dds_reader_ = dds_create_reader(dds_participant_, topic, qos, nullptr);
        dds_delete_qos(qos);
        if (dds_reader_ < 0) {
            spdlog::error("[HLIPMdn] dds_create_reader failed: {}", dds_reader_);
            dds_delete(dds_participant_);
            depth_running_ = false;
            return;
        }

        spdlog::info("[HLIPMdn] Depth reader ready on '{}'.", DEPTH_TOPIC);

        // NULL sample pointer: CycloneDDS allocates and owns the buffer.
        // Required for variable-length sequence types to avoid heap corruption.
        void*             samples[1] = { nullptr };
        dds_sample_info_t si[1];

        while (depth_running_) {
            samples[0] = nullptr;
            int n = dds_take(dds_reader_, samples, si, 1, 1);
            if (n > 0 && si[0].valid_data && samples[0] != nullptr) {
                const DepthImage_* frame = static_cast<const DepthImage_*>(samples[0]);
                const int npix = static_cast<int>(frame->depth_data._length);

                if (frame->width  != DEPTH_WIDTH  ||
                    frame->height != DEPTH_HEIGHT  ||
                    npix          != DEPTH_PIXELS) {
                    spdlog::warn("[HLIPMdn] Unexpected frame {}x{} len={}",
                                 frame->width, frame->height, npix);
                } else {
                    std::array<float, DEPTH_PIXELS> tmp;
                    for (int i = 0; i < DEPTH_PIXELS; ++i) {
                        float d = frame->depth_data._buffer[i];
                        if (!std::isfinite(d)) d = DEPTH_MAX;
                        tmp[i] = std::clamp(d, DEPTH_MIN, DEPTH_MAX);
                    }
                    std::lock_guard<std::mutex> lk(depth_mutex_);
                    depth_buf_     = tmp;
                    depth_last_ts_ = frame->timestamp_us;
                    if (!depth_received_) {
                        depth_received_ = true;
                        depth_ready_cv_.notify_one();
                    }
                }
                dds_return_loan(dds_reader_, samples, n);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        dds_delete(dds_participant_);
        dds_participant_ = 0;
        dds_reader_      = 0;
    });
}

void State_HLIPMdn::stop_depth_reader()
{
    depth_running_ = false;
    if (depth_thread_.joinable()) depth_thread_.join();
}

// =============================================================================
// Logging
// =============================================================================

void State_HLIPMdn::init_logging()
{
    namespace fs = std::filesystem;

    // Close any previous log file
    if (log_obs_file_.is_open()) log_obs_file_.close();
    log_step_ = 0;

    // Create timestamped run directory: deploy/robots/g1/logs/hlip_mdn_YYYYMMDD_HHMMSS/
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ts;
    ts << std::put_time(&tm, "%Y%m%d_%H%M%S");

    // Place logs next to the executable's config directory
    fs::path base_log_dir = fs::path("logs");
    log_run_dir_   = base_log_dir / ("hlip_mdn_" + ts.str());
    log_depth_dir_ = log_run_dir_ / "depth";

    fs::create_directories(log_depth_dir_);

    // Open JSONL observation log
    auto obs_log_path = log_run_dir_ / "observations.jsonl";
    log_obs_file_.open(obs_log_path, std::ios::out | std::ios::trunc);

    if (log_obs_file_.is_open()) {
        spdlog::info("[HLIPMdn] Logging to: {}", log_run_dir_.string());
    } else {
        spdlog::error("[HLIPMdn] Failed to open log file: {}", obs_log_path.string());
    }
}

void State_HLIPMdn::log_step(
    const std::vector<float>& student_vec,
    const std::vector<float>& depth_obs,
    const std::vector<float>& raw_action)
{
    // ── 1. Write vector observation + action as JSON line ─────────────────────
    if (log_obs_file_.is_open()) {
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        log_obs_file_ << "{\"step\":" << log_step_
                      << ",\"timestamp_us\":" << now_us
                      << ",\"student_vec\":[";
        for (size_t i = 0; i < student_vec.size(); ++i) {
            if (i > 0) log_obs_file_ << ',';
            log_obs_file_ << student_vec[i];
        }
        log_obs_file_ << "],\"raw_action\":[";
        for (size_t i = 0; i < raw_action.size(); ++i) {
            if (i > 0) log_obs_file_ << ',';
            log_obs_file_ << raw_action[i];
        }
        log_obs_file_ << "]}\n";

        // Flush periodically (every 50 steps) to avoid loss on crash
        if (log_step_ % 50 == 0) log_obs_file_.flush();
    }

    // ── 2. Save depth frame as raw binary float32 ────────────────────────────
    {
        std::ostringstream fname;
        fname << "depth_" << std::setfill('0') << std::setw(6) << log_step_ << ".bin";
        auto depth_path = log_depth_dir_ / fname.str();
        std::ofstream df(depth_path, std::ios::binary);
        if (df.is_open()) {
            df.write(reinterpret_cast<const char*>(depth_obs.data()),
                     depth_obs.size() * sizeof(float));
        }
    }

    log_step_++;
}
