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
        const auto dt = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(env_->step_dt));
        auto sleep_till = clock::now() + dt;

        while (policy_thread_running_) {
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

            env_->action_manager->process_action(raw_action);
            env_->episode_length += 1;

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
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

std::vector<float> State_HLIPMdn::build_student_vec()
{
    using G1 = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1* robot = dynamic_cast<G1*>(env_->robot.get());
    const auto& motors = robot->lowstate->msg_.motor_state();
    const auto& imu    = robot->lowstate->msg_.imu_state();
    const auto& data   = env_->robot->data;
    const int N = (int)data.joint_ids_map.size(); // 29

    std::vector<float> obs;
    obs.reserve(96);

    // [0:3] base_ang_vel — IMU gyroscope (rad/s), scale 1.0
    for (int i = 0; i < 3; ++i) obs.push_back(imu.gyroscope()[i]);

    // [3:6] projected_gravity — gravity vector expressed in robot base frame
    // g_base = R_base_world^T * [0, 0, -1]
    // IMU quaternion convention: (w, x, y, z)
    {
        const auto& q = imu.quaternion();
        Eigen::Quaternionf quat(q[0], q[1], q[2], q[3]);
        Eigen::Vector3f g_base = quat.toRotationMatrix().transpose() *
                                 Eigen::Vector3f(0.f, 0.f, -1.f);
        for (int i = 0; i < 3; ++i) obs.push_back(g_base[i]);
    }

    // [6:9] velocity_commands: [vx, 0, 0] * 2.0
    // vx = joy.ly() * 0.5, vy = 0, yaw = 0.
    // Scaled x2: ObservationTermCfg(scale=(2.0, 2.0, 2.0)).
    {
        constexpr float VX_MAX = 0.5f;
        const float vx = FSMState::lowstate->joystick.ly() * VX_MAX;
        obs.push_back(vx * 2.0f);  // vx in [-1, 1]
        obs.push_back(0.0f);        // vy = 0
        obs.push_back(0.0f);        // yaw_rate = 0
    }

    // [9:38] joint_pos — q - default_joint_pos, scale 1.0
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].q() - data.default_joint_pos[i]);

    // [38:67] joint_vel — dq * 0.05, scale 0.05
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].dq() * 0.05f);

    // [67:96] last_action — RAW network output (before scale/offset), scale 1.0
    // Training: ObservationTermCfg(func=mdp.last_action)
    // mdp.last_action returns the raw policy output stored before apply_actions().
    {
        std::lock_guard<std::mutex> lk(last_action_mutex_);
        for (int i = 0; i < N; ++i)
            obs.push_back(i < (int)last_raw_action_.size() ? last_raw_action_[i] : 0.f);
    }

    return obs;
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
