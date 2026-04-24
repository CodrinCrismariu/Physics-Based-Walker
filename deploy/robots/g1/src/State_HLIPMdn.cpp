/**
 * State_HLIPMdn.cpp — CNNTransformerMDN distillation student deploy state.
 *
 * Depth subscription uses CycloneDDS C API (dds.h) with a hand-crafted
 * dds_topic_descriptor_t, avoiding the C++ typed API that requires
 * IDL-generated serialisation traits.
 *
 * Wire format of DepthImage_ (Python @final @autoid("sequential") IDL):
 *   offset  0 : int64  timestamp_us
 *   offset  8 : int32  width
 *   offset 12 : int32  height
 *   offset 16 : float  depth_data[768]   (total frame = 3088 bytes)
 */

#include "State_HLIPMdn.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "param.h"

#include <dds/dds.h>
#include <dds/ddsc/dds_opcodes.h>

#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <cstring>

// =============================================================================
// CycloneDDS C descriptor for DepthImage_
//
// IDL equivalent:
//   @final
//   @autoid(SEQUENTIAL)
//   struct DepthImage_ {
//       long long  timestamp_us;  // 8 bytes, offset 0
//       long       width;         // 4 bytes, offset 8
//       long       height;        // 4 bytes, offset 12
//       float      depth_data[768]; // 3072 bytes, offset 16
//   };
//
// Ops encoding (CycloneDDS dds_opcodes.h):
//   DDS_OP_ADR | DDS_OP_TYPE_8BY  → 8-byte field (long long)
//   DDS_OP_ADR | DDS_OP_TYPE_4BY  → 4-byte field (long / float)
//   DDS_OP_ADR | DDS_OP_TYPE_ARR  → fixed-size array
//     followed by: element count, element subtype
//   DDS_OP_FLAG_FP  → floating-point flag (bit 1)
//   DDS_OP_FLAG_SGN → signed flag (bit 2)
//   DDS_OP_RTS      → end of struct
// =============================================================================

static const uint32_t DepthImage_ops[] = {
    /* timestamp_us: int64, offset 0 */
    DDS_OP_ADR | DDS_OP_TYPE_8BY | DDS_OP_FLAG_SGN,
    offsetof(DepthFrameRaw, timestamp_us),

    /* width: int32, offset 8 */
    DDS_OP_ADR | DDS_OP_TYPE_4BY | DDS_OP_FLAG_SGN,
    offsetof(DepthFrameRaw, width),

    /* height: int32, offset 12 */
    DDS_OP_ADR | DDS_OP_TYPE_4BY | DDS_OP_FLAG_SGN,
    offsetof(DepthFrameRaw, height),

    /* depth_data: float[768], offset 16
     * Format: DDS_OP_ADR|ARR, offset, count, element-subtype */
    DDS_OP_ADR | DDS_OP_TYPE_ARR | DDS_OP_FLAG_FP,
    offsetof(DepthFrameRaw, depth_data),
    768,
    DDS_OP_SUBTYPE_4BY | DDS_OP_FLAG_FP,

    DDS_OP_RTS
};

static const dds_topic_descriptor_t DepthImage_desc = {
    sizeof(DepthFrameRaw),                          /* m_size   */
    alignof(DepthFrameRaw),                         /* m_align  */
    DDS_TOPIC_FIXED_SIZE | DDS_TOPIC_FIXED_KEY,     /* m_flagset */
    0,                                              /* m_nkeys  */
    "DepthImage_",                                  /* m_typename */
    nullptr,                                        /* m_keys   */
    9,                                              /* m_nops   */
    DepthImage_ops,                                 /* m_ops    */
    "",                                             /* m_meta   */
};

// =============================================================================
// Stub observations (the policy thread builds obs manually; these are
// registered so the ObservationManager doesn't error on the YAML groups)
// =============================================================================

namespace isaaclab
{

REGISTER_OBSERVATION(depth_image)
{
    (void)env; (void)params;
    // Real depth is injected by State_HLIPMdn::get_depth_obs().
    return std::vector<float>(DEPTH_PIXELS, DEPTH_MAX);
}

REGISTER_OBSERVATION(hlip_mdn_velocity_commands)
{
    // Joystick-driven [vx, vy, 0] × 2.0.  Read in build_student_vec() too;
    // this stub lets the ObservationManager initialise without errors.
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
        [&]()->bool{ return isaaclab::mdp::bad_orientation(env_.get(), 1.0); },
        FSMStringMap.right.at("Passive")
    ));
}

void State_HLIPMdn::enter()
{
    for (int i = 0; i < (int)env_->robot->data.joint_stiffness.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = env_->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env_->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    start_depth_subscriber();

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
            obs_map["student_vec"]        = build_student_vec();
            obs_map["head_camera_depth"]  = get_depth_obs();

            auto action = env_->alg->act(obs_map);
            env_->action_manager->process_action(action);
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

    std::vector<float> obs;
    obs.reserve(96);

    // base_ang_vel [3]
    for (int i = 0; i < 3; ++i) obs.push_back(imu.gyroscope()[i]);

    // projected_gravity [3]: g_b = R^T * [0,0,-1]
    Eigen::Matrix3f R = data.root_quat_w.toRotationMatrix();
    Eigen::Vector3f g = R.transpose() * Eigen::Vector3f(0.f, 0.f, -1.f);
    for (int i = 0; i < 3; ++i) obs.push_back(g[i]);

    // velocity_commands [3]: [vx, vy, 0] × 2.0 from joystick
    static auto cmd_cfg = env_->cfg["commands"]["hlip_mdn"]["ranges"];
    const float vx_max = cmd_cfg["lin_vel_x"][1].as<float>();
    const float vy_max = cmd_cfg["lin_vel_y"][1].as<float>();
    const auto& joy = FSMState::lowstate->joystick;
    // The SDK's Axis::operator()() is non-const; use const_cast as a workaround.
    auto& joy_mut = const_cast<std::remove_const_t<std::remove_reference_t<decltype(joy)>>&>(joy);
    obs.push_back(joy_mut.ly() * vx_max * 2.0f);
    obs.push_back(0.0f); // -joy_mut.lx() * vy_max * 2.0f); // No Lateral Velocity for Corridor Testing
    obs.push_back(0.0f);

    // joint_pos_rel [29]
    const int N = (int)data.joint_ids_map.size();
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].q() - data.default_joint_pos[i]);

    // joint_vel_rel [29] × 0.05
    for (int i = 0; i < N; ++i)
        obs.push_back(motors[(int)data.joint_ids_map[i]].dq() * 0.05f);

    // last_action [29]
    const auto& act = env_->action_manager->processed_actions();
    for (int i = 0; i < N; ++i)
        obs.push_back(i < (int)act.size() ? act[i] : 0.0f);

    return obs;
}

std::vector<float> State_HLIPMdn::get_depth_obs()
{
    std::lock_guard<std::mutex> lk(depth_mutex_);
    return std::vector<float>(depth_buf_.begin(), depth_buf_.end());
}

// =============================================================================
// CycloneDDS C API depth subscriber
// =============================================================================

void State_HLIPMdn::start_depth_subscriber()
{
    depth_running_ = true;
    {
        std::lock_guard<std::mutex> lk(depth_mutex_);
        depth_buf_.fill(DEPTH_MAX);
    }

    depth_thread_ = std::thread([this] {
        // Use domain 0 — same as the unitree_sdk2 network.
        dds_participant_ = dds_create_participant(0, nullptr, nullptr);
        if (dds_participant_ < 0) {
            spdlog::error("[HLIPMdn] dds_create_participant failed: {}", dds_participant_);
            return;
        }

        dds_topic_ = dds_create_topic(
            dds_participant_, &DepthImage_desc,
            "rt/body_depth_camera", nullptr, nullptr);
        if (dds_topic_ < 0) {
            spdlog::error("[HLIPMdn] dds_create_topic failed: {}", dds_topic_);
            dds_delete(dds_participant_);
            return;
        }

        dds_qos_t* qos = dds_create_qos();
        dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, 0);
        dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 1);
        dds_reader_ = dds_create_reader(dds_participant_, dds_topic_, qos, nullptr);
        dds_delete_qos(qos);
        if (dds_reader_ < 0) {
            spdlog::error("[HLIPMdn] dds_create_reader failed: {}", dds_reader_);
            dds_delete(dds_participant_);
            return;
        }

        spdlog::info("[HLIPMdn] Depth subscriber ready on 'rt/body_depth_camera'.");

        DepthFrameRaw frame;
        void* samples[1] = { &frame };
        dds_sample_info_t si[1];

        while (depth_running_) {
            int n = dds_take(dds_reader_, samples, si, 1, 1);
            if (n > 0 && si[0].valid_data) {
                if (frame.width == DEPTH_WIDTH && frame.height == DEPTH_HEIGHT) {
                    std::array<float, DEPTH_PIXELS> tmp;
                    for (int i = 0; i < DEPTH_PIXELS; ++i) {
                        float d = frame.depth_data[i];
                        if (!std::isfinite(d)) d = DEPTH_MAX;
                        tmp[i] = std::clamp(d, DEPTH_MIN, DEPTH_MAX);
                    }
                    std::lock_guard<std::mutex> lk(depth_mutex_);
                    depth_buf_ = tmp;
                    depth_received_ = true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        dds_delete(dds_participant_);  // cascade-deletes topic + reader
    });
}

void State_HLIPMdn::stop_depth_subscriber()
{
    depth_running_ = false;
    if (depth_thread_.joinable()) depth_thread_.join();
}
