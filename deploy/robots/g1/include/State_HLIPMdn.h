#pragma once

/**
 * State_HLIPMdn — Deploy state for the CNNTransformerMDN distillation student.
 *
 * Depth frames are received from the external Python publisher via CycloneDDS
 * C API (dds.h), using raw void* buffers.  This avoids all IDL code-generation
 * requirements that the C++ ddscxx typed API imposes.
 *
 * ONNX inputs
 *   "student_vec"        [1, 96]        – proprioception + velocity command
 *   "head_camera_depth"  [1, 1, 24, 32] – depth image (metres)
 *
 * student_vec layout (matches training make_hlip_distillation_env_cfg):
 *   [0:3]   base_ang_vel
 *   [3:6]   projected_gravity
 *   [6:9]   velocity_commands * 2.0  (vx, vy, 0 — yaw zeroed)
 *   [9:38]  joint_pos_rel
 *   [38:67] joint_vel_rel * 0.05
 *   [67:96] last_action
 */

#include "FSM/State_RLBase.h"

#include <dds/dds.h>   // CycloneDDS C API — no IDL traits needed

#include <atomic>
#include <array>
#include <mutex>
#include <thread>

// ---------------------------------------------------------------------------
// Depth image constants
// ---------------------------------------------------------------------------
static constexpr int  DEPTH_WIDTH  = 32;
static constexpr int  DEPTH_HEIGHT = 24;
static constexpr int  DEPTH_PIXELS = DEPTH_WIDTH * DEPTH_HEIGHT;   // 768
static constexpr float DEPTH_MIN   = 0.1f;
static constexpr float DEPTH_MAX   = 10.0f;

// Wire layout matching the Python DepthImage_ IDL struct
// (@final @autoid("sequential") → plain CDR, fields in order).
#pragma pack(push, 1)
struct DepthFrameRaw {
    int64_t  timestamp_us;
    int32_t  width;
    int32_t  height;
    float    depth_data[DEPTH_PIXELS];
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// State_HLIPMdn
// ---------------------------------------------------------------------------
class State_HLIPMdn : public FSMState
{
public:
    State_HLIPMdn(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running_ = false;
        if (policy_thread_.joinable()) policy_thread_.join();
        stop_depth_subscriber();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;

    std::thread policy_thread_;
    bool policy_thread_running_ = false;

    // Depth ring-buffer written by DDS thread, read by policy thread
    mutable std::mutex depth_mutex_;
    std::array<float, DEPTH_PIXELS> depth_buf_{};
    bool depth_received_ = false;

    // CycloneDDS C API handles
    dds_entity_t dds_participant_ = 0;
    dds_entity_t dds_topic_      = 0;
    dds_entity_t dds_reader_     = 0;
    std::thread  depth_thread_;
    std::atomic<bool> depth_running_{false};

    void start_depth_subscriber();
    void stop_depth_subscriber();

    std::vector<float> build_student_vec();
    std::vector<float> get_depth_obs();
};

REGISTER_FSM(State_HLIPMdn)
