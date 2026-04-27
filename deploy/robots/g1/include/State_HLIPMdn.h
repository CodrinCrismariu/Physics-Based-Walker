#pragma once

/**
 * State_HLIPMdn — Deploy state for the CNNTransformerMDN distillation student.
 *
 * Depth subscription uses the raw CycloneDDS C API with the IDL-generated
 * descriptor in DepthImage_.hpp (correct XTypes type hash, matching the Python
 * unitree_sdk2py ChannelPublisher writer).
 *
 * ONNX inputs:
 *   "student_vec"        [1, 96]        – proprioception + velocity command
 *   "head_camera_depth"  [1, 1, 24, 32] – depth image (metres)
 */

#include "FSM/State_RLBase.h"
#include "DepthImage_.hpp"

#include <atomic>
#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

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
        stop_depth_reader();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;

    std::thread policy_thread_;
    bool policy_thread_running_ = false;
    bool abort_to_passive_      = false;

    // Depth buffer — updated by DDS reader thread, consumed by policy thread
    mutable std::mutex              depth_mutex_;
    std::condition_variable         depth_ready_cv_;
    std::array<float, DEPTH_PIXELS> depth_buf_{};
    bool                            depth_received_ = false;
    int64_t                         depth_last_ts_  = 0;

    // Raw network output cache — set by policy thread before process_action(),
    // read by build_student_vec() for the last_action observation.
    mutable std::mutex      last_action_mutex_;
    std::vector<float>      last_raw_action_;

    // CycloneDDS C API handles for the depth reader
    dds_entity_t dds_participant_ = 0;
    dds_entity_t dds_reader_      = 0;
    std::thread  depth_thread_;
    std::atomic<bool> depth_running_{false};

    void start_depth_reader();
    void stop_depth_reader();

    std::vector<float> build_student_vec();
    std::vector<float> get_depth_obs();
};

REGISTER_FSM(State_HLIPMdn)
