#pragma once

/**
 * State_HLIPMdn — Deploy state for the CNNTransformerMDN distillation student.
 *
 * Depth subscription uses the raw CycloneDDS C API with the IDL-generated
 * descriptor in DepthImage_.hpp (correct XTypes type hash, matching the Python
 * unitree_sdk2py ChannelPublisher writer).
 *
 * ONNX inputs:
 *   "student_vec"        [1, 480]       – proprioception + velocity command (96 × 5 history)
 *   "head_camera_depth"  [1, 1, 24, 32] – depth image (metres)
 */

#include "FSM/State_RLBase.h"
#include "DepthImage_.hpp"

#include <atomic>
#include <array>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

// History buffer constants — must match training ObservationGroupCfg.
static constexpr int OBS_HISTORY_LENGTH = 5;  // history_length=5
static constexpr int RAW_VEC_DIM = 96;        // per-frame student_vec before history
static constexpr int HIST_VEC_DIM = RAW_VEC_DIM * OBS_HISTORY_LENGTH;  // 480

// Per-term dimensions (in order) for term-major history flattening.
// base_ang_vel(3), projected_gravity(3), velocity_commands(3),
// joint_pos(29), joint_vel(29), actions(29)
static constexpr int TERM_DIMS[] = {3, 3, 3, 29, 29, 29};
static constexpr int NUM_TERMS = 6;

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
        // Close log file
        if (log_obs_file_.is_open()) {
            log_obs_file_.close();
        }
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

    // Per-term observation history buffers (oldest at front, newest at back).
    // Each entry is one frame of that term's obs, with scale already applied.
    // Mirrors simulation CircularBuffer(max_len=5) per ObservationTermCfg.
    std::deque<std::vector<float>> obs_history_[NUM_TERMS];

    // CycloneDDS C API handles for the depth reader
    dds_entity_t dds_participant_ = 0;
    dds_entity_t dds_reader_      = 0;
    std::thread  depth_thread_;
    std::atomic<bool> depth_running_{false};

    void start_depth_reader();
    void stop_depth_reader();

    // Build raw 96-d proprioception vector (single frame, with scales applied).
    std::vector<float> build_raw_obs_frame();
    // Push raw frame into per-term history buffers and return 480-d flattened output.
    std::vector<float> build_student_vec();
    std::vector<float> get_depth_obs();

    // ── Logging ──────────────────────────────────────────────────────────────
    std::filesystem::path log_run_dir_;       // per-run directory
    std::filesystem::path log_depth_dir_;     // depth images subdirectory
    std::ofstream         log_obs_file_;      // JSONL observation log
    int                   log_step_ = 0;      // step counter
    // Hz tracking
    int                   hz_step_count_ = 0;
    std::chrono::high_resolution_clock::time_point hz_window_start_;

    void init_logging();
    void log_step(const std::vector<float>& student_vec,
                  const std::vector<float>& depth_obs,
                  const std::vector<float>& raw_action);
};

REGISTER_FSM(State_HLIPMdn)
