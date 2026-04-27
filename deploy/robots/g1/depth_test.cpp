/**
 * depth_test.cpp — Standalone depth subscriber diagnostic tool.
 *
 * Uses the raw CycloneDDS C API with the IDL-generated descriptor from
 * DepthImage_.hpp. depth_data is a variable-length sequence[float32].
 *
 * Usage:
 *   ./depth_test [network_interface]
 */

#include "DepthImage_.hpp"
#include <unitree/robot/channel/channel_factory.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <string>
#include <thread>

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

static constexpr const char* RAMP = " .:;=+xX$#";

static void print_ascii(const float* pixels, int w, int h, float dmin, float dmax)
{
    const float range = (dmax > dmin) ? (dmax - dmin) : 1.f;
    spdlog::info("+-{}-+", std::string(w * 2, '-'));
    for (int row = 0; row < h; ++row) {
        std::string line(1, '|');
        for (int col = 0; col < w; ++col) {
            float d = pixels[row * w + col];
            int idx = std::clamp(static_cast<int>((d - dmin) / range * 9.f), 0, 9);
            line += RAMP[idx]; line += RAMP[idx];
        }
        line += '|';
        spdlog::info("{}", line);
    }
    spdlog::info("+-{}-+", std::string(w * 2, '-'));
}

int main(int argc, char** argv)
{
    spdlog::set_pattern("[%H:%M:%S.%e] %v");
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (argc > 1) {
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
        spdlog::info("Network interface: {}", argv[1]);
    } else {
        unitree::robot::ChannelFactory::Instance()->Init(0);
        spdlog::info("Auto-detecting network interface.");
    }

    dds_entity_t participant = dds_create_participant(0, nullptr, nullptr);
    if (participant < 0) {
        spdlog::error("dds_create_participant failed: {}", participant); return 1;
    }

    dds_entity_t topic = dds_create_topic(
        participant, &DepthImage__desc, DEPTH_TOPIC, nullptr, nullptr);
    if (topic < 0) {
        spdlog::error("dds_create_topic failed: {}", topic);
        dds_delete(participant); return 1;
    }

    dds_qos_t* qos = dds_create_qos();
    dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, 0);
    dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 1);
    dds_entity_t reader = dds_create_reader(participant, topic, qos, nullptr);
    dds_delete_qos(qos);
    if (reader < 0) {
        spdlog::error("dds_create_reader failed: {}", reader);
        dds_delete(participant); return 1;
    }

    spdlog::info("Subscribed to '{}'. Waiting for frames (Ctrl-C to stop).", DEPTH_TOPIC);

    using clock = std::chrono::steady_clock;
    auto t_start   = clock::now();
    int  frame_cnt = 0;
    bool show_ascii = true;

    // Use a NULL sample pointer: CycloneDDS allocates and owns the buffer.
    // This is the correct pattern for variable-length sequence types.
    void*             samples[1] = { nullptr };
    dds_sample_info_t si[1];

    while (g_running) {
        samples[0] = nullptr;  // reset each iteration so CycloneDDS loans fresh
        int n = dds_take(reader, samples, si, 1, 1);
        if (n > 0 && si[0].valid_data && samples[0] != nullptr) {
            const DepthImage_* frame = static_cast<const DepthImage_*>(samples[0]);
            const int npix = static_cast<int>(frame->depth_data._length);

            if (frame->width != DEPTH_WIDTH || frame->height != DEPTH_HEIGHT || npix != DEPTH_PIXELS) {
                spdlog::warn("Unexpected frame {}x{} seq_len={}", frame->width, frame->height, npix);
            } else {
                const float* pixels = frame->depth_data._buffer;
                float dmin = 1e9f, dmax = -1e9f, dsum = 0.f;
                int n_inf = 0;
                for (int i = 0; i < npix; ++i) {
                    float d = pixels[i];
                    if (!std::isfinite(d)) { ++n_inf; d = DEPTH_MAX; }
                    if (d < dmin) dmin = d;
                    if (d > dmax) dmax = d;
                    dsum += d;
                }

                int64_t now_us = static_cast<int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                double age_ms  = (now_us - frame->timestamp_us) / 1000.0;
                double elapsed = std::chrono::duration<double>(clock::now() - t_start).count();

                ++frame_cnt;
                spdlog::info("[+{:.3f}s] {}x{}  min={:.3f}m  max={:.3f}m  mean={:.3f}m  age={:.1f}ms  #{}{}",
                    elapsed, (int)frame->width, (int)frame->height,
                    dmin, dmax, dsum / float(npix), age_ms, frame_cnt,
                    n_inf > 0 ? fmt::format("  WARNING: {} non-finite", n_inf) : "");

                if (show_ascii || frame_cnt % 30 == 0) {
                    show_ascii = false;
                    print_ascii(pixels, (int)frame->width, (int)frame->height, dmin, dmax);
                }
            }
            // Return the loan — CycloneDDS frees the internal buffer
            dds_return_loan(reader, samples, n);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    spdlog::info("Stopped after {} frames.", frame_cnt);
    dds_delete(participant);
    return 0;
}
