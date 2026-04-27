#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_HLIPMdn.h"

#include <csignal>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

// Global flag so the signal handler can signal the main loop.
static volatile bool g_shutdown = false;

// ---------------------------------------------------------------------------
// SIGINT / SIGTERM handler
// First Ctrl+C: zero torques (passive/damped), then exit.
// Second Ctrl+C: immediate force-exit.
// ---------------------------------------------------------------------------
static void signal_handler(int sig)
{
    if (g_shutdown) {
        spdlog::warn("Force exit.");
        std::_Exit(1);
    }
    g_shutdown = true;
    spdlog::warn("Signal {} received — commanding Passive and exiting...", sig);

    if (FSMState::lowcmd) {
        for (int i = 0; i < 29; ++i) {
            auto& mc = FSMState::lowcmd->msg_.motor_cmd()[i];
            mc.kp()  = 0;
            mc.kd()  = 2;   // light damping so robot doesn't slam down
            mc.dq()  = 0;
            mc.tau() = 0;
            if (FSMState::lowstate)
                mc.q() = FSMState::lowstate->msg_.motor_state()[i].q();
        }
        // Publish several times to ensure the robot receives it.
        for (int i = 0; i < 5; ++i) {
            FSMState::lowcmd->unlockAndPublish();
            usleep(5000);
        }
    }
    std::exit(0);
}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Register signal handlers before anything else.
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Controls:\n";
    std::cout << "  L2 + Up      → FixStand\n";
    std::cout << "  R2 + B       → HLIPMdn (MDN distillation policy)\n";
    std::cout << "  R2 + A       → Velocity (baseline policy)\n";
    std::cout << "  L2 + B       → Passive  (safe stop, damped)\n";
    std::cout << "  Ctrl+C       → Emergency stop and exit\n";

    while (!g_shutdown)
    {
        sleep(1);
    }

    return 0;
}
