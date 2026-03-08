from hlip_clf_g1.env_cfgs import unitree_g1_hlip_stairs_env_cfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
import torch

def test():
    cfg = unitree_g1_hlip_stairs_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    obs = env.reset()
    
    cmd = env.command_manager.get_term("hlip")
    # Step a few times to get past init
    actions = torch.zeros(1, 12, device=env.device)
    for _ in range(5):
        env.step(actions)
        
    print("z_sw_neg actual:", cmd.y_out[0, 8]) # Wait, foot_pos is com (3) + pelv_eul (3) + foot_pos (3) => indices 6, 7, 8
    
    print("foot_pos_z in act_traj:", cmd.y_act[0, 8])
    
if __name__ == "__main__":
    test()
