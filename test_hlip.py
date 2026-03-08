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
        
    print("foot_target:", cmd.foot_target)
    
    # We want to see what closest_z_w and z_sw_neg are!
    hm = env.scene.sensors["heightmap"].data.hit_pos_w
    
    local_target = torch.cat([cmd.foot_target[:, :2], torch.zeros(1, 1, device=env.device)], dim=1)
    try:
        from hlip_clf_g1.mdp.hlip_command import yaw_quat, quat_apply
        world_target_pos = cmd.stance_foot_pos_0 + quat_apply(yaw_quat(cmd.stance_foot_ori_quat_0), local_target)
    except:
        world_target_pos = cmd.stance_foot_pos_0 + local_target # hack for testing
        
    diffs = hm[:, :, :2] - world_target_pos.unsqueeze(1)[:, :, :2]
    dist_sq = (diffs ** 2).sum(dim=-1)
    closest_indices = dist_sq.argmin(dim=-1)
    closest_z_w = torch.gather(hm[:, :, 2], 1, closest_indices.unsqueeze(1)).squeeze(1)
    z_sw_neg = closest_z_w - cmd.stance_foot_pos_0[:, 2]
    
    print("stance Z:", cmd.stance_foot_pos_0[:, 2])
    print("closest_z_w:", closest_z_w)
    print("z_sw_neg:", z_sw_neg)
    print("cfg.z_sw_min:", cmd.cfg.z_sw_min)
    
if __name__ == "__main__":
    test()
