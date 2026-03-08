from hlip_clf_g1.env_cfgs import unitree_g1_hlip_stairs_env_cfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
import torch

def test():
    cfg = unitree_g1_hlip_stairs_env_cfg(play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    obs = env.reset()
    
    heightmap = env.scene.sensors["heightmap"]
    print("Shape:", heightmap.data.hit_pos_w.shape)
    print("Init hit[0,0]:", heightmap.data.hit_pos_w[0, 0])
    
    actions = torch.zeros(2, 12, device=env.device)
    obs, rewards, dones, truncs, infos = env.step(actions)
    print("after step hit[0,0]:", heightmap.data.hit_pos_w[0, 0])
    
if __name__ == "__main__":
    test()
