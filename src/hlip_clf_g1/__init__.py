from mjlab.tasks.registry import register_mjlab_task
from hlip_clf_g1.rl import HLIPOnPolicyRunner
from hlip_clf_g1.env_cfgs import (
  unitree_g1_hlip_env_cfg,
  unitree_g1_hlip_stepping_stone_env_cfg,
)
from hlip_clf_g1.rl_cfg import unitree_g1_hlip_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Unitree-G1",
  env_cfg=unitree_g1_hlip_env_cfg(),
  play_env_cfg=unitree_g1_hlip_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-SteppingStone-Unitree-G1",
  env_cfg=unitree_g1_hlip_stepping_stone_env_cfg(),
  play_env_cfg=unitree_g1_hlip_stepping_stone_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)


def main() -> None:
    print("Hello from pure-mjlab-code!")
