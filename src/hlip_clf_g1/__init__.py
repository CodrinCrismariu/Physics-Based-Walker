from mjlab.tasks.registry import register_mjlab_task
from hlip_clf_g1.rl import (
  HLIPOnPolicyRunner,
  HLIPDistilledOnPolicyRunner,
)
from hlip_clf_g1.env_cfgs import (
  unitree_g1_hlip_env_cfg,
  unitree_g1_hlip_random_box_grid_env_cfg,
  unitree_g1_hlip_simple_stepping_stone_env_cfg,
  unitree_g1_hlip_two_platform_stepping_corridor_env_cfg,
  unitree_g1_hlip_random_box_grid_teacher_env_cfg,
  unitree_g1_hlip_stairs_env_cfg,
  unitree_g1_hlip_distillation_stepping_stone_env_cfg,
  unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg,
  unitree_g1_distillation_hlip_stairs_env_cfg,
  unitree_g1_hlip_distillation_env_cfg,
)
from hlip_clf_g1.rl_cfg import (
  unitree_g1_hlip_ppo_runner_cfg,
  unitree_g1_hlip_corridor_ppo_from_distillation_mdn_runner_cfg,
  unitree_g1_hlip_distillation_runner_cfg,
  unitree_g1_hlip_distillation_mdn_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Unitree-G1",
  env_cfg=unitree_g1_hlip_env_cfg(),
  play_env_cfg=unitree_g1_hlip_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Stairs-Unitree-G1",
  env_cfg=unitree_g1_hlip_stairs_env_cfg(),
  play_env_cfg=unitree_g1_hlip_stairs_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Simple-Stepping-Stones-Unitree-G1",
  env_cfg=unitree_g1_hlip_simple_stepping_stone_env_cfg(),
  play_env_cfg=unitree_g1_hlip_simple_stepping_stone_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Random-Box-Grid-Unitree-G1",
  env_cfg=unitree_g1_hlip_random_box_grid_env_cfg(),
  play_env_cfg=unitree_g1_hlip_random_box_grid_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Two-Platform-Stepping-Corridor-Unitree-G1",
  env_cfg=unitree_g1_hlip_two_platform_stepping_corridor_env_cfg(),
  play_env_cfg=unitree_g1_hlip_two_platform_stepping_corridor_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Random-Box-Grid-Teacher-Unitree-G1",
  env_cfg=unitree_g1_hlip_random_box_grid_teacher_env_cfg(),
  play_env_cfg=unitree_g1_hlip_random_box_grid_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_ppo_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-PPO-Finetune-MDN-Two-Platform-Stepping-Corridor-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_corridor_ppo_from_distillation_mdn_runner_cfg(),
  runner_cls=HLIPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-Stepping-Stones-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_stepping_stone_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_stepping_stone_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-Stairs-Unitree-G1",
  env_cfg=unitree_g1_distillation_hlip_stairs_env_cfg(),
  play_env_cfg=unitree_g1_distillation_hlip_stairs_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-MDN-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_mdn_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-MDN-Stepping-Stones-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_stepping_stone_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_stepping_stone_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_mdn_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-MDN-Two-Platform-Stepping-Corridor-Unitree-G1",
  env_cfg=unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg(),
  play_env_cfg=unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_mdn_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HLIP-CLF-Distillation-MDN-Stairs-Unitree-G1",
  env_cfg=unitree_g1_distillation_hlip_stairs_env_cfg(),
  play_env_cfg=unitree_g1_distillation_hlip_stairs_env_cfg(play=True),
  rl_cfg=unitree_g1_hlip_distillation_mdn_runner_cfg(),
  runner_cls=HLIPDistilledOnPolicyRunner,
)

def main() -> None:
    print("Hello from pure-mjlab-code!")
