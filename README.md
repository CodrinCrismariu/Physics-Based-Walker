To train: uv run train Mjlab-HLIP-CLF-Unitree-G1 --env.scene.num-envs 4096
To train simple stepping stones: uv run train Mjlab-HLIP-CLF-Simple-Stepping-Stones-Unitree-G1 --env.scene.num-envs 4096
To train two-platform stepping corridor: uv run train Mjlab-HLIP-CLF-Two-Platform-Stepping-Corridor-Unitree-G1 --env.scene.num-envs 4096

To play: uv run play Mjlab-HLIP-CLF-Unitree-G1
To play simple stepping stones: uv run play Mjlab-HLIP-CLF-Simple-Stepping-Stones-Unitree-G1
To play two-platform stepping corridor: uv run play Mjlab-HLIP-CLF-Two-Platform-Stepping-Corridor-Unitree-G1

Run Distillation-MDN on two-platform stepping corridor:
uv run custom_train Mjlab-HLIP-CLF-Distillation-MDN-Two-Platform-Stepping-Corridor-Unitree-G1

Fine-tune Distillation-MDN corridor student with PPO:
uv run custom_train Mjlab-HLIP-CLF-PPO-Finetune-MDN-Two-Platform-Stepping-Corridor-Unitree-G1 \
  --agent.resume True \
  --agent.load-run logs/rsl_rl/g1_hlip_clf_distillation_mdn/<run_id>/model_*.pt \
  --env.scene.num-envs 4096

HLIP + MPC internals: see src/hlip_clf_g1/mdp/README_HLIP_MPC.md

Run Distillation: 
uv run custom_train Mjlab-HLIP-CLF-Distillation-Stairs-Unitree-G1 \
  --agent.load-run logs/rsl_rl/g1_hlip_clf/2026-03-16_20-12-12/ \
  --student-load-run logs/rsl_rl/g1_hlip_clf_distillation/2026-03-17_11-04-00/model_200.pt \
  --env.scene.num-envs 4096
  
updated xml from: https://huggingface.co/lerobot/unitree-g1-mujoco/blob/main/assets/g1_29dof_no_hand.xml