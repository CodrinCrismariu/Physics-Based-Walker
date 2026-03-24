"""Script to train RL agent with RSL-RL."""

import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from hlip_clf_g1.rl import RslRlDistillationRunnerCfg, RslRlDistillationFineTuneRunnerCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg | RslRlDistillationRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  student_load_run: str | None = None
  """Optional student run directory/checkpoint path to load for distillation."""
  student_load_checkpoint: str = "model_.*.pt"
  """Regex used when student_load_run points to a directory of checkpoints."""
  student_wandb_run_path: str | None = None
  """Optional W&B run path used to initialize student weights."""
  student_wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the student W&B run to load."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg | RslRlDistillationRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _load_state_dict_by_shape(model, source_state_dict: dict) -> int:
  """Load matching parameters by key and shape, leaving others untouched.

  Returns the number of parameters copied.
  """
  target_state_dict = model.state_dict()
  merged_state_dict = dict(target_state_dict)
  loaded_count = 0
  for key, value in source_state_dict.items():
    if key in merged_state_dict and merged_state_dict[key].shape == value.shape:
      merged_state_dict[key] = value
      loaded_count += 1
  model.load_state_dict(merged_state_dict, strict=False)
  return loaded_count


def _bootstrap_distillation_from_rl_checkpoint(runner, checkpoint_path: Path) -> bool:
  """Initialize distillation student/teacher from a PPO checkpoint.

  This is used when direct ``runner.load`` fails because the privileged teacher
  uses a different observation dimension than the PPO actor checkpoint.
  """
  import torch

  loaded_dict = torch.load(str(checkpoint_path), map_location="cpu")
  actor_state_dict = loaded_dict.get("actor_state_dict")
  if actor_state_dict is None:
    return False

  alg = getattr(runner, "alg", None)
  student = getattr(alg, "student", None)
  teacher = getattr(alg, "teacher", None)
  if student is None or teacher is None:
    return False

  # Student receives actor observations, so actor weights are generally compatible.
  _load_state_dict_by_shape(student, actor_state_dict)

  # Teacher uses privileged observations. Actor and critic checkpoints may have
  # different structures; apply shape-compatible subsets from both.
  teacher_loaded_from_actor = _load_state_dict_by_shape(teacher, actor_state_dict)

  teacher_loaded_from_critic = 0
  critic_state_dict = loaded_dict.get("critic_state_dict")
  if critic_state_dict is not None:
    teacher_loaded_from_critic = _load_state_dict_by_shape(teacher, critic_state_dict)

  alg.teacher_loaded = True
  print(
    "[INFO] Distillation bootstrap loaded teacher params by shape: "
    f"actor={teacher_loaded_from_actor}, critic={teacher_loaded_from_critic}."
  )
  return True


def _resolve_local_checkpoint_path(
  log_root_path: Path,
  load_run: str,
  load_checkpoint: str,
) -> Path:
  """Resolve a checkpoint from a local path, run folder, or regex run id."""
  load_run_path = Path(load_run).expanduser()
  if not load_run_path.is_absolute():
    load_run_path = (Path.cwd() / load_run_path).resolve()

  if load_run_path.exists():
    if load_run_path.is_file():
      return load_run_path
    if load_run_path.is_dir():
      model_checkpoints = [
        f.name for f in load_run_path.iterdir() if re.match(load_checkpoint, f.name)
      ]
      if len(model_checkpoints) == 0:
        raise ValueError(
          f"No checkpoint found in {load_run_path} matching {load_checkpoint}"
        )
      model_checkpoints.sort(key=lambda m: f"{m:0>15}")
      return load_run_path / model_checkpoints[-1]

  return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  is_distillation = isinstance(cfg.agent, RslRlDistillationRunnerCfg)
  if not is_distillation and (
    cfg.student_load_run is not None or cfg.student_wandb_run_path is not None
  ):
    raise ValueError(
      "--student-load-run/--student-wandb-run-path are only supported "
      "for distillation tasks."
    )

  resume_path: Path | None = None
  student_resume_path: Path | None = None
  should_resume = cfg.agent.resume
  if is_distillation and not should_resume:
    # Distillation requires teacher weights; infer resume intent when a source is provided.
    should_resume = cfg.wandb_run_path is not None or cfg.agent.load_run != ".*"
    if should_resume and rank == 0:
      print("[INFO] Enabling resume for distillation to load teacher checkpoint.")

  if should_resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = _resolve_local_checkpoint_path(
        log_root_path,
        cfg.agent.load_run,
        cfg.agent.load_checkpoint,
      )

  if is_distillation:
    if cfg.student_wandb_run_path is not None:
      student_resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path,
        Path(cfg.student_wandb_run_path),
        cfg.student_wandb_checkpoint_name,
      )
      if rank == 0:
        run_id = student_resume_path.parent.name
        checkpoint_name = student_resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading student checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    elif cfg.student_load_run is not None:
      student_resume_path = _resolve_local_checkpoint_path(
        log_root_path,
        cfg.student_load_run,
        cfg.student_load_checkpoint,
      )

    if student_resume_path is not None and resume_path is None and rank == 0:
      print(
        "[WARN] Student checkpoint configured without a teacher checkpoint "
        "source. Teacher will remain randomly initialized."
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  if isinstance(cfg.agent, RslRlDistillationFineTuneRunnerCfg):
    stance_source = getattr(cfg.agent, "stance_refeed_source", "prediction")
    if stance_source == "mpc":
      cfg.agent.actor.class_name = "hlip_clf_g1.rl.stance_model:StanceMpcInputMLPModel"
    else:
      cfg.agent.actor.class_name = "hlip_clf_g1.rl.stance_model:StanceRefeedMLPModel"
    if rank == 0:
      print(f"[INFO] Fine-tune stance refeed source: {stance_source}.")

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    try:
      runner.load(str(resume_path))
    except RuntimeError as err:
      if not is_distillation:
        raise
      if _bootstrap_distillation_from_rl_checkpoint(runner, resume_path):
        print(
          "[INFO] Direct distillation checkpoint load was incompatible with "
          "privileged teacher observations. Bootstrapped student/teacher "
          "weights from PPO actor/critic tensors by key/shape."
        )
      else:
        raise RuntimeError(
          "Failed to load teacher weights for distillation. Provide a compatible "
          "distillation checkpoint via --agent.load-run or a PPO checkpoint with "
          "actor_state_dict/critic_state_dict."
        ) from err

  if student_resume_path is not None:
    print(f"[INFO]: Loading distillation student checkpoint from: {student_resume_path}")
    try:
      runner.load(
        str(student_resume_path),
        load_cfg={
          "student": True,
          "teacher": False,
          "optimizer": False,
          "iteration": False,
        },
      )
      print("[INFO] Loaded distillation student weights.")
    except (RuntimeError, KeyError) as err:
      raise RuntimeError(
        "Failed to load student weights. Provide a compatible distillation "
        "checkpoint containing student_state_dict."
      ) from err

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
