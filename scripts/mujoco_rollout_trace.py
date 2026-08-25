# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Trace one MuJoCo rollout step-by-step for sim2sim divergence diagnosis."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one MuJoCo rollout and record per-control-step tracking data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to trace.")
    parser.add_argument("--motion-file", required=True, help="MuJoCo motion file to track.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for trace.csv and trace.png. Defaults to "
            "results/mujoco_rollout_trace/<checkpoint-stem>."
        ),
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Optional runtime EMA alpha applied to actions before env.step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on control steps. By default the full motion is traced.",
    )
    return parser


parser = create_parser()
args, _unknown_args = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

_AppLauncher = import_simulator_before_torch("mujoco")

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def resolve_output_dir(output_dir: str | None, checkpoint: str, repo_root: Path) -> Path:
    if output_dir is None:
        output_path = Path("results") / "mujoco_rollout_trace" / Path(checkpoint).stem
    else:
        output_path = Path(output_dir)

    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def load_agent_and_env(args: argparse.Namespace) -> tuple[Any, Any]:
    checkpoint = Path(args.checkpoint)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if current_simulator != "mujoco":
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to 'mujoco' (trace)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="mujoco",
            robot_config=robot_config,
        )

    simulator_config.num_envs = 1
    simulator_config.headless = True
    motion_lib_config.motion_file = args.motion_file

    fabric_config = FabricConfig(
        accelerator="cpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
    )

    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=checkpoint.parent,
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False)
    agent.eval()
    return agent, env


def action_delta_mean(action: torch.Tensor, previous: torch.Tensor | None) -> float:
    if previous is None:
        return 0.0
    return (action - previous).abs().mean().item()


def collect_trace(agent: Any, env: Any, ema_alpha: float | None, max_steps: int | None):
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.motion_manager.motion_ids[env_ids] = 0
    env.motion_manager.motion_times[env_ids] = 0.0

    obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    motion_lengths = env.motion_lib.get_motion_length(env.motion_manager.motion_ids[env_ids])
    motion_steps = int(torch.floor(motion_lengths.max() / env.dt).item())
    rollout_steps = motion_steps if max_steps is None else min(max_steps, motion_steps)

    rows = []
    previous_raw_action = None
    previous_applied_action = None
    previous_processed_action = None

    with torch.no_grad():
        for step_idx in range(rollout_steps):
            model_outs = agent.model(obs_td)
            raw_action = model_outs.get("mean_action", model_outs.get("action"))
            raw_delta = action_delta_mean(raw_action, previous_raw_action)

            applied_action = raw_action
            if ema_alpha is not None:
                if previous_applied_action is None:
                    previous_applied_action = raw_action.clone()
                applied_action = (
                    ema_alpha * raw_action + (1.0 - ema_alpha) * previous_applied_action
                )
                previous_applied_action = applied_action.clone()

            obs, _rewards, _dones, _terminated, _extras = env.step(applied_action)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            ctx = env.context
            per_body_error = (
                ctx.current.rigid_body_pos - ctx.mimic.ref_state.rigid_body_pos
            ).norm(dim=-1)
            processed_action = env._current_processed_action.clone()
            processed_delta = action_delta_mean(
                processed_action,
                previous_processed_action,
            )

            rows.append(
                {
                    "step": step_idx,
                    "sim_time": env.motion_manager.motion_times[0].item(),
                    "tracking_error_mean": per_body_error[0].mean().item(),
                    "tracking_error_max": per_body_error[0].max().item(),
                    "root_height": ctx.current.rigid_body_pos[0, 0, 2].item(),
                    "reference_root_height": ctx.mimic.ref_state.rigid_body_pos[
                        0, 0, 2
                    ].item(),
                    "raw_action_delta_mean_abs": raw_delta,
                    "processed_pd_target_delta_mean_abs": processed_delta,
                    "max_abs_joint_velocity": ctx.current.dof_vel[0].abs().max().item(),
                }
            )

            previous_raw_action = raw_action.clone()
            previous_processed_action = processed_action

    return rows


def write_trace_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "sim_time",
        "tracking_error_mean",
        "tracking_error_max",
        "root_height",
        "reference_root_height",
        "raw_action_delta_mean_abs",
        "processed_pd_target_delta_mean_abs",
        "max_abs_joint_velocity",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_trace_png(rows: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib is not importable; skipping trace.png")
        return

    times = [row["sim_time"] for row in rows]
    err_mean = [row["tracking_error_mean"] for row in rows]
    err_max = [row["tracking_error_max"] for row in rows]
    root_height = [row["root_height"] for row in rows]
    ref_root_height = [row["reference_root_height"] for row in rows]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(times, err_mean, label="mean")
    axes[0].plot(times, err_max, label="max")
    axes[0].axhline(0.5, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("tracking error (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, root_height, label="root")
    axes[1].plot(times, ref_root_height, label="reference")
    axes[1].axhline(0.4, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("root height (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_verdict(rows: list[dict[str, float]]) -> None:
    first_diverged = next(
        (row for row in rows if row["tracking_error_mean"] > 0.5),
        None,
    )
    first_fall = next((row for row in rows if row["root_height"] < 0.4), None)

    if first_diverged is None:
        print("Verdict: mean tracking error never exceeded 0.5.")
    else:
        print(
            "Verdict: mean tracking error first exceeded 0.5 at "
            f"step {first_diverged['step']} "
            f"(t={first_diverged['sim_time']:.3f}s)."
        )

    if first_fall is not None:
        print(
            "Root-height diagnosis: fall, root height first dropped below 0.4 m at "
            f"step {first_fall['step']} (t={first_fall['sim_time']:.3f}s)."
        )
    elif first_diverged is not None:
        print("Root-height diagnosis: oscillation/drift; root stayed above 0.4 m.")
    else:
        print("Root-height diagnosis: no fall and no large tracking divergence.")


def main() -> int:
    global args
    args = parser.parse_args()

    if args.ema_alpha is not None and not 0.0 <= args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in [0, 1]")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(args.output_dir, args.checkpoint, repo_root)

    agent, env = load_agent_and_env(args)
    try:
        rows = collect_trace(agent, env, args.ema_alpha, args.max_steps)
        write_trace_csv(rows, output_dir / "trace.csv")
        maybe_write_trace_png(rows, output_dir / "trace.png")
        print_verdict(rows)
        print(f"Wrote {output_dir / 'trace.csv'}")
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
