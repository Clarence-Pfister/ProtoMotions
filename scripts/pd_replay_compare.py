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
"""
PD Replay Plant Comparison
==========================

Plant-level diagnostic: track a reference motion with raw PD control (no policy).
Each control step, the PD position target is set to the reference dof_pos a small
lookahead into the future. This isolates the simulator plant (actuation, contacts,
solver) from policy behavior, so the same motion can be compared across simulators.

Usage:
    python scripts/pd_replay_compare.py \
        --robot-name g1 \
        --simulator mujoco \
        --motion-file data/g1-gemx-generated/reference-clip-first7s_retarget_g1_fixed_under_body.pt \
        --output-dir results/pd_replay/mujoco
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_parser():
    parser = argparse.ArgumentParser(
        description="Track a motion with raw PD control (no policy) to compare simulator plants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-name", type=str, default="g1")
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        choices=["mujoco", "newton", "isaaclab"],
    )
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--lookahead-steps",
        type=int,
        default=1,
        help="PD target = reference pose this many control steps ahead.",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True, help="Always headless."
    )
    return parser


args = create_parser().parse_args()

# Simulators must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.factory import get_simulator_config_class  # noqa: E402


def build_simulator(robot_cfg, device):
    """Construct a single-env simulator over flat terrain."""
    terrain_cfg = TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        border_size=40.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
    terrain = Terrain(config=terrain_cfg, num_envs=1, device=device)
    scene_lib = SceneLib.empty(num_envs=1, device=str(device), terrain=terrain)

    SimConfigClass = get_simulator_config_class(args.simulator)
    sim_params = getattr(robot_cfg.simulation_params, args.simulator, None)
    if sim_params is None:
        sim_params = SimConfigClass.__dataclass_fields__["sim"].default_factory()
    sim_cfg = SimConfigClass(
        sim=sim_params,
        headless=True,
        num_envs=1,
        experiment_name=f"pd_replay_{args.simulator}",
    )

    extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": True, "device": str(device)})
        extra_params["simulation_app"] = app_launcher.app

    from protomotions.utils.component_builder import build_simulator_from_config

    simulator = build_simulator_from_config(
        simulator_config=sim_cfg,
        robot_config=robot_cfg,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
        **extra_params,
    )
    simulator._initialize_with_markers({})
    return simulator


def main():
    device = torch.device("cpu" if args.simulator == "mujoco" else "cuda")
    out_dir = args.output_dir or os.path.join("results", "pd_replay", args.simulator)
    os.makedirs(out_dir, exist_ok=True)

    robot_cfg = robot_config(args.robot_name)
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=args.motion_file), device=str(device)
    )
    motion_len = motion_lib.motion_lengths[0].item()
    motion_ids = torch.zeros(1, dtype=torch.long, device=device)

    simulator = build_simulator(robot_cfg, device)
    frame_dt = simulator.dt
    num_steps = int(motion_len / frame_dt) - args.lookahead_steps
    print(
        f"Plant: {args.simulator} | control dt {frame_dt:.4f}s | "
        f"motion {motion_len:.2f}s -> {num_steps} steps | "
        f"lookahead {args.lookahead_steps} step(s)"
    )

    # Place the motion on valid flat terrain (env origins differ per simulator)
    flat_xy = simulator.terrain.sample_flat_locations(1)[0, :2].to(device)
    first = motion_lib.get_motion_state(
        motion_ids, torch.zeros(1, device=device)
    )
    xy_offset = (flat_xy - first.root_pos[0, :2]).view(1, 2)
    print(f"spawn offset applied: {xy_offset.tolist()}")

    def ref_at(t: float):
        times = torch.tensor([min(t, motion_len - 1e-4)], device=device)
        state = motion_lib.get_motion_state(motion_ids, times)
        state.root_pos[:, :2] += xy_offset
        state.rigid_body_pos[:, :, :2] += xy_offset.unsqueeze(1)
        return state

    simulator.reset_envs(ref_at(0.0))

    rows = []
    first_03 = first_05 = None
    for k in range(num_steps):
        t_now = k * frame_dt
        target = ref_at(t_now + args.lookahead_steps * frame_dt)
        simulator.step(target.dof_pos)

        sim_state = simulator.get_robot_state()
        ref = ref_at(t_now + frame_dt)

        body_err = (sim_state.rigid_body_pos - ref.rigid_body_pos).norm(dim=-1)[0]
        dof_err = (sim_state.dof_pos - ref.dof_pos).abs()[0]
        row = {
            "step": k,
            "sim_time": round(t_now + frame_dt, 4),
            "body_err_mean": body_err.mean().item(),
            "body_err_max": body_err.max().item(),
            "dof_err_mean": dof_err.mean().item(),
            "dof_err_max": dof_err.max().item(),
            "root_height": sim_state.root_pos[0, 2].item(),
            "ref_root_height": ref.root_pos[0, 2].item(),
        }
        rows.append(row)
        if first_03 is None and row["body_err_mean"] > 0.3:
            first_03 = row
        if first_05 is None and row["body_err_mean"] > 0.5:
            first_05 = row

    csv_path = os.path.join(out_dir, "replay.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    min_h = min(r["root_height"] for r in rows)
    mean_err = sum(r["body_err_mean"] for r in rows) / len(rows)
    max_err = max(r["body_err_max"] for r in rows)
    print(f"\n=== PD replay summary ({args.simulator}) ===")
    print(f"body_err mean over rollout: {mean_err:.4f} m | worst body: {max_err:.4f} m")
    print(f"min root height: {min_h:.3f} m (ref ~{rows[0]['ref_root_height']:.3f})")
    for label, hit in [("err>0.3", first_03), ("err>0.5", first_05)]:
        if hit is None:
            print(f"first {label}: never")
        else:
            print(f"first {label}: step {hit['step']} (t={hit['sim_time']:.2f}s)")
    verdict = (
        "PLANT TRACKS (divergence is policy-related)"
        if first_03 is None and min_h > 0.4
        else "PLANT DIVERGES (simulator-level gap, independent of policy)"
    )
    print(f"verdict: {verdict}")
    print(f"csv: {csv_path}")

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()


if __name__ == "__main__":
    main()
