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
Observation Parity Check
========================

Cross-simulator diagnostic: run the same policy from the same deterministic reset
in a simulator and record the observation dict and the policy's mean action at
every step. Recordings from two simulators can then be diffed to localize where
they diverge — at reset (observation computation mismatch) or gradually
(closed-loop dynamics drift).

Usage (record, once per simulator):
    python scripts/obs_parity_check.py --record \
        --checkpoint results/<exp>/epoch_X.ckpt --simulator mujoco \
        --motion-file <file> --num-steps 50 --output results/obs_parity/mujoco.pt

Usage (compare two recordings, any python with torch):
    python scripts/obs_parity_check.py --compare \
        results/obs_parity/mujoco.pt results/obs_parity/isaaclab.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_parser():
    parser = argparse.ArgumentParser(
        description="Record/compare per-step observations and actions across simulators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--compare", nargs=2, metavar=("A_PT", "B_PT"))
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--simulator", type=str)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    return parser


args = create_parser().parse_args()


def compare(path_a: str, path_b: str) -> None:
    import torch

    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    name_a = a["simulator"]
    name_b = b["simulator"]
    steps = min(len(a["records"]), len(b["records"]))
    print(f"Comparing {name_a} vs {name_b} over {steps} steps\n")

    keys = sorted(a["records"][0]["obs"].keys())
    header = f"{'step':>4} " + " ".join(f"{k[:18]:>18}" for k in keys) + f" {'action':>10} {'rootH_a':>8} {'rootH_b':>8}"
    print(header)
    for i in range(steps):
        ra, rb = a["records"][i], b["records"][i]
        cells = []
        for k in keys:
            d = (ra["obs"][k] - rb["obs"][k]).abs().max().item()
            cells.append(f"{d:18.5f}")
        act_d = (ra["action"] - rb["action"]).abs().max().item()
        print(
            f"{i:>4} " + " ".join(cells) + f" {act_d:10.5f} "
            f"{ra['root_height']:8.3f} {rb['root_height']:8.3f}"
        )

    print("\nInterpretation:")
    print("  large obs diff at step 0        -> observation computation mismatch")
    print("  step-0 match, growing diffs     -> closed-loop dynamics drift")


def record() -> None:
    assert args.checkpoint and args.simulator, "--record needs --checkpoint and --simulator"

    from protomotions.utils.simulator_imports import import_simulator_before_torch

    AppLauncher = import_simulator_before_torch(args.simulator)

    import torch
    from dataclasses import asdict
    from pathlib import Path
    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class

    checkpoint = Path(args.checkpoint)
    resolved = torch.load(
        checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    simulator_config.num_envs = 1
    simulator_config.headless = True
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric = Fabric(
        **asdict(
            FabricConfig(
                accelerator=accelerator, devices=1, num_nodes=1, loggers=[], callbacks=[]
            )
        )
    )
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": True, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    obs, _ = env.reset(sample_flat=True, disable_motion_resample=True)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    records = []
    with torch.no_grad():
        for _ in range(args.num_steps):
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            sim_state = env.simulator.get_robot_state()
            records.append(
                {
                    "obs": {k: v.detach().cpu().clone() for k, v in obs.items()},
                    "action": actions.detach().cpu().clone(),
                    "root_height": sim_state.root_pos[0, 2].item(),
                }
            )
            obs, _, _, _, _ = env.step(actions)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

    out = args.output or os.path.join("results", "obs_parity", f"{args.simulator}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"simulator": args.simulator, "records": records}, out)
    print(f"saved {len(records)} steps -> {out}")

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    if args.compare:
        compare(*args.compare)
    else:
        record()
