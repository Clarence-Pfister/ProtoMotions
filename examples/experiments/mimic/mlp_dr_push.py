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
"""DR-mild plus push continuation for the standard mimic MLP.

This keeps the same observation/action/model structure as ``mlp.py`` and
``mlp_dr_mild.py`` so those checkpoints can be warm-started safely, then adds
small external push perturbations for MuJoCo sim2sim robustness.
"""

from pathlib import Path
import importlib.util
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    PushDomainRandomizationConfig,
    SimulatorConfig,
)
import argparse


_MILD_PATH = Path(__file__).with_name("mlp_dr_mild.py")
_MILD_SPEC = importlib.util.spec_from_file_location("_mimic_mlp_dr_mild", _MILD_PATH)
_mild = importlib.util.module_from_spec(_MILD_SPEC)
assert _MILD_SPEC.loader is not None
_MILD_SPEC.loader.exec_module(_mild)

agent_config = _mild.agent_config
env_config = _mild.env_config
motion_lib_config = _mild.motion_lib_config
scene_lib_config = _mild.scene_lib_config
terrain_config = _mild.terrain_config
_base_configure_robot_and_simulator = _mild.configure_robot_and_simulator


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Add push DR on top of the checkpoint-compatible mild DR setup."""
    _base_configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    simulator_cfg.domain_randomization.push = PushDomainRandomizationConfig(
        push_interval_range=(2.0, 4.0),
        max_linear_velocity=(0.20, 0.20, 0.05),
        max_angular_velocity=(0.20, 0.20, 0.30),
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    """Use clean physics for inference/evaluation of the DR-push policy."""
    _mild.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
