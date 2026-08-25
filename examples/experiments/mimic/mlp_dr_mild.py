# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""DR-mild continuation for the standard mimic MLP.

This keeps the same observation/action/model structure as ``mlp.py`` so an
``mlp.py`` checkpoint can be warm-started safely, then adds conservative
physics-side randomization.
"""

from pathlib import Path
import importlib.util
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    RobotNoiseConfig,
    SimulatorConfig,
)
import argparse


_MLP_PATH = Path(__file__).with_name("mlp.py")
_MLP_SPEC = importlib.util.spec_from_file_location("_mimic_mlp_base", _MLP_PATH)
_mlp = importlib.util.module_from_spec(_MLP_SPEC)
assert _MLP_SPEC.loader is not None
_MLP_SPEC.loader.exec_module(_mlp)

agent_config = _mlp.agent_config
env_config = _mlp.env_config
motion_lib_config = _mlp.motion_lib_config
scene_lib_config = _mlp.scene_lib_config
terrain_config = _mlp.terrain_config
_base_configure_robot_and_simulator = _mlp.configure_robot_and_simulator


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Add mild DR while preserving the original MLP checkpoint interface."""
    _base_configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.03,
        root_pos_noise=[0.02, 0.02, 0.005],
        root_rot_noise=[0.03, 0.03, 0.06],
        root_vel_noise=[0.03, 0.03, 0.02],
        root_ang_vel_noise=[0.03, 0.03, 0.03],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.005, 0.005),
            dof_names=[".*"],
            dof_indices=None,
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.8, 1.2),
            dynamic_friction_range=(0.8, 1.2),
            restitution_range=(0.0, 0.0),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.015, 0.015),
            },
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
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
    """Use clean physics for inference/evaluation of the DR-trained policy."""
    _mlp.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
