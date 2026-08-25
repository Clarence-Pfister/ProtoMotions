# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible MuJoCo-smooth MLP fine-tune with L2C2.

This is BM/L2C2-inspired, but intentionally keeps the same actor, critic, and
action interface as ``mlp.py``. That makes it suitable for warm-starting from
``mlp.py`` or ``mlp_dr_mild.py`` checkpoints.

Compared with ``mlp_dr_mild.py`` this adds:
- noisy actor observations plus clean observation counterparts for L2C2
- stronger processed-action smoothness reward
- a default PD target acceleration clamp for MuJoCo-style jitter reduction
- moderate observation noise and small push perturbations
"""

import argparse
import importlib.util
from pathlib import Path

from protomotions.agents.ppo.config import PPOAgentConfig, L2C2Config
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    PushDomainRandomizationConfig,
    RobotNoiseConfig,
    SimulatorConfig,
)


_MLP_PATH = Path(__file__).with_name("mlp.py")
_MLP_SPEC = importlib.util.spec_from_file_location("_mimic_mlp_base", _MLP_PATH)
_mlp = importlib.util.module_from_spec(_MLP_SPEC)
assert _MLP_SPEC.loader is not None
_MLP_SPEC.loader.exec_module(_mlp)

motion_lib_config = _mlp.motion_lib_config
scene_lib_config = _mlp.scene_lib_config
terrain_config = _mlp.terrain_config
_base_configure_robot_and_simulator = _mlp.configure_robot_and_simulator
_base_apply_inference_overrides = _mlp.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    """Build a warm-start-compatible MLP env with noisy/clean L2C2 pairs."""
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
    )

    env_cfg = _mlp.env_config(robot_cfg, args)

    # Keep the original actor key names, but make those observations noisy during
    # training. Add clean counterparts under new keys for L2C2.
    env_cfg.observation_components["max_coords_obs"] = max_coords_obs_factory(
        use_noisy=True
    )
    env_cfg.observation_components["mimic_target_poses"] = (
        mimic_target_poses_max_coords_factory(use_noisy=True, with_velocities=True)
    )
    env_cfg.observation_components["clean_max_coords_obs"] = max_coords_obs_factory(
        use_noisy=False
    )
    env_cfg.observation_components["clean_mimic_target_poses"] = (
        mimic_target_poses_max_coords_factory(use_noisy=False, with_velocities=True)
    )

    if "action_smoothness" in env_cfg.reward_components:
        env_cfg.reward_components["action_smoothness"].static_params["weight"] = -0.08

    return env_cfg


def agent_config(
    robot_config: RobotConfig, env_config, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build a checkpoint-compatible PPO config with conservative L2C2."""
    agent_cfg = _mlp.agent_config(robot_config, env_config, args)

    # Extra observation keys are present in the batch for L2C2 only. The actor
    # and critic still read the original three keys, so checkpoint shapes match.
    for key in ["clean_max_coords_obs", "clean_mimic_target_poses"]:
        if key not in agent_cfg.model.in_keys:
            agent_cfg.model.in_keys.append(key)

    agent_cfg.l2c2 = L2C2Config(
        enabled=True,
        lambda_l2c2=0.1,
        obs_pairs={
            "max_coords_obs": "clean_max_coords_obs",
            "mimic_target_poses": "clean_mimic_target_poses",
        },
    )

    # Conservative defaults for continuation. The PPO loader reapplies these
    # after loading optimizer state when adaptive LR is disabled.
    agent_cfg.model.actor_optimizer.lr = 1e-5
    agent_cfg.model.critic_optimizer.lr = 5e-5
    return agent_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Add MuJoCo-focused smoothness and moderate DR."""
    _base_configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.04,
        root_pos_noise=[0.03, 0.03, 0.008],
        root_rot_noise=[0.05, 0.05, 0.10],
        root_vel_noise=[0.05, 0.05, 0.03],
        root_ang_vel_noise=[0.05, 0.05, 0.05],
    )

    simulator_cfg.pd_target_max_accel = 0.05
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.01, 0.01),
            dof_names=[".*"],
            dof_indices=None,
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.6, 1.4),
            dynamic_friction_range=(0.6, 1.3),
            restitution_range=(0.0, 0.1),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (-0.02, 0.02),
            },
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=RobotNoiseConfig(
            body_pos_noise=0.003,
            body_rot_noise=0.01,
            body_vel_noise=0.05,
            body_ang_vel_noise=0.08,
            ground_height_noise=0.005,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(2.0, 4.0),
            max_linear_velocity=(0.20, 0.20, 0.05),
            max_angular_velocity=(0.20, 0.20, 0.30),
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
    """Use clean observations and clean physics for evaluation/inference."""
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
    )

    _base_apply_inference_overrides(
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

    env_cfg.observation_components["max_coords_obs"] = max_coords_obs_factory(
        use_noisy=False
    )
    env_cfg.observation_components["mimic_target_poses"] = (
        mimic_target_poses_max_coords_factory(use_noisy=False, with_velocities=True)
    )
    env_cfg.observation_components.pop("clean_max_coords_obs", None)
    env_cfg.observation_components.pop("clean_mimic_target_poses", None)
    agent_cfg.l2c2.enabled = False
