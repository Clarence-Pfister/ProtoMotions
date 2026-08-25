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
from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.robot_configs.base import RobotAssetConfig
from protomotions.robot_configs.g1 import G1RobotConfig


@dataclass
class G1_23DOFRobotConfig(G1RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_wrist_roll_rubber_hand"],
            "all_right_hand_bodies": ["right_wrist_roll_rubber_hand"],
            # This 23-DOF MJCF has head geometry on torso_link, not a separate head body.
            "head_body_name": ["torso_link"],
            "torso_body_name": ["torso_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "left_wrist_roll_rubber_hand",
            "right_wrist_roll_rubber_hand",
        ]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/g1_23dof_holo_compat.xml",
            usd_asset_file_name="usd/g1_23dof_holo_compat/g1_23dof_holo_compat_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
