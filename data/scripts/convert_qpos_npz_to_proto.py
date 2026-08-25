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
Convert robot qpos NPZ files to ProtoMotions .motion files.

Expected NPZ layout:
    qpos: (T, nq) MuJoCo qpos in robot MJCF order

For a free-root 23-DOF G1 model, nq should be 30:
    [root_pos_xyz(3), root_quat_wxyz(4), joint_angles(23)]

Optional keys:
    fps: scalar input frame rate
    foot_contacts: (T, 2) [left, right] or (T, 4)
                   [left heel, left toe, right heel, right toe]

Example:
    python data/scripts/convert_qpos_npz_to_proto.py \
        --input-dir amass/retargeted_g1_23dof \
        --output-dir motions/g1_23dof_proto \
        --mjcf-path ../my_mjcf/g1_23dof_holo_compat.xml \
        --left-foot-body left_ankle_roll_link \
        --right-foot-body right_ankle_roll_link \
        --output-fps 30
"""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from tqdm import tqdm

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)


def _resolve_input_fps(data, input_fps: Optional[int], fps_key: str, npz_file: Path) -> int:
    if input_fps is not None:
        return input_fps

    if fps_key not in data:
        raise ValueError(
            f"{npz_file.name}: --input-fps was not provided and NPZ has no '{fps_key}' key"
        )

    return int(np.asarray(data[fps_key]).item())


def _load_qpos_and_contacts(
    npz_file: Path,
    qpos_key: str,
    foot_contacts_key: str,
    input_fps: Optional[int],
    output_fps: int,
    ignore_first_n_frames: int,
):
    data = np.load(npz_file, allow_pickle=True)

    if qpos_key not in data:
        raise ValueError(f"{npz_file.name}: missing '{qpos_key}'")

    actual_input_fps = _resolve_input_fps(data, input_fps, "fps", npz_file)
    if actual_input_fps % output_fps != 0:
        raise ValueError(
            f"{npz_file.name}: input_fps ({actual_input_fps}) must be divisible by "
            f"output_fps ({output_fps})"
        )

    downsample_factor = actual_input_fps // output_fps
    qpos = np.asarray(data[qpos_key])
    if qpos.ndim != 2:
        raise ValueError(f"{npz_file.name}: '{qpos_key}' must be rank-2, got {qpos.shape}")

    qpos = qpos[::downsample_factor]
    embedded_contacts = None

    if foot_contacts_key in data:
        embedded_contacts = np.asarray(data[foot_contacts_key])[::downsample_factor]
        if embedded_contacts.ndim != 2 or embedded_contacts.shape[1] not in (2, 4):
            raise ValueError(
                f"{npz_file.name}: '{foot_contacts_key}' must have shape (T, 2) or "
                f"(T, 4), got {embedded_contacts.shape}"
            )

    if ignore_first_n_frames > 0:
        qpos = qpos[ignore_first_n_frames:]
        if embedded_contacts is not None:
            embedded_contacts = embedded_contacts[ignore_first_n_frames:]

    if qpos.shape[0] < 2:
        raise ValueError(f"{npz_file.name}: motion must have at least 2 frames")

    return qpos, embedded_contacts, actual_input_fps


def _normalize_root_quat(qpos: torch.Tensor, motion_name: str) -> torch.Tensor:
    qpos = qpos.clone()
    root_quat = qpos[:, 3:7]
    norm = torch.linalg.norm(root_quat, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError(f"{motion_name}: root quaternion contains near-zero values")
    qpos[:, 3:7] = root_quat / norm
    return qpos


def _embedded_contacts_to_lr(embedded_contacts: np.ndarray) -> np.ndarray:
    if embedded_contacts.shape[1] == 2:
        return embedded_contacts

    left = np.max(embedded_contacts[:, 0:2], axis=1)
    right = np.max(embedded_contacts[:, 2:4], axis=1)
    return np.stack([left, right], axis=-1)


def _make_foot_contacts(
    foot_contacts_lr: np.ndarray,
    motion_length: int,
    num_bodies: int,
    left_foot_idx: int,
    right_foot_idx: int,
    device,
):
    if foot_contacts_lr.shape[0] != motion_length:
        raise ValueError(
            f"Foot contact length ({foot_contacts_lr.shape[0]}) does not match "
            f"motion length ({motion_length})"
        )

    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)
    rigid_body_contacts[:, left_foot_idx] = foot_contacts_lr[:, 0] > 0.5
    rigid_body_contacts[:, right_foot_idx] = foot_contacts_lr[:, 1] > 0.5
    return torch.from_numpy(rigid_body_contacts).to(device)


def _load_contact_labels(
    contact_labels_dir: Path,
    motion_file: Path,
    input_fps: int,
    output_fps: int,
    ignore_first_n_frames: int,
) -> np.ndarray:
    base_filename = motion_file.stem
    if base_filename.endswith("_retargeted"):
        base_filename = base_filename[: -len("_retargeted")]

    contact_labels_path = contact_labels_dir / f"{base_filename}_contacts.npz"
    if not contact_labels_path.exists():
        raise FileNotFoundError(f"Contact labels file not found: {contact_labels_path}")

    contact_data = np.load(contact_labels_path, allow_pickle=True)
    if "foot_contacts" not in contact_data:
        raise ValueError(f"{contact_labels_path}: missing 'foot_contacts'")

    factor = input_fps // output_fps
    foot_contacts = np.asarray(contact_data["foot_contacts"])[::factor]
    if ignore_first_n_frames > 0:
        foot_contacts = foot_contacts[ignore_first_n_frames:]

    if foot_contacts.ndim != 2 or foot_contacts.shape[1] not in (2, 4):
        raise ValueError(
            f"{contact_labels_path}: foot_contacts must have shape (T, 2) or (T, 4), "
            f"got {foot_contacts.shape}"
        )

    return _embedded_contacts_to_lr(foot_contacts)


def _convert_qpos_to_motion(
    qpos_np: np.ndarray,
    kinematic_info,
    output_fps: int,
    device,
    dtype,
    motion_name: str,
    fix_height_per_frame: bool,
    fix_height_offset: float,
):
    qpos = torch.from_numpy(qpos_np).to(device=device, dtype=dtype)
    qpos = _normalize_root_quat(qpos, motion_name)

    expected_nq = kinematic_info.nq
    if qpos.shape[-1] != expected_nq:
        raise ValueError(
            f"{motion_name}: qpos has {qpos.shape[-1]} columns, expected {expected_nq} "
            f"from MJCF ({kinematic_info.num_dofs} DOFs + 7 root)"
        )

    joint_angles = qpos[:, 7:]
    root_pos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    reconstructed_qpos = extract_qpos_from_transforms(
        kinematic_info, root_pos, joint_rot_mats
    )
    motion.dof_pos = reconstructed_qpos[:, 7:]

    allowed_delta = [0.0, 2 * np.pi, 4 * np.pi]
    delta = (reconstructed_qpos[:, 7:] - joint_angles).abs()
    allowed = torch.zeros_like(delta, dtype=torch.bool)
    for offset in allowed_delta:
        allowed |= (delta - offset).abs() < 1e-4
    if not allowed.all():
        max_delta = delta.min(dim=0).values.max().item()
        raise AssertionError(
            f"{motion_name}: reconstructed qpos and source joint angles diverge "
            f"(max min-column delta {max_delta:.6f})"
        )

    dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles.unsqueeze(1),
        fps=output_fps,
    )
    motion.dof_vel = dof_vel.squeeze(1)

    if fix_height_per_frame:
        translation_vecs = motion.fix_height_per_frame(height_offset=fix_height_offset)
        if motion.rigid_body_vel is not None and motion.fps is not None:
            vel_delta = torch.zeros(
                translation_vecs.shape[0],
                1,
                3,
                device=motion.rigid_body_vel.device,
                dtype=motion.rigid_body_vel.dtype,
            )
            vel_delta[:-1] = (
                (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1)
                / motion.motion_dt
            )
            motion.rigid_body_vel = motion.rigid_body_vel + vel_delta
    else:
        motion.fix_height(height_offset=fix_height_offset)

    motion.local_rigid_body_rot = None
    return motion


@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Directory containing qpos .npz files."),
    output_dir: Path = typer.Option(..., help="Directory to save .motion files."),
    mjcf_path: Path = typer.Option(
        ...,
        help="MJCF used for the qpos order, e.g. ../my_mjcf/g1_23dof_holo_compat.xml.",
    ),
    input_fps: Optional[int] = typer.Option(
        None,
        help="Input motion FPS. If omitted, read scalar 'fps' from each NPZ.",
    ),
    output_fps: int = typer.Option(30, help="Target output FPS."),
    qpos_key: str = typer.Option("qpos", help="NPZ key containing MuJoCo qpos."),
    foot_contacts_key: str = typer.Option(
        "foot_contacts",
        help="Optional NPZ key with (T, 2) or (T, 4) foot contacts.",
    ),
    left_foot_body: str = typer.Option(
        "left_ankle_roll_link", help="MJCF body name for left foot contacts."
    ),
    right_foot_body: str = typer.Option(
        "right_ankle_roll_link", help="MJCF body name for right foot contacts."
    ),
    contact_source: str = typer.Option(
        "auto",
        help="'auto', 'embedded', 'heuristic', or 'none'. contact_labels_dir overrides this.",
    ),
    contact_labels_dir: Optional[Path] = typer.Option(
        None,
        help="Directory with *_contacts.npz files. Overrides contact_source when set.",
    ),
    force_remake: bool = False,
    ignore_first_n_frames: int = 0,
    fix_height_per_frame: bool = typer.Option(
        True, help="Lift each frame that penetrates the ground."
    ),
    fix_height_offset: float = typer.Option(0.02, help="Minimum body height after fixing."),
    apply_motion_filter: bool = typer.Option(False, help="Apply motion quality filter."),
    min_height_threshold: float = typer.Option(-0.05),
    max_velocity_threshold: float = typer.Option(15.0),
    max_dof_vel_threshold: float = typer.Option(40.0),
    duration_height_filter: float = typer.Option(0.1),
    duration_height_seconds: float = typer.Option(0.6),
    yaml_output_name: Optional[str] = typer.Option(
        None, help="Optional YAML filename listing converted motions."
    ),
    num_rank: int = typer.Option(
        1, help="Total number of parallel ranks for file-level splitting."
    ),
    slurm_rank: int = typer.Option(
        0, help="This rank index for file-level splitting."
    ),
):
    if contact_source not in {"auto", "embedded", "heuristic", "none"}:
        raise ValueError("--contact-source must be one of: auto, embedded, heuristic, none")

    if num_rank < 1 or not (0 <= slurm_rank < num_rank):
        raise ValueError("--num-rank must be >= 1 and --slurm-rank must be in [0, num_rank)")

    device = torch.device("cpu")
    dtype = torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)

    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

    kinematic_info = extract_kinematic_info(str(mjcf_path))
    body_names = kinematic_info.body_names
    if left_foot_body not in body_names:
        raise ValueError(f"left_foot_body '{left_foot_body}' not found in {body_names}")
    if right_foot_body not in body_names:
        raise ValueError(f"right_foot_body '{right_foot_body}' not found in {body_names}")
    left_foot_idx = body_names.index(left_foot_body)
    right_foot_idx = body_names.index(right_foot_body)

    npz_files = sorted(input_dir.rglob("*.npz"))
    print(f"MJCF: {mjcf_path}")
    print(f"nq={kinematic_info.nq}, num_dofs={kinematic_info.num_dofs}, num_bodies={kinematic_info.num_bodies}")
    print(f"Left foot:  {left_foot_body} (index {left_foot_idx})")
    print(f"Right foot: {right_foot_body} (index {right_foot_idx})")
    print(f"Found {len(npz_files)} NPZ files in {input_dir}")

    output_motions_yaml = []

    for npz_file in tqdm(npz_files, desc="Processing qpos motions"):
        rel_path = npz_file.relative_to(input_dir)
        file_hash = int(hashlib.sha256(str(rel_path).encode("utf-8")).hexdigest(), 16)
        if file_hash % num_rank != slurm_rank:
            continue

        output_file = (output_dir / str(rel_path).replace(" ", "_")).with_suffix(".motion")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not force_remake and output_file.exists():
            continue

        print(f"Processing {npz_file}")
        try:
            qpos_np, embedded_contacts, actual_input_fps = _load_qpos_and_contacts(
                npz_file=npz_file,
                qpos_key=qpos_key,
                foot_contacts_key=foot_contacts_key,
                input_fps=input_fps,
                output_fps=output_fps,
                ignore_first_n_frames=ignore_first_n_frames,
            )

            motion = _convert_qpos_to_motion(
                qpos_np=qpos_np,
                kinematic_info=kinematic_info,
                output_fps=output_fps,
                device=device,
                dtype=dtype,
                motion_name=npz_file.name,
                fix_height_per_frame=fix_height_per_frame,
                fix_height_offset=fix_height_offset,
            )

            selected_contact_source = contact_source
            if contact_labels_dir is not None:
                contacts_lr = _load_contact_labels(
                    contact_labels_dir=contact_labels_dir,
                    motion_file=npz_file,
                    input_fps=actual_input_fps,
                    output_fps=output_fps,
                    ignore_first_n_frames=ignore_first_n_frames,
                )
                selected_contact_source = "contact_labels_dir"
                motion.rigid_body_contacts = _make_foot_contacts(
                    foot_contacts_lr=contacts_lr,
                    motion_length=motion.rigid_body_pos.shape[0],
                    num_bodies=motion.rigid_body_pos.shape[1],
                    left_foot_idx=left_foot_idx,
                    right_foot_idx=right_foot_idx,
                    device=device,
                )
            elif contact_source in {"auto", "embedded"} and embedded_contacts is not None:
                contacts_lr = _embedded_contacts_to_lr(embedded_contacts)
                selected_contact_source = "embedded"
                motion.rigid_body_contacts = _make_foot_contacts(
                    foot_contacts_lr=contacts_lr,
                    motion_length=motion.rigid_body_pos.shape[0],
                    num_bodies=motion.rigid_body_pos.shape[1],
                    left_foot_idx=left_foot_idx,
                    right_foot_idx=right_foot_idx,
                    device=device,
                )
            elif contact_source == "embedded":
                raise ValueError(f"{npz_file.name}: requested embedded contacts, but none found")
            elif contact_source in {"auto", "heuristic"}:
                selected_contact_source = "heuristic"
                motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
                    positions=motion.rigid_body_pos,
                    velocity=motion.rigid_body_vel,
                    vel_thres=0.15,
                    height_thresh=0.1,
                ).to(torch.bool)
            else:
                selected_contact_source = "none"
                motion.rigid_body_contacts = torch.zeros(
                    motion.rigid_body_pos.shape[0],
                    motion.rigid_body_pos.shape[1],
                    device=device,
                    dtype=torch.bool,
                )

            if apply_motion_filter and not passes_exclude_motion_filter(
                motion,
                min_height_threshold=min_height_threshold,
                max_velocity_threshold=max_velocity_threshold,
                max_dof_vel_threshold=max_dof_vel_threshold,
                duration_height_filter=duration_height_filter,
                duration_height_seconds=duration_height_seconds,
            ):
                print(f"Skipping {npz_file.name} (failed motion filter)")
                continue

            print(f"  input_fps:       {actual_input_fps}")
            print(f"  contact_source:  {selected_contact_source}")
            print(f"  dof_pos:         {motion.dof_pos.shape}")
            print(f"  dof_vel:         {motion.dof_vel.shape}")
            print(f"  rigid_body_pos:  {motion.rigid_body_pos.shape}")
            print(f"  rigid_body_rot:  {motion.rigid_body_rot.shape}")
            print(f"  Saving to {output_file}")
            torch.save(motion.to_dict(), str(output_file))

            if yaml_output_name is not None:
                output_motions_yaml.append(
                    {"file": str(output_file.relative_to(output_dir)), "fps": output_fps}
                )

        except Exception as exc:
            print(f"Error processing {npz_file}: {exc}")
            import traceback

            traceback.print_exc()
            continue

    if yaml_output_name is not None:
        import yaml

        yaml_output = output_dir / yaml_output_name
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
