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
import importlib.util
import sys
from pathlib import Path


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "mujoco_clamp_sweep.py"
    spec = importlib.util.spec_from_file_location("mujoco_clamp_sweep", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_eval_block_extracts_metrics():
    sweep = _load_sweep_module()
    text = """
noise before
============================================================
EVALUATION RESULTS
============================================================
  eval/action_delta_max_rad: 0.110200
  eval/action_delta_mean_rad: 0.020000
  eval/gt_error/max: 0.064700
  eval/gt_error/mean: 0.033500
  eval/max_joint_error/mean: 0.082300
  eval/normalized_jerk_mean: 724.500000
  eval/success_rate: 1.000000
============================================================
  Overall Score: 1.000000
============================================================
noise after
"""

    metrics = sweep.parse_eval_block(text)

    assert metrics["eval/success_rate"] == 1.0
    assert metrics["eval/gt_error/mean"] == 0.0335
    assert metrics["eval/gt_error/max"] == 0.0647
    assert metrics["eval/action_delta_max_rad"] == 0.1102
    assert "Overall Score" not in metrics


def test_summary_table_uses_verdict_and_missing_values():
    sweep = _load_sweep_module()
    result_pass = sweep.SweepResult(
        clamp="none",
        command=[],
        log_path=Path("clamp_none.log"),
        returncode=0,
        timed_out=False,
        metrics={
            "eval/success_rate": 1.0,
            "eval/gt_error/mean": 0.0335,
            "eval/gt_error/max": 0.0647,
            "eval/max_joint_error/mean": 0.0823,
            "eval/normalized_jerk_mean": 724.5,
            "eval/action_delta_mean_rad": 0.02,
            "eval/action_delta_max_rad": 0.1102,
        },
    )
    result_fail = sweep.SweepResult(
        clamp="0.05",
        command=[],
        log_path=Path("clamp_0.05.log"),
        returncode=1,
        timed_out=False,
        metrics={},
    )

    summary = sweep.build_summary([result_pass, result_fail])

    assert "| none | 1 | 0.0335 | 0.0647 | 0.0823 | 724.5 | 0.02 | 0.1102 | PASS |" in summary
    assert "| 0.05 | - | - | - | - | - | - | - | FAIL |" in summary
    assert "Best passing setting: no clamp needed." in summary


def test_build_inference_command_omits_none_override():
    sweep = _load_sweep_module()

    command = sweep.build_inference_command("python", "ckpt.ckpt", "motion.motion", "none")

    assert "--overrides" not in command
    assert command[:4] == [
        "python",
        "protomotions/inference_agent.py",
        "--checkpoint",
        "ckpt.ckpt",
    ]
    assert "--simulator" in command
    assert "mujoco" in command
    assert "--num-envs" in command
    assert "1" in command


def test_build_inference_command_adds_clamp_override():
    sweep = _load_sweep_module()

    command = sweep.build_inference_command("python", "ckpt.ckpt", "motion.motion", "0.1")

    assert command[-2:] == ["--overrides", "simulator.pd_target_max_accel=0.1"]
