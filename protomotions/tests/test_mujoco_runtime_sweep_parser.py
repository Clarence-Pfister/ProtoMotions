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
    module_path = repo_root / "scripts" / "mujoco_runtime_sweep.py"
    spec = importlib.util.spec_from_file_location("mujoco_runtime_sweep", module_path)
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
  eval/action_delta_mean_rad: 0.020000
  eval/gt_error/max: 0.064700
  eval/gt_error/mean: 0.033500
  eval/high_jerk_frame_percentage_mean: 0.000000
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
    assert metrics["eval/high_jerk_frame_percentage_mean"] == 0.0
    assert "Overall Score" not in metrics


def test_summary_table_uses_verdict_and_missing_values():
    sweep = _load_sweep_module()
    result_pass = sweep.SweepResult(
        run="baseline",
        command=[],
        log_path=Path("baseline.log"),
        returncode=0,
        timed_out=False,
        metrics={
            "eval/success_rate": 1.0,
            "eval/gt_error/mean": 0.0335,
            "eval/gt_error/max": 0.0647,
            "eval/normalized_jerk_mean": 724.5,
            "eval/action_delta_mean_rad": 0.02,
            "eval/high_jerk_frame_percentage_mean": 0.0,
        },
    )
    result_fail = sweep.SweepResult(
        run="ema_0.4",
        command=[],
        log_path=Path("ema_0.4.log"),
        returncode=1,
        timed_out=False,
        metrics={},
    )

    summary = sweep.build_summary([result_pass, result_fail])

    assert (
        "| baseline | 1 | 0.0335 | 0.0647 | 724.5 | 0.02 | 0 | PASS |"
        in summary
    )
    assert "| ema_0.4 | - | - | - | - | - | - | FAIL |" in summary
    assert "Best passing run: baseline." in summary


def test_build_inference_command_omits_baseline_override():
    sweep = _load_sweep_module()

    command = sweep.build_inference_command(
        "python",
        "ckpt.ckpt",
        "motion.motion",
        None,
    )

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


def test_build_inference_command_adds_ema_override():
    sweep = _load_sweep_module()

    command = sweep.build_inference_command("python", "ckpt.ckpt", "motion.motion", "0.6")

    assert command[-2:] == [
        "--overrides",
        "agent.evaluator.eval_action_ema_alpha=0.6",
    ]
