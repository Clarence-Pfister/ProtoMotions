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
"""Sweep MuJoCo PD target acceleration clamps for sim2sim validation."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CLAMP_VALUES = ["none", "0.2", "0.1", "0.05"]
SUMMARY_COLUMNS = [
    "success_rate",
    "gt_error_mean",
    "gt_error_max",
    "max_joint_error_mean",
    "normalized_jerk_mean",
    "action_delta_mean_rad",
    "action_delta_max_rad",
]
METRIC_KEYS = {
    "success_rate": ("eval/success_rate",),
    "gt_error_mean": ("eval/gt_error_mean", "eval/gt_error/mean"),
    "gt_error_max": ("eval/gt_error_max", "eval/gt_error/max"),
    "max_joint_error_mean": (
        "eval/max_joint_error_mean",
        "eval/max_joint_error/mean",
    ),
    "normalized_jerk_mean": ("eval/normalized_jerk_mean",),
    "action_delta_mean_rad": ("eval/action_delta_mean_rad",),
    "action_delta_max_rad": ("eval/action_delta_max_rad",),
}
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
EVAL_LINE_RE = re.compile(rf"^\s+([^:]+):\s+({FLOAT_PATTERN})\s*$", re.IGNORECASE)


@dataclass
class SweepResult:
    clamp: str
    command: list[str]
    log_path: Path
    returncode: int | None
    timed_out: bool
    metrics: dict[str, float]


def parse_eval_block(text: str) -> dict[str, float]:
    """Parse the full-eval EVALUATION RESULTS block from inference output."""
    lines = text.splitlines()
    try:
        start_idx = next(
            idx for idx, line in enumerate(lines) if line.strip() == "EVALUATION RESULTS"
        )
    except StopIteration:
        return {}

    metrics: dict[str, float] = {}
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("="):
            if metrics:
                break
            continue

        match = EVAL_LINE_RE.match(line)
        if match is None:
            continue

        key, value = match.groups()
        metrics[key.strip()] = float(value)

    return metrics


def normalize_clamp_value(value: str) -> str:
    """Normalize CLI clamp tokens while preserving user-facing values."""
    normalized = value.strip().lower()
    if normalized in {"none", "null", "off", "disabled"}:
        return "none"

    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Clamp value must be 'none' or a float, got {value!r}"
        ) from exc

    if numeric_value < 0:
        raise argparse.ArgumentTypeError("Clamp value must be non-negative")
    return value


def clamp_log_name(clamp: str) -> str:
    return f"clamp_{clamp}.log"


def build_inference_command(
    python_exe: str,
    checkpoint: str,
    motion_file: str,
    clamp: str,
) -> list[str]:
    command = [
        python_exe,
        "protomotions/inference_agent.py",
        "--checkpoint",
        checkpoint,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--headless",
        "--full-eval",
        "--motion-file",
        motion_file,
    ]
    if clamp != "none":
        command.extend(["--overrides", f"simulator.pd_target_max_accel={clamp}"])
    return command


def run_command(
    command: list[str],
    log_path: Path,
    timeout: int,
    repo_root: Path,
) -> tuple[int | None, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {timeout}s\n")
            return None, True

    return completed.returncode, False


def value_for_column(metrics: dict[str, float], column: str) -> float | None:
    for key in METRIC_KEYS[column]:
        if key in metrics:
            return metrics[key]
    return None


def verdict_for(metrics: dict[str, float], returncode: int | None, timed_out: bool) -> str:
    if timed_out or returncode != 0:
        return "FAIL"

    success_rate = value_for_column(metrics, "success_rate")
    gt_error_max = value_for_column(metrics, "gt_error_max")
    if success_rate == 1.0 and gt_error_max is not None and gt_error_max < 0.5:
        return "PASS"
    return "FAIL"


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def markdown_table(results: Iterable[SweepResult]) -> str:
    rows = []
    for result in results:
        row = [result.clamp]
        row.extend(format_metric(value_for_column(result.metrics, col)) for col in SUMMARY_COLUMNS)
        row.append(verdict_for(result.metrics, result.returncode, result.timed_out))
        rows.append(row)

    headers = ["clamp", *SUMMARY_COLUMNS, "verdict"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def best_passing_summary(results: list[SweepResult]) -> str:
    passing = [
        result
        for result in results
        if verdict_for(result.metrics, result.returncode, result.timed_out) == "PASS"
    ]
    if not passing:
        return "Best passing setting: no clamp value passed."

    if any(result.clamp == "none" for result in passing):
        return "Best passing setting: no clamp needed."

    numeric_passing = [result for result in passing if result.clamp != "none"]
    best = max(numeric_passing, key=lambda result: float(result.clamp))
    return f"Best passing setting: largest passing clamp is {best.clamp}."


def build_summary(results: list[SweepResult]) -> str:
    table = markdown_table(results)
    note = (
        "Note: inference clears termination components, so failures appear as "
        "tracking blow-up in gt_error metrics rather than early termination. "
        "Jitter should be read from normalized_jerk_mean and action_delta_*."
    )
    return f"{table}\n\n{best_passing_summary(results)}\n\n{note}\n"


def resolve_output_dir(output_dir: str | None, checkpoint: str, repo_root: Path) -> Path:
    if output_dir is None:
        output_path = Path("results") / "mujoco_sim2sim" / Path(checkpoint).stem
    else:
        output_path = Path(output_dir)

    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a checkpoint in MuJoCo across PD target acceleration clamps."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--motion-file", required=True, help="MuJoCo motion file to track.")
    parser.add_argument(
        "--clamp-values",
        nargs="+",
        default=DEFAULT_CLAMP_VALUES,
        type=normalize_clamp_value,
        help="Clamp values to sweep. Use 'none' to omit the override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for logs and summary.md. Defaults to "
            "results/mujoco_sim2sim/<checkpoint-stem>."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch protomotions/inference_agent.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(args.output_dir, args.checkpoint, repo_root)

    commands = [
        (
            clamp,
            build_inference_command(
                args.python,
                args.checkpoint,
                args.motion_file,
                clamp,
            ),
        )
        for clamp in args.clamp_values
    ]

    if args.dry_run:
        print(f"Output directory: {output_dir}")
        for _, command in commands:
            print(shlex.join(command))
        return 0

    results: list[SweepResult] = []
    for clamp, command in commands:
        log_path = output_dir / clamp_log_name(clamp)
        print(f"Running clamp={clamp}; log={log_path}")
        returncode, timed_out = run_command(command, log_path, args.timeout, repo_root)
        metrics = parse_eval_block(log_path.read_text(encoding="utf-8"))
        results.append(
            SweepResult(
                clamp=clamp,
                command=command,
                log_path=log_path,
                returncode=returncode,
                timed_out=timed_out,
                metrics=metrics,
            )
        )

    summary = build_summary(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
