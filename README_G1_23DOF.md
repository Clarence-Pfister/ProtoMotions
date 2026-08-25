# G1 23DoF — `feature/g1-23dof`

Adds the Unitree **G1 23DoF** variant (the holosoma-compatible MJCF) as a
first-class robot in ProtoMotions, alongside the upstream 29DoF `g1`.

Everything below is run from the repo root with the IsaacLab environment
active:

```bash
cd ~/masumura/ProtoMotions
source env_isaaclab/bin/activate
```

Pass `--robot g1_23dof` (visualizer) or `--robot-name g1_23dof` (training)
wherever the upstream docs say `g1`. `inference_agent.py` has no robot flag —
it reads the robot from the checkpoint. **A policy trained for `g1` is not
loadable on `g1_23dof`**: the DoF count and body list differ.

## What this branch adds

| Path | Purpose |
|---|---|
| `protomotions/robot_configs/g1_23dof.py` | `G1_23DOFRobotConfig`, subclasses the 29DoF `G1RobotConfig` |
| `protomotions/robot_configs/factory.py` | registers the `g1_23dof` name |
| `protomotions/data/assets/mjcf/g1_23dof_holo_compat.xml` | MJCF used for training and as the qpos reference |
| `protomotions/data/assets/mjcf/g1_23dof_holo_compat_flat.xml` | flattened MJCF, input to USD conversion |
| `protomotions/data/assets/usd/g1_23dof_holo_compat/` | USD assets consumed by IsaacLab |
| `protomotions/data/assets/mesh/G1_23DOF/` | collision/visual meshes |
| `data/scripts/convert_qpos_npz_to_proto.py` | MuJoCo qpos `.npz` → `.motion` |
| `examples/motion_libs_visualizer.py` | accepts the new robot name |

Config specifics worth knowing: this MJCF has no separate head body, so
`head_body_name` maps to `torso_link`. Hands are
`left/right_wrist_roll_rubber_hand`, feet are `left/right_ankle_roll_link`, and
those five bodies plus `torso_link` form `trackable_bodies_subset`.

## 1. Motion → `.motion`

The 23DoF source path is a MuJoCo qpos `.npz` (holosoma retargeting output, or
the SOMA retargeter). The MJCF must be the one the qpos was produced against,
or the joint order will be silently wrong:

```bash
python data/scripts/convert_qpos_npz_to_proto.py \
    --input-dir  <dir-with-qpos-npz> \
    --output-dir <dir-with-qpos-npz>/proto \
    --mjcf-path  protomotions/data/assets/mjcf/g1_23dof_holo_compat.xml \
    --output-fps 30
```

`--input-fps` is read from each NPZ's scalar `fps` key when omitted. Other
options you may need:

- `--contact-source auto|embedded|heuristic|none` — where foot contacts come
  from. `--contact-labels-dir <dir>` (files named `*_contacts.npz`) overrides it.
- `--fix-height-per-frame / --no-fix-height-per-frame` — lifts frames that
  penetrate the ground; on by default, with `--fix-height-offset 0.02`.
  **Turn it off for motions with a flight phase**, or jumps get flattened.
- `--force-remake` — ignore existing outputs.
- `--yaml-output-name <name>.yaml` — emit a YAML listing the converted motions.

From a G1 CSV instead of qpos, use `convert_g1_csv_to_proto.py` (same
`--input-dir/--output-dir/--input-fps/--output-fps` shape).

## 2. Package into a MotionLib `.pt`

```bash
python protomotions/components/motion_lib.py \
    --motion-path <dir>/proto/<clip>.motion \
    --output-file <dir>/<clip>.pt
```

## 3. Visualize before training

Always do this — it catches retargeting bugs for free, and a bad retarget looks
like a training failure hours later:

```bash
python examples/motion_libs_visualizer.py \
    --motion_files <dir>/<clip>.pt \
    --robot g1_23dof --simulator isaaclab
```

## 4. Train a tracking policy

```bash
python protomotions/train_agent.py \
    --robot-name g1_23dof --simulator isaaclab \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name <clip>-23dof \
    --motion-file <dir>/<clip>.pt \
    --num-envs 2048 --batch-size 16384 --ngpu 1
```

Monitor with `tensorboard --logdir results/<clip>-23dof/lightning_logs`. An
empty `results/<clip>-23dof/failed_motions/*.txt` means every motion succeeded
at that evaluation.

On a 12 GB GPU, drop to `--num-envs 1024 --batch-size 8192` if you hit OOM.

## 5. Inspect the trained policy

```bash
python protomotions/inference_agent.py \
    --checkpoint results/<clip>-23dof/last.ckpt \
    --motion-file <dir>/<clip>.pt \
    --simulator isaaclab
```

Swap `--simulator mujoco` for a sim2sim check. The pretrained
`data/pretrained_models/motion_tracker/g1-bones-deploy` tracker is a **29DoF
`g1`** model and will not load here.

## Regenerating the USD assets

Only needed if the MJCF changes:

```bash
python usd_convert/flatten_mjcf.py \
    protomotions/data/assets/mjcf/g1_23dof_holo_compat.xml \
    --output protomotions/data/assets/mjcf/g1_23dof_holo_compat_flat.xml

python usd_convert/convert_robot_mjcf_to_usda.py \
    protomotions/data/assets/mjcf/g1_23dof_holo_compat_flat.xml \
    --output-dir protomotions/data/assets/usd/g1_23dof_holo_compat/
```

## Related

- [`feature/sim2real-tooling`](../../tree/feature/sim2real-tooling)
  — the video→dance pipeline. It targets the 29DoF `g1`; the two branches are
  independent and merge cleanly into `integration/all`.
- `Clarence-Pfister/soma-retargeter-g1-23dof` and
  `Clarence-Pfister/holosoma-g1-23dof-retargeting` — where the qpos `.npz`
  inputs come from.
