# USD Adjustment Scripts README

This document explains how to use the USD adjustment scripts in `scripts/` for:

- building `mapping.yaml` from URDF + USD,
- generating joints into USD,
- adding PhysX attachments,
- setting rigid-body density on mapped link meshes.


## Environment

Use the `rac` conda environment:

```bash
conda activate rac
```

All commands below assume you run them from:

```bash
/home/hangong/Reproduction/rac
```


## Scripts

- `scripts/map_urdf_to_usd.py`
  - Generates `mapping.yaml` (`link_to_mesh`, `joint_to_mesh`) by matching URDF link positions to USD mesh centroids.
- `scripts/urdf_joints_to_usd.py`
  - Creates joints in USD from URDF + mapping.
  - Supports classic joint creation or collapsing `_R/_P/_Y` chains to one D6-style `UsdPhysics.Joint`.
- `scripts/add_physx_attachments.py`
  - Creates/updates `PhysxPhysicsAttachment` prims between one soft body mesh and all mapped link meshes.
- `scripts/assign_density_to_links.py`
  - Applies `UsdPhysics.MassAPI` density to all mapped link meshes.


## 1) Build Mapping YAML

Template:

```bash
python scripts/map_urdf_to_usd.py \
  --usd <path/to/input.usd> \
  --urdf <path/to/template.urdf> \
  --out mapping.yaml \
  --relative
```

Cat example:

```bash
python scripts/map_urdf_to_usd.py \
  --usd /home/hangong/Reproduction/FISH/source/FISH/FISH/tasks/direct/fish/agents/cat/cat.usd \
  --urdf mesh_material/wolf_mod_revised.urdf \
  --out mapping.yaml \
  --relative
```

Notes:

- Use absolute USD paths when the USD is outside this repo.
- `mapping.yaml` is usually manually corrected after auto-mapping.


## 2) Generate Joints in USD

### Recommended mode (collapse R/P/Y to one D6-style joint)

Template:

```bash
python scripts/urdf_joints_to_usd.py \
  --usd <path/to/input.usd> \
  --urdf <path/to/template.urdf> \
  --mapping mapping.yaml \
  --out <path/to/output.usd> \
  --joint_pos gap_midpoint \
  --override_revolute_limits_deg 180 \
  --collapse_rpy_to_d6 \
  --ensure_rigid_bodies \
  --ensure_collision \
  --d6_drive \
  --d6_drive_stiffness 50000 \
  --d6_drive_damping 5000 \
  --d6_drive_max_force 1000000
```

Cat example:

```bash
python scripts/urdf_joints_to_usd.py \
  --usd /home/hangong/Reproduction/FISH/source/FISH/FISH/tasks/direct/fish/agents/cat/cat.usd \
  --urdf mesh_material/wolf_mod_revised.urdf \
  --mapping mapping.yaml \
  --out /home/hangong/Reproduction/FISH/source/FISH/FISH/tasks/direct/fish/agents/cat/cat_with_joints_from_mapping_gap_midpoint_grouped_limits180_d6_rb_drive.usd \
  --joint_pos gap_midpoint \
  --override_revolute_limits_deg 180 \
  --collapse_rpy_to_d6 \
  --ensure_rigid_bodies \
  --ensure_collision \
  --d6_drive \
  --d6_drive_stiffness 50000 \
  --d6_drive_damping 5000 \
  --d6_drive_max_force 1000000
```

Useful switches:

- `--joint_pos {urdf,parent,child,midpoint,gap_midpoint}`
- `--joint_xyz_csv <csv>` to place joints from CSV (`joint,x,y,z`)
- `--d6_drive_type {force,acceleration}`
- `--d6_drive_target_pos`, `--d6_drive_target_vel`

If you do **not** use `--collapse_rpy_to_d6`, the script creates revolute/prismatic/fixed joints directly and skips any parent-child pair that maps to the same mesh prim.


## 3) Add PhysX Attachments

Template:

```bash
python scripts/add_physx_attachments.py \
  --usd <path/to/jointed.usd> \
  --mapping mapping.yaml \
  --soft_body /root/mesh_rest_999/mesh_rest_999_002
```

Cat example:

```bash
python scripts/add_physx_attachments.py \
  --usd /home/hangong/Reproduction/FISH/source/FISH/FISH/tasks/direct/fish/agents/cat/cat_with_joints_from_mapping_gap_midpoint_grouped_limits180_d6_rb_drive.usd \
  --mapping mapping.yaml \
  --soft_body /root/mesh_rest_999/mesh_rest_999_002
```

Notes:

- In-place by default.
- Use `--out <new.usd>` to write a new file.


## 4) Set Density on Mapped Link Meshes

Template:

```bash
python scripts/assign_density_to_links.py \
  --usd <path/to/jointed.usd> \
  --mapping mapping.yaml \
  --density 1000
```

Cat example:

```bash
python scripts/assign_density_to_links.py \
  --usd /home/hangong/Reproduction/FISH/source/FISH/FISH/tasks/direct/fish/agents/cat/cat_with_joints_from_mapping_gap_midpoint_grouped_limits180_d6_rb_drive.usd \
  --mapping mapping.yaml \
  --density 1000
```

Notes:

- In-place by default.
- Use `--out <new.usd>` to write a new file.
- This script only sets density; it does not provide a `clear/remove` flag.


## Troubleshooting

- `Failed to open layer @...usd@`
  - Use an absolute path for `--usd`.
- `CreateJoint - you cannot create a joint between a body and itself`
  - Ensure mapped links point to distinct rigid bodies where expected.
  - For URDF `_R/_P/_Y` chains, prefer `--collapse_rpy_to_d6`.
  - Use `--ensure_rigid_bodies` so mapped link meshes are real rigid bodies.
- Robot settles into strange pose after sim starts
  - Tune `--d6_drive_stiffness`, `--d6_drive_damping`, and `--d6_drive_max_force`.
  - Verify joint frames/placement mode (`--joint_pos` or `--joint_xyz_csv`).
