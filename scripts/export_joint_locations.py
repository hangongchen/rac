#!/usr/bin/env python3
"""
Export per-frame URDF joint locations given joint angles.

- Reads angles from .csv (with header) or .npy
- Computes joint positions in world frame using URDF forward kinematics
- Optionally scales positions (e.g., USD is 50x bigger)
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from urdfpy import URDF


def _resolve_urdf_path(pre_skel: str) -> str:
    if pre_skel.endswith(".urdf") or "/" in pre_skel:
        return pre_skel
    mapping = {
        "a1": "mesh_material/a1/urdf/a1.urdf",
        "wolf": "mesh_material/wolf.urdf",
        "wolf_mod": "mesh_material/wolf_mod.urdf",
        "wolf_mod_revised": "mesh_material/wolf_mod_revised.urdf",
        "laikago": "mesh_material/laikago/laikago.urdf",
        "human": "mesh_material/human.urdf",
        "human_mod": "mesh_material/human_mod.urdf",
    }
    if pre_skel not in mapping:
        raise ValueError(f"Unknown pre_skel '{pre_skel}'. Provide a URDF path or a supported name.")
    return mapping[pre_skel]


def _load_angles(path: str) -> Tuple[List[str], np.ndarray]:
    if path.endswith(".npy"):
        angles = np.load(path)
        if angles.ndim == 1:
            angles = angles[None, :]
        return [], angles

    if not path.endswith(".csv"):
        raise ValueError("angles must be .csv (with header) or .npy")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = [h.strip() for h in header if h.strip()]
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    angles = np.array(rows, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    return names, angles


def _load_joint_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _link_name(link) -> str:
    return link.name if hasattr(link, "name") else str(link)


def _compute_joint_positions(urdf: URDF, joint_names: List[str], angles: np.ndarray) -> np.ndarray:
    # angles: T x J
    positions = []
    for t in range(angles.shape[0]):
        cfg: Dict[str, float] = {}
        for j, name in enumerate(joint_names):
            cfg[name] = float(angles[t, j])
        link_fk = urdf.link_fk(cfg=cfg)  # link -> 4x4
        link_fk_by_name = {_link_name(link): mat for link, mat in link_fk.items()}
        joints_t = []
        for name in joint_names:
            joint = urdf.joint_map[name]
            parent_link_name = _link_name(joint.parent)
            if parent_link_name not in link_fk_by_name:
                raise KeyError(f"Parent link '{parent_link_name}' not found in FK results")
            parent_T = link_fk_by_name[parent_link_name]
            joint_T = parent_T @ joint.origin
            joints_t.append(joint_T[:3, 3])
        positions.append(np.stack(joints_t, axis=0))
    return np.stack(positions, axis=0)


def _write_outputs(out_dir: str, prefix: str, joint_names: List[str], positions: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}.npy"), positions)
    # CSV: flatten to T x (J*3) with header
    header = []
    for name in joint_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    flat = positions.reshape(positions.shape[0], -1)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    np.savetxt(csv_path, flat, delimiter=",", header=",".join(header), comments="")
    # joint names
    with open(os.path.join(out_dir, f"{prefix}_joint_names.txt"), "w", encoding="utf-8") as f:
        for name in joint_names:
            f.write(f"{name}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", required=True, help="Path to angles.csv or angles.npy.")
    parser.add_argument("--joint_names", default=None, help="Optional joint_names.txt (required for .npy angles).")
    parser.add_argument("--pre_skel", default="", help="Predefined skeleton name or URDF path.")
    parser.add_argument("--urdf", default="", help="Explicit URDF path (overrides --pre_skel).")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: angles file directory).")
    parser.add_argument("--prefix", default="joint_locations", help="Output file prefix.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale positions (e.g., 50 for USD).")
    args = parser.parse_args()

    if not os.path.exists(args.angles):
        raise FileNotFoundError(args.angles)

    if args.urdf:
        urdf_path = args.urdf
    elif args.pre_skel:
        urdf_path = _resolve_urdf_path(args.pre_skel)
    else:
        raise ValueError("Provide --urdf or --pre_skel.")

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(urdf_path)

    names_from_csv, angles = _load_angles(args.angles)
    if args.joint_names:
        joint_names = _load_joint_names(args.joint_names)
    else:
        joint_names = names_from_csv

    if not joint_names:
        raise ValueError("Joint names not found. Provide --joint_names or use CSV with header.")

    if angles.shape[1] != len(joint_names):
        raise ValueError(
            f"Angle dimension mismatch: angles has {angles.shape[1]} cols, names has {len(joint_names)}"
        )

    urdf = URDF.load(urdf_path)
    positions = _compute_joint_positions(urdf, joint_names, angles)
    if args.scale != 1.0:
        positions = positions * float(args.scale)

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.angles))
    _write_outputs(out_dir, args.prefix, joint_names, positions)
    print(f"Saved joint locations to {out_dir} with prefix '{args.prefix}'")


if __name__ == "__main__":
    main()
