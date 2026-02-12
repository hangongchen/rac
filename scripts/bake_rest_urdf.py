#!/usr/bin/env python3
"""
Bake a rest pose into a URDF by folding joint angles into joint origins.

Result: zero joint angles reproduce the provided rest pose.
"""

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from urdfpy import URDF


def _load_angles_csv(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"No data rows in {path}")
    if len(row) != len(header):
        raise ValueError("CSV row length does not match header length")
    return {name.strip(): float(val) for name, val in zip(header, row) if name.strip()}


def _rot_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _motion_transform(joint, angle: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if joint.joint_type in ("revolute", "continuous"):
        R = _rot_from_axis_angle(joint.axis, angle)
        T[:3, :3] = R
    elif joint.joint_type == "prismatic":
        axis = np.asarray(joint.axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n > 1e-8:
            axis = axis / n
        T[:3, 3] = axis * angle
    # fixed: identity
    return T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Input URDF path.")
    parser.add_argument("--angles_csv", required=True, help="CSV with header joint names and one row of angles.")
    parser.add_argument("--out", required=True, help="Output URDF path.")
    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        raise FileNotFoundError(args.urdf)
    if not os.path.exists(args.angles_csv):
        raise FileNotFoundError(args.angles_csv)

    angles = _load_angles_csv(args.angles_csv)
    urdf = URDF.load(args.urdf)

    for joint in urdf.joints:
        if joint.joint_type == "fixed":
            continue
        if joint.name not in angles:
            continue
        angle = angles[joint.name]
        motion = _motion_transform(joint, angle)
        # origin is a 4x4 transform from parent link to joint frame
        joint.origin = joint.origin @ motion

    urdf.save(args.out)
    print(f"Wrote baked URDF to {args.out}")


if __name__ == "__main__":
    main()
