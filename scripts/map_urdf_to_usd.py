#!/usr/bin/env python3
"""
Generate a YAML mapping between URDF links/joints and USD mesh prims by
matching rest-pose link positions to mesh centroids.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from urdfpy import URDF

from pxr import Usd, UsdGeom


def _load_usd_mesh_centroids(usd_path: str, root_prim_path: Optional[str]) -> Dict[str, np.ndarray]:
    stage = Usd.Stage.Open(usd_path)
    if root_prim_path:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Root prim '{root_prim_path}' not found in USD.")
    else:
        root_prim = stage.GetPseudoRoot()

    xform_cache = UsdGeom.XformCache()
    centroids: Dict[str, np.ndarray] = {}

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        pts = np.array(points, dtype=np.float64)
        xf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        # Apply transform to points (homogeneous)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts_w = (xf @ pts_h.T).T[:, :3]
        centroid = pts_w.mean(axis=0)
        centroids[prim.GetPath().pathString] = centroid

    if not centroids:
        raise ValueError("No mesh prims found in USD.")
    return centroids


def _load_urdf_link_positions(urdf_path: str) -> Dict[str, np.ndarray]:
    urdf = URDF.load(urdf_path)
    link_fk = urdf.link_fk(cfg=None)
    link_pos: Dict[str, np.ndarray] = {}
    for link, mat in link_fk.items():
        link_pos[link.name] = np.array(mat[:3, 3], dtype=np.float64)
    return link_pos


def _greedy_match(
    link_pos: Dict[str, np.ndarray],
    mesh_pos: Dict[str, np.ndarray],
) -> Tuple[Dict[str, str], List[Tuple[str, str, float]]]:
    link_names = list(link_pos.keys())
    mesh_names = list(mesh_pos.keys())
    pairs: List[Tuple[str, str, float]] = []
    for ln in link_names:
        for mn in mesh_names:
            d = np.linalg.norm(link_pos[ln] - mesh_pos[mn])
            pairs.append((ln, mn, float(d)))

    pairs.sort(key=lambda x: x[2])
    assigned_links = set()
    assigned_meshes = set()
    mapping: Dict[str, str] = {}
    debug_pairs: List[Tuple[str, str, float]] = []
    for ln, mn, d in pairs:
        if ln in assigned_links or mn in assigned_meshes:
            continue
        mapping[ln] = mn
        assigned_links.add(ln)
        assigned_meshes.add(mn)
        debug_pairs.append((ln, mn, d))
    return mapping, debug_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Path to USD file.")
    parser.add_argument("--urdf", required=True, help="Path to URDF file.")
    parser.add_argument("--out", required=True, help="Output YAML mapping file.")
    parser.add_argument("--root_prim", default=None, help="Root prim path to search meshes under.")
    parser.add_argument("--usd_scale", type=float, default=1.0, help="Optional scale for USD (default 1.0).")
    parser.add_argument("--relative", action="store_true", help="Match using positions relative to root.")
    args = parser.parse_args()

    usd_centroids = _load_usd_mesh_centroids(args.usd, args.root_prim)
    urdf_link_pos = _load_urdf_link_positions(args.urdf)

    # Optional scale (if user wants it)
    if args.usd_scale != 1.0:
        usd_centroids = {k: v / args.usd_scale for k, v in usd_centroids.items()}

    # Use relative positions (centered at root) if requested
    if args.relative:
        # USD: center at mean of all mesh centroids
        usd_center = np.mean(np.stack(list(usd_centroids.values()), axis=0), axis=0)
        usd_centroids = {k: v - usd_center for k, v in usd_centroids.items()}
        # URDF: center at base_link if present, else mean of link positions
        if "base_link" in urdf_link_pos:
            urdf_center = urdf_link_pos["base_link"]
        else:
            urdf_center = np.mean(np.stack(list(urdf_link_pos.values()), axis=0), axis=0)
        urdf_link_pos = {k: v - urdf_center for k, v in urdf_link_pos.items()}

    mapping, debug_pairs = _greedy_match(urdf_link_pos, usd_centroids)

    # Build joint->mesh mapping (child link)
    urdf = URDF.load(args.urdf)
    joint_to_mesh: Dict[str, str] = {}
    for joint in urdf.joints:
        child = joint.child
        if child in mapping:
            joint_to_mesh[joint.name] = mapping[child]

    out = {
        "usd": os.path.abspath(args.usd),
        "urdf": os.path.abspath(args.urdf),
        "usd_scale": args.usd_scale,
        "relative": args.relative,
        "root_prim": args.root_prim,
        "link_to_mesh": mapping,
        "joint_to_mesh": joint_to_mesh,
        "debug": [{"link": ln, "mesh": mn, "distance": d} for ln, mn, d in debug_pairs],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"Wrote mapping to {args.out}")


if __name__ == "__main__":
    main()
