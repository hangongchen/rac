#!/usr/bin/env python3
"""
Create USD Physics joints from a URDF kinematic tree using a link->mesh mapping.

This is intended for verification of a mapping: it connects parent/child mesh prims
and writes joints under a dedicated scope.
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from urdfpy import URDF

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def _sanitize_prim_name(name: str) -> str:
    # USD prim names must be valid identifiers (no dots, spaces, etc.).
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe:
        safe = "joint"
    if not re.match(r"[A-Za-z_]", safe[0]):
        safe = f"j_{safe}"
    return safe


def _unique_name(base: str, used: set) -> str:
    name = base
    idx = 1
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def _ensure_scope(stage: Usd.Stage, scope_path: str) -> None:
    if not scope_path.startswith("/"):
        raise ValueError(f"scope_path must be absolute, got: {scope_path}")
    parts = [p for p in scope_path.split("/") if p]
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        prim = stage.GetPrimAtPath(cur)
        if not prim or not prim.IsValid():
            UsdGeom.Scope.Define(stage, cur)


def _mat_to_quat(rot: np.ndarray) -> Tuple[float, float, float, float]:
    # Returns (w, x, y, z)
    m = rot
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return w, x, y, z


def _axis_to_token(axis: np.ndarray) -> str:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or np.allclose(axis, 0.0):
        return "X"
    idx = int(np.argmax(np.abs(axis)))
    return ["X", "Y", "Z"][idx]


def _get_joint_type(joint) -> Optional[str]:
    jt = getattr(joint, "joint_type", None)
    if jt is None:
        jt = getattr(joint, "type", None)
    if jt is None:
        return None
    return str(jt)


def _link_name(link) -> str:
    return link.name if hasattr(link, "name") else str(link)


def _prim_world_origin(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    xf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    return xf[:3, 3]


def _prim_world_centroid(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points:
            pts = np.array(points, dtype=np.float64)
            xf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
            pts_w = (xf @ pts_h.T).T[:, :3]
            return pts_w.mean(axis=0)
    return _prim_world_origin(xform_cache, prim)


def _world_to_local_pos(xform_cache: UsdGeom.XformCache, prim: Usd.Prim, world_pos: np.ndarray) -> np.ndarray:
    xf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    inv = np.linalg.inv(xf)
    hp = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float64)
    lp = inv @ hp
    return lp[:3]


def _prim_mesh_centroid_or_origin(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    """Return mesh centroid if prim is a mesh, otherwise fall back to origin."""
    if prim.IsA(UsdGeom.Mesh):
        return _prim_world_centroid(xform_cache, prim)
    return _prim_world_origin(xform_cache, prim)


def _prim_mesh_world_points(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> Optional[np.ndarray]:
    if not prim.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    pts = np.array(points, dtype=np.float64)
    xf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_w = (xf @ pts_h.T).T[:, :3]
    return pts_w


def _mesh_gap_midpoint(
    xform_cache: UsdGeom.XformCache,
    prim_a: Usd.Prim,
    prim_b: Usd.Prim,
    max_points: int = 2000,
) -> Optional[np.ndarray]:
    pts_a = _prim_mesh_world_points(xform_cache, prim_a)
    pts_b = _prim_mesh_world_points(xform_cache, prim_b)
    if pts_a is None or pts_b is None:
        return None
    # downsample for speed
    if pts_a.shape[0] > max_points:
        step = max(1, pts_a.shape[0] // max_points)
        pts_a = pts_a[::step]
    if pts_b.shape[0] > max_points:
        step = max(1, pts_b.shape[0] // max_points)
        pts_b = pts_b[::step]
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    tree = cKDTree(pts_b)
    d, idx = tree.query(pts_a, k=1)
    i = int(np.argmin(d))
    pa = pts_a[i]
    pb = pts_b[int(idx[i])]
    return 0.5 * (pa + pb)


def _split_rpy_suffix(joint_name: str) -> Tuple[str, Optional[str]]:
    if joint_name.endswith(("_R", "_P", "_Y")):
        return joint_name[:-2], joint_name[-1]
    return joint_name, None


def _apply_limit_api(prim: Usd.Prim, axis_name: str, low: Optional[float], high: Optional[float]) -> None:
    limit_api = UsdPhysics.LimitAPI.Apply(prim, axis_name)
    if low is not None:
        limit_api.CreateLowAttr().Set(float(low))
    if high is not None:
        limit_api.CreateHighAttr().Set(float(high))


def _apply_drive_api(
    prim: Usd.Prim,
    axis_name: str,
    drive_type: str,
    stiffness: float,
    damping: float,
    max_force: float,
    target_pos: float,
    target_vel: float,
) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, axis_name)
    drive.CreateTypeAttr().Set(drive_type)
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(float(target_pos))
    drive.CreateTargetVelocityAttr().Set(float(target_vel))


def _set_joint_local_from_world(
    joint_prim,
    xform_cache: UsdGeom.XformCache,
    parent_prim: Usd.Prim,
    child_prim: Usd.Prim,
    world_pos: np.ndarray,
) -> None:
    lp0 = _world_to_local_pos(xform_cache, parent_prim, world_pos)
    lp1 = _world_to_local_pos(xform_cache, child_prim, world_pos)
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(lp0[0]), float(lp0[1]), float(lp0[2])))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(float(lp1[0]), float(lp1[1]), float(lp1[2])))
    joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def _set_joint_local_from_local(joint_prim, pos: np.ndarray) -> None:
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
    joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def _set_joint_local_from_urdf_origin(joint_prim, joint) -> None:
    origin = getattr(joint, "origin", None)
    if origin is not None:
        mat = np.array(origin, dtype=np.float64)
        t = mat[:3, 3]
        r = mat[:3, :3]
        w, x, y, z = _mat_to_quat(r)
        joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    else:
        joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def _ensure_rigid_body(prim: Usd.Prim, enable_collision: bool) -> None:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    if enable_collision:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)


def _compute_world_pos_from_meshes(
    xform_cache: UsdGeom.XformCache,
    parent_prim: Usd.Prim,
    child_prim: Usd.Prim,
    joint_pos: str,
) -> np.ndarray:
    p_pos = _prim_world_centroid(xform_cache, parent_prim)
    c_pos = _prim_world_centroid(xform_cache, child_prim)
    if joint_pos == "parent":
        return p_pos
    if joint_pos == "child":
        return c_pos
    if joint_pos == "gap_midpoint":
        gap_mid = _mesh_gap_midpoint(xform_cache, parent_prim, child_prim)
        return gap_mid if gap_mid is not None else 0.5 * (p_pos + c_pos)
    return 0.5 * (p_pos + c_pos)


def _load_joint_xyz(csv_path: str) -> Dict[str, np.ndarray]:
    joint_xyz: Dict[str, np.ndarray] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        # expect columns: joint,x,y,z
        required = {"joint", "x", "y", "z"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns {sorted(required)}. Found: {reader.fieldnames}"
            )
        for row in reader:
            name = row["joint"].strip()
            if not name:
                continue
            joint_xyz[name] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    if not joint_xyz:
        raise ValueError(f"No joint rows found in {csv_path}")
    return joint_xyz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Input USD file.")
    parser.add_argument("--urdf", required=True, help="URDF file.")
    parser.add_argument("--mapping", required=True, help="YAML mapping file with link_to_mesh.")
    parser.add_argument("--out", required=True, help="Output USD file.")
    parser.add_argument(
        "--joint_root",
        default="/JointsFromUrdf",
        help="USD path to create joints under (default: /JointsFromUrdf).",
    )
    parser.add_argument(
        "--articulation_root",
        default=None,
        help="Optional prim path to apply ArticulationRootAPI.",
    )
    parser.add_argument(
        "--joint_pos",
        default="urdf",
        choices=["urdf", "parent", "child", "midpoint", "gap_midpoint"],
        help="Joint placement mode: 'urdf' uses URDF origin; "
        "'parent'/'child' use mesh centroid; 'midpoint' averages both; "
        "'gap_midpoint' uses closest points between parent/child meshes.",
    )
    parser.add_argument(
        "--joint_xyz_csv",
        default=None,
        help="CSV with columns joint,x,y,z (world-space joint positions). Overrides --joint_pos.",
    )
    parser.add_argument(
        "--joint_xyz_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to joint_xyz positions (default 1.0).",
    )
    parser.add_argument(
        "--joint_xyz_translate",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "TZ"),
        help="Translation applied to joint_xyz positions (after scale), in world units.",
    )
    parser.add_argument(
        "--joint_xyz_space",
        default="world",
        choices=["world", "local"],
        help="Interpret joint_xyz as world-space (default) or write directly into joint localPos.",
    )
    parser.add_argument(
        "--joint_xyz_origin_link",
        default=None,
        help="Link name whose USD position should align with joint_xyz origin (e.g., base_link).",
    )
    parser.add_argument(
        "--joint_xyz_origin_mode",
        default="xform",
        choices=["xform", "mesh"],
        help="When aligning joint_xyz origin, use prim xform or mesh centroid.",
    )
    parser.add_argument(
        "--joint_xyz_align_to_mesh",
        action="store_true",
        help="Align joint_xyz by a global translation so joint positions match mapped mesh centroids.",
    )
    parser.add_argument(
        "--joint_xyz_align_mode",
        default="mesh",
        choices=["mesh", "xform"],
        help="When aligning to mesh, use mesh centroid (default) or prim xform.",
    )
    parser.add_argument(
        "--joint_xyz_align_target",
        default="child",
        choices=["child", "parent", "midpoint"],
        help="Target used for alignment offset when aligning to mesh.",
    )
    parser.add_argument(
        "--override_revolute_limits_deg",
        type=float,
        default=None,
        help="If set, override all revolute/continuous joint limits to +/- this value (degrees).",
    )
    parser.add_argument(
        "--collapse_rpy_to_d6",
        action="store_true",
        help="Collapse _R/_P/_Y joint chains into a single D6 (PhysicsJoint) with rotX/rotY/rotZ limits.",
    )
    parser.add_argument(
        "--ensure_rigid_bodies",
        action="store_true",
        help="Apply RigidBodyAPI to all mapped link meshes before creating joints.",
    )
    parser.add_argument(
        "--ensure_collision",
        action="store_true",
        help="When ensuring rigid bodies, also apply CollisionAPI to mapped link meshes.",
    )
    parser.add_argument(
        "--d6_drive",
        action="store_true",
        help="Enable angular drives on collapsed D6 joints (rotX/rotY/rotZ).",
    )
    parser.add_argument(
        "--d6_drive_stiffness",
        type=float,
        default=5e4,
        help="Stiffness for D6 angular drives.",
    )
    parser.add_argument(
        "--d6_drive_damping",
        type=float,
        default=5e3,
        help="Damping for D6 angular drives.",
    )
    parser.add_argument(
        "--d6_drive_max_force",
        type=float,
        default=1e6,
        help="Max force for D6 angular drives.",
    )
    parser.add_argument(
        "--d6_drive_type",
        choices=["force", "acceleration"],
        default="force",
        help="Drive type for D6 angular drives.",
    )
    parser.add_argument(
        "--d6_drive_target_pos",
        type=float,
        default=0.0,
        help="Target position for D6 angular drives (radians).",
    )
    parser.add_argument(
        "--d6_drive_target_vel",
        type=float,
        default=0.0,
        help="Target velocity for D6 angular drives.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.usd):
        raise FileNotFoundError(args.usd)
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(args.urdf)
    if not os.path.exists(args.mapping):
        raise FileNotFoundError(args.mapping)

    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping_data = yaml.safe_load(f)
    link_to_mesh: Dict[str, str] = mapping_data.get("link_to_mesh", {})
    if not link_to_mesh:
        raise ValueError("mapping.yaml has no link_to_mesh entries.")

    urdf = URDF.load(args.urdf)
    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args.usd}")

    if args.ensure_rigid_bodies:
        for mesh_path in set(link_to_mesh.values()):
            prim = stage.GetPrimAtPath(mesh_path)
            if not prim or not prim.IsValid():
                continue
            _ensure_rigid_body(prim, args.ensure_collision)

    _ensure_scope(stage, args.joint_root)

    rpy_groups: Dict[str, Dict[str, object]] = {}
    rpy_joint_names: set = set()
    if args.collapse_rpy_to_d6:
        for j in urdf.joints:
            base, suffix = _split_rpy_suffix(j.name)
            if suffix is None:
                continue
            rpy_groups.setdefault(base, {})[suffix] = j
            rpy_joint_names.add(j.name)

    if args.articulation_root:
        root_prim = stage.GetPrimAtPath(args.articulation_root)
        if not root_prim or not root_prim.IsValid():
            raise ValueError(f"articulation_root not found: {args.articulation_root}")
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    xform_cache = UsdGeom.XformCache()
    joint_xyz = None
    joint_xyz_offset = np.zeros(3, dtype=np.float64)
    if args.joint_xyz_csv:
        if not os.path.exists(args.joint_xyz_csv):
            raise FileNotFoundError(args.joint_xyz_csv)
        joint_xyz = _load_joint_xyz(args.joint_xyz_csv)
        # Optional: compute a global translation offset so CSV joint positions align with mapped mesh centroids.
        if args.joint_xyz_align_to_mesh:
            pairs = []
            for j in urdf.joints:
                if j.name not in joint_xyz:
                    continue
                parent_name = _link_name(j.parent)
                child_name = _link_name(j.child)
                if parent_name not in link_to_mesh or child_name not in link_to_mesh:
                    continue
                parent_prim = stage.GetPrimAtPath(link_to_mesh[parent_name])
                child_prim = stage.GetPrimAtPath(link_to_mesh[child_name])
                if not parent_prim or not parent_prim.IsValid() or not child_prim or not child_prim.IsValid():
                    continue
                if args.joint_xyz_align_mode == "mesh":
                    p_pos = _prim_mesh_centroid_or_origin(xform_cache, parent_prim)
                    c_pos = _prim_mesh_centroid_or_origin(xform_cache, child_prim)
                else:
                    p_pos = _prim_world_origin(xform_cache, parent_prim)
                    c_pos = _prim_world_origin(xform_cache, child_prim)
                if args.joint_xyz_align_target == "parent":
                    target = p_pos
                elif args.joint_xyz_align_target == "child":
                    target = c_pos
                else:
                    target = 0.5 * (p_pos + c_pos)
                csv_pos = joint_xyz[j.name] * float(args.joint_xyz_scale)
                pairs.append(target - csv_pos)
            if pairs:
                joint_xyz_offset = np.mean(np.stack(pairs, axis=0), axis=0)
        if args.joint_xyz_origin_link and args.joint_xyz_space == "world":
            if args.joint_xyz_origin_link not in link_to_mesh:
                raise ValueError(
                    f"joint_xyz_origin_link '{args.joint_xyz_origin_link}' not found in link_to_mesh mapping"
                )
            origin_prim_path = link_to_mesh[args.joint_xyz_origin_link]
            origin_prim = stage.GetPrimAtPath(origin_prim_path)
            if not origin_prim or not origin_prim.IsValid():
                raise ValueError(f"Origin prim not found: {origin_prim_path}")
            if args.joint_xyz_origin_mode == "mesh":
                usd_origin = _prim_mesh_centroid_or_origin(xform_cache, origin_prim)
            else:
                usd_origin = _prim_world_origin(xform_cache, origin_prim)
            csv_origin = joint_xyz.get(args.joint_xyz_origin_link, np.zeros(3, dtype=np.float64))
            joint_xyz_offset = usd_origin - (csv_origin * float(args.joint_xyz_scale))

    used_names = set()
    created: List[str] = []
    created_d6: List[str] = []
    skipped: List[str] = []
    skipped_d6: List[str] = []
    skipped_rpy: List[str] = []
    skipped_same_body: List[str] = []

    # Precompute anchor joint world positions for R/P/Y chains when using mesh-based placement.
    anchor_world_pos: Dict[str, np.ndarray] = {}
    if joint_xyz is None and args.joint_pos != "urdf":
        suffix_priority = {"R": 0, "P": 1, "Y": 2}
        anchor_joint_for_base: Dict[str, str] = {}
        anchor_suffix_for_base: Dict[str, str] = {}
        for j in urdf.joints:
            base, suffix = _split_rpy_suffix(j.name)
            if suffix is None:
                continue
            parent_name = _link_name(j.parent)
            child_name = _link_name(j.child)
            if parent_name not in link_to_mesh or child_name not in link_to_mesh:
                continue
            if link_to_mesh[parent_name] == link_to_mesh[child_name]:
                continue
            prev = anchor_suffix_for_base.get(base)
            if prev is None or suffix_priority[suffix] < suffix_priority.get(prev, 99):
                anchor_joint_for_base[base] = j.name
                anchor_suffix_for_base[base] = suffix
        for base, jname in anchor_joint_for_base.items():
            j = urdf.joint_map[jname]
            parent_name = _link_name(j.parent)
            child_name = _link_name(j.child)
            parent_prim = stage.GetPrimAtPath(link_to_mesh[parent_name])
            child_prim = stage.GetPrimAtPath(link_to_mesh[child_name])
            if not parent_prim or not parent_prim.IsValid() or not child_prim or not child_prim.IsValid():
                continue
            anchor_world_pos[base] = _compute_world_pos_from_meshes(
                xform_cache, parent_prim, child_prim, args.joint_pos
            )

    # Optionally collapse R/P/Y chains into a single D6 joint.
    if args.collapse_rpy_to_d6 and rpy_groups:
        axis_map = {"R": "rotX", "P": "rotY", "Y": "rotZ"}
        for base, group in rpy_groups.items():
            anchor = None
            for suffix in ("R", "P", "Y"):
                j = group.get(suffix)
                if j is None:
                    continue
                parent_link = _link_name(j.parent)
                child_link = _link_name(j.child)
                parent_prim_path = link_to_mesh.get(parent_link)
                child_prim_path = link_to_mesh.get(child_link)
                if not parent_prim_path or not child_prim_path:
                    continue
                if parent_prim_path != child_prim_path:
                    anchor = j
                    break
            if anchor is None:
                skipped_d6.append(base)
                continue

            parent_link = _link_name(anchor.parent)
            child_link = _link_name(anchor.child)
            parent_prim_path = link_to_mesh.get(parent_link)
            child_prim_path = link_to_mesh.get(child_link)
            if not parent_prim_path or not child_prim_path:
                skipped_d6.append(base)
                continue
            parent_prim = stage.GetPrimAtPath(parent_prim_path)
            child_prim = stage.GetPrimAtPath(child_prim_path)
            if not parent_prim or not parent_prim.IsValid() or not child_prim or not child_prim.IsValid():
                skipped_d6.append(base)
                continue
            if parent_prim_path == child_prim_path:
                skipped_d6.append(base)
                continue

            # Resolve placement for the collapsed joint.
            local_pos = None
            world_pos = None
            if joint_xyz is not None:
                if anchor.name not in joint_xyz:
                    skipped_d6.append(base)
                    continue
                pos = joint_xyz[anchor.name] * float(args.joint_xyz_scale)
                if args.joint_xyz_translate is not None:
                    pos = pos + np.array(args.joint_xyz_translate, dtype=np.float64)
                if args.joint_xyz_space == "world":
                    world_pos = pos + joint_xyz_offset
                else:
                    local_pos = pos
            elif args.joint_pos == "urdf":
                pass
            else:
                world_pos = anchor_world_pos.get(base)
                if world_pos is None:
                    world_pos = _compute_world_pos_from_meshes(
                        xform_cache, parent_prim, child_prim, args.joint_pos
                    )

            safe_name = _unique_name(_sanitize_prim_name(base), used_names)
            joint_path = f"{args.joint_root}/{safe_name}"
            joint_prim = UsdPhysics.Joint.Define(stage, joint_path)
            prim = joint_prim.GetPrim()

            joint_prim.CreateBody0Rel().SetTargets([parent_prim_path])
            joint_prim.CreateBody1Rel().SetTargets([child_prim_path])
            joint_prim.CreateJointEnabledAttr().Set(True)

            if joint_xyz is not None:
                if local_pos is not None:
                    _set_joint_local_from_local(joint_prim, local_pos)
                else:
                    _set_joint_local_from_world(joint_prim, xform_cache, parent_prim, child_prim, world_pos)
            elif args.joint_pos == "urdf":
                _set_joint_local_from_urdf_origin(joint_prim, anchor)
            else:
                _set_joint_local_from_world(joint_prim, xform_cache, parent_prim, child_prim, world_pos)

            # Lock translations.
            for axis in ("transX", "transY", "transZ"):
                _apply_limit_api(prim, axis, 1.0, -1.0)

            # Apply rotation limits per axis.
            for suffix, axis_name in axis_map.items():
                j = group.get(suffix)
                if j is None:
                    continue
                low = None
                high = None
                if args.override_revolute_limits_deg is not None and _get_joint_type(j) in ("revolute", "continuous"):
                    lim = float(args.override_revolute_limits_deg)
                    low, high = -lim, lim
                else:
                    if _get_joint_type(j) != "continuous":
                        limit = getattr(j, "limit", None)
                        if limit is not None:
                            low = getattr(limit, "lower", None)
                            high = getattr(limit, "upper", None)
                if low is None and high is None:
                    continue
                _apply_limit_api(prim, axis_name, low, high)

            # Optional angular drives to stabilize the rest pose.
            if args.d6_drive:
                for suffix, axis_name in axis_map.items():
                    if group.get(suffix) is None:
                        continue
                    _apply_drive_api(
                        prim,
                        axis_name,
                        args.d6_drive_type,
                        args.d6_drive_stiffness,
                        args.d6_drive_damping,
                        args.d6_drive_max_force,
                        args.d6_drive_target_pos,
                        args.d6_drive_target_vel,
                    )

            prim.SetDisplayName(base)
            prim.CreateAttribute("urdf:jointName", Sdf.ValueTypeNames.String).Set(base)
            prim.CreateAttribute("urdf:rpyJoints", Sdf.ValueTypeNames.StringArray).Set(
                [j.name for j in group.values()]
            )
            prim.CreateAttribute("urdf:parent", Sdf.ValueTypeNames.String).Set(parent_link)
            prim.CreateAttribute("urdf:child", Sdf.ValueTypeNames.String).Set(child_link)

            created_d6.append(base)

    for joint in urdf.joints:
        joint_name = joint.name
        if args.collapse_rpy_to_d6 and joint_name in rpy_joint_names:
            skipped_rpy.append(joint_name)
            continue
        joint_type = _get_joint_type(joint)
        if joint_type is None:
            skipped.append(joint_name)
            continue

        parent_link = _link_name(joint.parent)
        child_link = _link_name(joint.child)
        parent_prim_path = link_to_mesh.get(parent_link)
        child_prim_path = link_to_mesh.get(child_link)
        if not parent_prim_path or not child_prim_path:
            skipped.append(joint_name)
            continue

        parent_prim = stage.GetPrimAtPath(parent_prim_path)
        child_prim = stage.GetPrimAtPath(child_prim_path)
        if not parent_prim or not parent_prim.IsValid() or not child_prim or not child_prim.IsValid():
            skipped.append(joint_name)
            continue
        if parent_prim_path == child_prim_path:
            skipped_same_body.append(joint_name)
            continue

        safe_name = _unique_name(_sanitize_prim_name(joint_name), used_names)
        joint_path = f"{args.joint_root}/{safe_name}"

        if joint_type in ("revolute", "continuous"):
            joint_prim = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        elif joint_type == "prismatic":
            joint_prim = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        elif joint_type == "fixed":
            joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)
        else:
            skipped.append(joint_name)
            continue

        # body relationships
        joint_prim.CreateBody0Rel().SetTargets([parent_prim_path])
        joint_prim.CreateBody1Rel().SetTargets([child_prim_path])
        joint_prim.CreateJointEnabledAttr().Set(True)

        # local frames
        # If using joint XYZ, place in world, then convert to each body's local frame.
        if joint_xyz is not None:
            if joint_name not in joint_xyz:
                skipped.append(joint_name)
                continue
            pos = joint_xyz[joint_name] * float(args.joint_xyz_scale)
            if args.joint_xyz_translate is not None:
                pos = pos + np.array(args.joint_xyz_translate, dtype=np.float64)
            if args.joint_xyz_space == "world":
                pos = pos + joint_xyz_offset
                lp0 = _world_to_local_pos(xform_cache, parent_prim, pos)
                lp1 = _world_to_local_pos(xform_cache, child_prim, pos)
                joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(lp0[0]), float(lp0[1]), float(lp0[2])))
                joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(float(lp1[0]), float(lp1[1]), float(lp1[2])))
            else:
                # Write directly into localPos (no world->local conversion).
                joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
                joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
            joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        # If using URDF origins, we assume mesh local frames align with URDF link frames.
        elif args.joint_pos == "urdf":
            origin = getattr(joint, "origin", None)
            if origin is not None:
                mat = np.array(origin, dtype=np.float64)
                t = mat[:3, 3]
                r = mat[:3, :3]
                w, x, y, z = _mat_to_quat(r)
                joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
                joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
            else:
                joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

            joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        else:
            # Place joint in world space based on mesh positions, then convert to each body's local frame.
            base, _ = _split_rpy_suffix(joint_name)
            if base in anchor_world_pos:
                w_pos = anchor_world_pos[base]
            else:
                w_pos = _compute_world_pos_from_meshes(
                    xform_cache, parent_prim, child_prim, args.joint_pos
                )

            lp0 = _world_to_local_pos(xform_cache, parent_prim, w_pos)
            lp1 = _world_to_local_pos(xform_cache, child_prim, w_pos)
            joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(float(lp0[0]), float(lp0[1]), float(lp0[2])))
            joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(float(lp1[0]), float(lp1[1]), float(lp1[2])))
            joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        # axis and limits (approximate axis token, store full axis as custom attr)
        axis = getattr(joint, "axis", None)
        if axis is not None and joint_type in ("revolute", "continuous", "prismatic"):
            axis = np.asarray(axis, dtype=np.float64)
            if axis.shape == (3,):
                axis_token = _axis_to_token(axis)
                joint_prim.CreateAxisAttr().Set(axis_token)
                joint_prim.GetPrim().CreateAttribute(
                    "urdf:axis", Sdf.ValueTypeNames.Float3
                ).Set(Gf.Vec3f(float(axis[0]), float(axis[1]), float(axis[2])))

        if args.override_revolute_limits_deg is not None and joint_type in ("revolute", "continuous"):
            lim = float(args.override_revolute_limits_deg)
            joint_prim.CreateLowerLimitAttr().Set(-lim)
            joint_prim.CreateUpperLimitAttr().Set(lim)
        elif joint_type != "continuous":
            limit = getattr(joint, "limit", None)
            if limit is not None:
                lower = getattr(limit, "lower", None)
                upper = getattr(limit, "upper", None)
                if lower is not None:
                    joint_prim.CreateLowerLimitAttr().Set(float(lower))
                if upper is not None:
                    joint_prim.CreateUpperLimitAttr().Set(float(upper))

        # store metadata for verification
        prim = joint_prim.GetPrim()
        prim.SetDisplayName(joint_name)
        prim.CreateAttribute("urdf:jointName", Sdf.ValueTypeNames.String).Set(joint_name)
        prim.CreateAttribute("urdf:parent", Sdf.ValueTypeNames.String).Set(parent_link)
        prim.CreateAttribute("urdf:child", Sdf.ValueTypeNames.String).Set(child_link)

        created.append(joint_name)

    out_path = args.out
    if os.path.abspath(out_path) == os.path.abspath(args.usd):
        stage.GetRootLayer().Save()
    else:
        stage.GetRootLayer().Export(out_path)

    if created_d6:
        print(f"Created D6 joints: {len(created_d6)}")
        for name in created_d6:
            print(f"  - {name}")
    print(f"Created joints: {len(created)}")
    if skipped:
        print(f"Skipped joints: {len(skipped)}")
        for name in skipped:
            print(f"  - {name}")
    if skipped_rpy:
        print(f"Collapsed R/P/Y joints: {len(skipped_rpy)}")
        for name in skipped_rpy:
            print(f"  - {name}")
    if skipped_d6:
        print(f"Skipped D6 groups: {len(skipped_d6)}")
        for name in skipped_d6:
            print(f"  - {name}")
    if skipped_same_body:
        print(f"Skipped joints (same body): {len(skipped_same_body)}")
        for name in skipped_same_body:
            print(f"  - {name}")
    print(f"Wrote USD to {out_path}")


if __name__ == "__main__":
    main()
