#!/usr/bin/env python
"""Add PhysX attachments between a soft body mesh and link meshes from mapping.yaml."""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Set

import yaml
from pxr import Sdf, Usd


def _load_link_meshes(mapping_path: str) -> List[str]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    link_to_mesh = data.get("link_to_mesh", {}) or {}
    meshes = []
    for _, mesh_path in link_to_mesh.items():
        if not mesh_path:
            continue
        meshes.append(str(mesh_path).strip())
    # Preserve stable order while deduping
    seen: Set[str] = set()
    unique = []
    for m in meshes:
        if m in seen:
            continue
        seen.add(m)
        unique.append(m)
    return unique


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe)
    return out or "attachment"


def _get_actor1_target(prim: Usd.Prim) -> Optional[str]:
    rel = prim.GetRelationship("actor1")
    if not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return targets[0].pathString


def _ensure_rel(prim: Usd.Prim, rel_name: str, targets: List[str]) -> None:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        rel = prim.CreateRelationship(rel_name)
    rel.SetTargets([Sdf.Path(t) for t in targets])


def _ensure_attr_bool(prim: Usd.Prim, attr_name: str, value: bool) -> None:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
    attr.Set(bool(value))


def _ensure_mask_shapes_rel(prim: Usd.Prim) -> None:
    rel = prim.GetRelationship("physxAutoAttachment:maskShapes")
    if not rel:
        prim.CreateRelationship("physxAutoAttachment:maskShapes")


def add_attachments(
    usd_path: str,
    mapping_path: str,
    soft_body_path: str,
    out_path: Optional[str],
) -> int:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise SystemExit(f"Failed to open USD: {usd_path}")

    soft_body = stage.GetPrimAtPath(soft_body_path)
    if not soft_body:
        raise SystemExit(f"Soft body prim not found: {soft_body_path}")

    link_meshes = _load_link_meshes(mapping_path)

    existing_by_actor1: Dict[str, Usd.Prim] = {}
    used_names: Set[str] = set()

    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysxPhysicsAttachment":
            continue
        # Track any attachment under the soft body prim
        if prim.GetParent() != soft_body:
            continue
        used_names.add(prim.GetName())
        actor1 = _get_actor1_target(prim)
        if actor1:
            existing_by_actor1[actor1] = prim

    created = 0
    updated = 0
    skipped_missing = 0

    for mesh_path in link_meshes:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim:
            skipped_missing += 1
            continue

        if mesh_path in existing_by_actor1:
            prim = existing_by_actor1[mesh_path]
            _ensure_rel(prim, "actor0", [soft_body_path])
            _ensure_rel(prim, "actor1", [mesh_path])
            _ensure_attr_bool(prim, "physxAutoAttachment:enableRigidSurfaceAttachments", True)
            _ensure_mask_shapes_rel(prim)
            updated += 1
            continue

        base_name = _sanitize_name(f"attachment_{mesh_prim.GetName()}")
        name = base_name
        idx = 1
        while name in used_names:
            name = f"{base_name}_{idx}"
            idx += 1
        used_names.add(name)

        prim = stage.DefinePrim(f"{soft_body_path}/{name}", "PhysxPhysicsAttachment")
        _ensure_rel(prim, "actor0", [soft_body_path])
        _ensure_rel(prim, "actor1", [mesh_path])
        _ensure_attr_bool(prim, "physxAutoAttachment:enableRigidSurfaceAttachments", True)
        _ensure_mask_shapes_rel(prim)
        created += 1

    root_layer = stage.GetRootLayer()
    if out_path and out_path != usd_path:
        root_layer.Export(out_path)
    else:
        root_layer.Save()

    print(
        f"Attachments: created={created}, updated={updated}, "
        f"skipped_missing={skipped_missing}, total_meshes={len(link_meshes)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, help="Input USD file")
    parser.add_argument("--mapping", required=True, help="mapping.yaml file")
    parser.add_argument(
        "--soft_body",
        default="/root/mesh_rest_999/mesh_rest_999_002",
        help="Prim path of the soft body mesh",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output USD path. If omitted, modifies the input USD in place.",
    )
    args = parser.parse_args()

    return add_attachments(
        usd_path=args.usd,
        mapping_path=args.mapping,
        soft_body_path=args.soft_body,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
