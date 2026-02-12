#!/usr/bin/env python3
"""Assign MassAPI density to all link meshes in mapping.yaml."""
from __future__ import annotations

import argparse
from typing import Dict, Set

import yaml
from pxr import Usd, UsdPhysics


def _load_link_meshes(mapping_path: str) -> Set[str]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    link_to_mesh = data.get("link_to_mesh", {}) or {}
    return {str(p).strip() for p in link_to_mesh.values() if p}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, help="USD file to modify")
    parser.add_argument("--mapping", required=True, help="mapping.yaml path")
    parser.add_argument("--density", type=float, default=1000.0, help="Density value")
    parser.add_argument(
        "--out",
        default=None,
        help="Output USD path. If omitted, modifies the input USD in place.",
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd)
    if not stage:
        raise SystemExit(f"Failed to open USD: {args.usd}")

    mesh_paths = _load_link_meshes(args.mapping)
    updated = 0
    missing = 0
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            missing += 1
            continue
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.CreateDensityAttr().Set(float(args.density))
        updated += 1

    out_path = args.out or args.usd
    if out_path == args.usd:
        stage.GetRootLayer().Save()
    else:
        stage.GetRootLayer().Export(out_path)

    print(f"Applied density={args.density} to {updated} prims (missing {missing}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
