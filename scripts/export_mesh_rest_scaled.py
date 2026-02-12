#!/usr/bin/env python3
"""
Apply learned sim3 to a canonical mesh_rest OBJ to recover actual scale.
"""

import argparse
import os

import numpy as np
import torch
import trimesh
from pytorch3d import transforms


def vec_to_sim3(vec: torch.Tensor):
    center = vec[:3]
    orient = vec[3:7]
    orient = orient / orient.norm(p=2)
    orient = transforms.quaternion_to_matrix(orient)
    scale = vec[7:].exp()
    return center, orient, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to mesh_rest-*.obj")
    parser.add_argument("--ckpt", required=True, help="Path to params_latest.pth")
    parser.add_argument("--out", required=True, help="Output OBJ path")
    parser.add_argument(
        "--sim3_key",
        default="module.nerf_body_rts.sim3",
        help="Checkpoint key for sim3 (default: module.nerf_body_rts.sim3)",
    )
    parser.add_argument(
        "--skip_sim3",
        action="store_true",
        help="Skip applying sim3 from checkpoint (useful if mesh is already in target frame).",
    )
    parser.add_argument(
        "--ref_mesh",
        default=None,
        help="Optional reference mesh to match size/center (e.g., robot_rest-999.obj).",
    )
    parser.add_argument(
        "--ref_scale_mode",
        default="mean",
        choices=["mean", "max", "per_axis"],
        help="How to match size to reference mesh if --ref_mesh is set.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mesh):
        raise FileNotFoundError(args.mesh)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    mesh = trimesh.load(args.mesh, process=False)
    v = mesh.vertices.copy()
    if not args.skip_sim3:
        state = torch.load(args.ckpt, map_location="cpu")
        sim3 = None
        if isinstance(state, dict):
            # checkpoint might be a flat state_dict (OrderedDict)
            if args.sim3_key in state:
                sim3 = state[args.sim3_key]
            else:
                # try hierarchical lookup
                cur = state
                for part in args.sim3_key.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        cur = None
                        break
                sim3 = cur
            # fallback to common keys
            if sim3 is None:
                for k in ("module.nerf_body_rts.sim3", "module.sim3", "sim3"):
                    if k in state:
                        sim3 = state[k]
                        break
        if sim3 is None:
            raise KeyError(f"sim3 key not found: {args.sim3_key}")
        sim3 = sim3.detach().cpu().float()

        center, orient, scale = vec_to_sim3(sim3)
        center = center.numpy()
        orient = orient.numpy()
        scale = scale.numpy()

        v = v * scale[None]
        v = v @ orient.T
        v = v + center[None]

    # If reference mesh provided, rescale/recenter to match it.
    if args.ref_mesh:
        ref = trimesh.load(args.ref_mesh, process=False)
        ref_bounds = ref.bounds
        ref_center = ref_bounds.mean(axis=0)
        ref_size = ref_bounds[1] - ref_bounds[0]

        bounds = np.vstack([v.min(axis=0), v.max(axis=0)])
        cur_center = bounds.mean(axis=0)
        cur_size = bounds[1] - bounds[0]

        if args.ref_scale_mode == "per_axis":
            scale_vec = ref_size / np.maximum(cur_size, 1e-9)
            v = (v - cur_center[None]) * scale_vec[None] + ref_center[None]
        else:
            if args.ref_scale_mode == "max":
                s = float(ref_size.max() / max(cur_size.max(), 1e-9))
            else:
                s = float(ref_size.mean() / max(cur_size.mean(), 1e-9))
            v = (v - cur_center[None]) * s + ref_center[None]

    mesh.vertices = v
    mesh.export(args.out)
    print(f"Wrote scaled mesh to {args.out}")


if __name__ == "__main__":
    main()
