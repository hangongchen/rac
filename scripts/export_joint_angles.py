# Export per-frame URDF joint angles for a pretrained model.
# Outputs joint_names.txt and angles.(npy|csv) in the model logdir.

from absl import app, flags
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.insert(0, 'third_party')

from utils.io import str_to_frame
from nnutils.train_utils import v2s_trainer

opts = flags.FLAGS
flags.DEFINE_bool("clamp_to_urdf", True, "Clamp angles to URDF joint limits when available.")
flags.DEFINE_bool("rest_only", False, "Export only the rest pose angles (x=None).")


def _ensure_pre_skel(trainer):
    if not hasattr(trainer.model, "robot") or not hasattr(trainer.model.robot, "urdf"):
        raise ValueError("pre_skel is not set; cannot export URDF joint angles.")


def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()
    trainer.define_model(data_info)
    _ensure_pre_skel(trainer)

    device = trainer.device
    urdf = trainer.model.robot.urdf
    joint_names = urdf.angle_names
    if opts.rest_only:
        with torch.no_grad():
            _, angles = trainer.model.nerf_body_rts.forward_abs(x=None)
        angles_all = angles.detach().cpu().numpy()
        frame_ids = []
    else:
        idx_render = str_to_frame(opts.test_frames, data_info)
        if len(idx_render) == 0:
            raise ValueError("No frames to export. Check --test_frames and dataset.")
        angles_all = []
        frame_ids = []
        chunk = max(1, opts.frame_chunk)
        for i in range(0, len(idx_render), chunk):
            idx_chunk = idx_render[i:i + chunk]
            frame_ids.extend(idx_chunk)
            query_time = torch.tensor(idx_chunk, device=device).long()
            with torch.no_grad():
                _, angles = trainer.model.nerf_body_rts.forward_abs(x=query_time)
            angles_all.append(angles.detach().cpu().numpy())
        angles_all = np.concatenate(angles_all, axis=0)

    if opts.clamp_to_urdf:
        for j, name in enumerate(joint_names):
            joint = urdf.joint_map[name]
            if joint.limit is None:
                continue
            lower = joint.limit.lower
            upper = joint.limit.upper
            if lower is None or upper is None:
                continue
            angles_all[:, j] = np.clip(angles_all[:, j], lower, upper)

    save_dir = opts.model_path.rsplit('/', 1)[0]
    os.makedirs(save_dir, exist_ok=True)
    if opts.rest_only:
        np.save(os.path.join(save_dir, "rest_angles.npy"), angles_all)
    else:
        np.save(os.path.join(save_dir, "angles.npy"), angles_all)
        np.save(os.path.join(save_dir, "frame_ids.npy"), np.array(frame_ids, dtype=np.int64))
    with open(os.path.join(save_dir, "joint_names.txt"), "w") as f:
        for name in joint_names:
            f.write(f"{name}\n")

    # CSV: header is joint names, rows aligned with frame_ids.npy
    csv_path = os.path.join(save_dir, "rest_angles.csv" if opts.rest_only else "angles.csv")
    header = ",".join(joint_names)
    np.savetxt(csv_path, angles_all, delimiter=",", header=header, comments="")

    if opts.rest_only:
        print("Saved rest pose angles to:")
        print(f"  {os.path.join(save_dir, 'rest_angles.npy')}")
        print(f"  {csv_path}")
        print(f"  {os.path.join(save_dir, 'joint_names.txt')}")
    else:
        print(f"Saved {angles_all.shape[0]} frames to:")
        print(f"  {os.path.join(save_dir, 'angles.npy')}")
        print(f"  {csv_path}")
        print(f"  {os.path.join(save_dir, 'joint_names.txt')}")
        print(f"  {os.path.join(save_dir, 'frame_ids.npy')}")


if __name__ == "__main__":
    app.run(main)
