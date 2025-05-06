#!/usr/bin/env python3
import argparse, yaml, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

# ------------------------------ helpers
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def cam_yaml_to_P(yml):
    fx, fy = yml["camera_matrix"]["data"][0], yml["camera_matrix"]["data"][4]
    cx, cy = yml["camera_matrix"]["data"][2], yml["camera_matrix"]["data"][5]
    return np.array([[fx, 0, cx, 0],
                     [0, fy, cy, 0],
                     [0,  0,  1, 0]], dtype=np.float64)

def quat_xyz_to_T(x, y, z, qx, qy, qz, qw):
    w, x_, y_, z_ = qw, qx, qy, qz
    R = np.array([
        [1-2*(y_**2+z_**2), 2*(x_*y_-z_*w),   2*(x_*z_+y_*w)],
        [2*(x_*y_+z_*w),     1-2*(x_**2+z_**2), 2*(y_*z_-x_*w)],
        [2*(x_*z_-y_*w),     2*(y_*z_+x_*w),   1-2*(x_**2+y_**2)]
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = [x, y, z]
    return T

def load_dense_global_pose_file(path):
    Ts = []
    with open(path, "r") as f:
        for line in f:
            toks = [float(t) for t in line.strip().split()]
            if len(toks) == 8:
                _, x, y, z, qx, qy, qz, qw = toks
            elif len(toks) == 7:
                x, y, z, qx, qy, qz, qw = toks
            else:
                continue
            Ts.append(quat_xyz_to_T(x, y, z, qx, qy, qz, qw))
    return Ts

def write_calib(out_path, P, Tr):
    def mat12(m): return " ".join(f"{v:.12e}" for v in m.reshape(-1)[:12])
    with open(out_path, "w") as f:
        for tag in ("P0", "P1", "P2", "P3"):
            f.write(f"{tag}: {mat12(P)}\n")
        f.write(f"Tr: {mat12(Tr[:3])}\n")

def write_poses(out_path, Ts, T0_inv):
    with open(out_path, "w") as f:
        for T in Ts:
            T_rel = T0_inv @ T
            line = " ".join(f"{v:.6e}" for v in T_rel[:3].reshape(-1))
            f.write(line + "\n")

# ------------------------------ main conversion
def convert_seq_range(seq_id, start, end, args):
    seq_str = str(seq_id)
    frame_range = range(start, end)

    src_dir = Path(args.coda_root) / "2d_rect" / "cam0" / seq_str
    src_imgs = sorted(src_dir.glob("*"), key=lambda x: int(str(x).split(".")[0].split("_")[-1].lstrip("0") or "0"))
    import ipdb; ipdb.set_trace()
    if not src_imgs or end > len(src_imgs):
        raise RuntimeError(f"Frame range invalid or images missing for seq {seq_str}")

    # Extract just desired frames
    selected_imgs = [src_imgs[i] for i in frame_range]

    out_seq = Path(args.out_root) / f"sequences/{seq_str}_{start}_{end}"
    img_dir = out_seq / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- calibration ---
    cam_yaml = Path(args.coda_root)/"calibrations"/seq_str/"calib_cam0_intrinsics.yaml"
    P0 = cam_yaml_to_P(load_yaml(cam_yaml))

    extr_path = Path(args.coda_root)/"calibrations"/seq_str/"calib_os1_to_cam0.yaml"
    extr_yaml = load_yaml(extr_path)
    Tr_full = np.array(extr_yaml["extrinsic_matrix"]["data"], dtype=np.float64).reshape(
        extr_yaml["extrinsic_matrix"]["rows"],
        extr_yaml["extrinsic_matrix"]["cols"]
    )
    write_calib(out_seq/"calib.txt", P0, Tr_full)

    # --- poses ---
    pose_file = Path(args.coda_root)/"poses"/"dense_global"/f"{seq_str}.txt"
    Ts_all = load_dense_global_pose_file(pose_file)
    Ts = Ts_all[start:end]
    if len(Ts) != len(selected_imgs):
        raise RuntimeError(f"Pose/image mismatch: {len(Ts)} vs {len(selected_imgs)}")
    T0_inv = np.linalg.inv(Ts[0])
    write_poses(out_seq/"poses.txt", Ts, T0_inv)

    # --- images ---
    for idx, img_path in enumerate(tqdm(selected_imgs, desc=f"Seq {seq_str} [{start}:{end})", leave=False)):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(img_dir/f"{idx:06d}.png"), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

# ------------------------------ CLI wrapper
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coda_root", default="/robodata/arthurz/Datasets/CODa_v2")
    ap.add_argument("--out_root",  default="/home/dynamo/Music/visodom/visual_odometry/CODA_kitti_odom")
    args = ap.parse_args()

    ranges = [
        (0, 55, 103),
        (0, 670, 720),
        (0, 720, 770),
        (0, 770, 820),
        (0, 820, 870),
        (0, 880, 930),
        (0, 930, 980),
        (0, 980, 1020),
        (0, 2095, 2145),
        (0, 2145, 2195),
        (0, 2195, 2245),
        (0, 2245, 2295),
        (0, 2295, 2345),
        (0, 2345, 2350),
        (14, 11000, 11050),
        (14, 11050, 11100),
        (0, 1580, 1630),
        (0, 1630, 1640),
        (0, 4320, 4370),
        (0, 4370, 4420),
        (0, 4420, 4440),
        (0, 5050, 5100),
        (0, 5100, 5150),
        (0, 5150, 5200),
        (0, 5200, 5250),
        (2, 2070, 2120),
        (2, 2120, 2170),
        (2, 2170, 2220),
        (2, 2220, 2270),
        (2, 2270, 2300),
        (3, 2580, 2630),
        (3, 2630, 2680),
        (3, 2680, 2730),
        (3, 2730, 2780),
        (3, 2780, 2820),
        (6, 1611, 1661),
        (6, 1661, 1711),
        (6, 1711, 1761),
        (6, 1761, 1811),
        (6, 1811, 1861),
        (6, 1861, 1911),
        (6, 1911, 1940),
        (7, 6630, 6680),
        (7, 6680, 6730),
        (7, 6730, 6780),
        (7, 6780, 6830),
        (7, 6830, 6880),
        (7, 6880, 6900),
        (8, 0, 50),
        (8, 50, 100),
        (8, 200, 250),
        (8, 250, 300),
        (8, 300, 350),
        (8, 350, 400),
        (8, 400, 450),
        (8, 450, 500),
        (8, 500, 550),
        (8, 550, 600),
        (13, 9344, 9394),
        (13, 9394, 9444),
        (13, 9444, 9494),
        (13, 9494, 9544),
        (13, 9544, 9594),
        (13, 9594, 9644),
        (13, 9644, 9694),
        (13, 9694, 9744),
        (13, 9744, 9794),
        (13, 9794, 9800),
        (13, 10550, 10600),
        (13, 10600, 10650),
        (13, 10650, 10700),
        (13, 10700, 10750),
        (13, 10750, 10800),
        (13, 10800, 10850),
        (13, 10850, 10900),
        (13, 10900, 10950),
        (13, 10950, 10970),
        (14, 2030, 2080),
        (14, 2080, 2130),
        (14, 2130, 2150),
        (14, 2270, 2320),
        (14, 2320, 2370),
        # (0, 0, 50),
        # (0, 50, 90),
        # (0, 90, 130),
        # (1, 120, 170),
        # (1, 170, 220),
        # (1, 220, 270),
        # (1, 270, 320),
        # (1, 320, 370),
        # (1, 370, 420),
        # (1, 420, 470),
        # (1, 470, 510),
    ]

    for i in tqdm(range(0, len(ranges)), desc="Converting sequences"):
        seq_id, start, end = ranges[i]
        convert_seq_range(seq_id, start, end, args)

if __name__ == "__main__":
    main()
