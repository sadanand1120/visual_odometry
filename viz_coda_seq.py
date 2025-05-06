import cv2
import time
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, cv2
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

def save_coda_sequence_video(seq_dir, out_path, fps=2, font_scale=1.0):
    images = os.listdir(seq_dir)
    images = sorted(images, key=lambda x: int(x.split(".")[0].split("_")[-1].lstrip("0") or "0"))
    image_paths = [os.path.join(seq_dir, img) for img in images]

    if not image_paths:
        print("No images found in", seq_dir)
        return

    # Read first frame to get dimensions
    frame0 = cv2.imread(image_paths[0])
    h, w = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for idx, img_path in enumerate(tqdm(image_paths, desc="Creating video")):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable frame {img_path}")
            continue

        # Overlay frame number (top-left corner)
        text = f'Frame {int(os.path.basename(img_path).split(".")[0].split("_")[-1].lstrip("0") or "0")}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Video saved to {out_path}")


def _make_args(seq_num: int):
    seq_dir = f"/robodata/smodak/jan25_2025_bkp_personal_laptop/visualOdometry/KITTI_odom/sequences/{seq_num:02}/image_2"
    out_path = Path(f"seq_{seq_num}.mp4")
    return seq_dir, out_path

if __name__ == "__main__":
    fps = 10
    seq_range = range(23)                      # 0 … 22
    n_workers = min(len(seq_range), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(save_coda_sequence_video, * _make_args(seq), fps): seq
            for seq in seq_range
        }
        for f in as_completed(futures):
            seq = futures[f]
            try:
                f.result()                     # re‑raise if the worker erred
            except Exception as e:
                print(f"[!] seq {seq} failed: {e}")
