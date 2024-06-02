import os
import numpy as np
import cv2


class KITTI_odom_loader:
    def __init__(self, root: str, seqnum: int = 1, lr: str = "l", color: bool = True):
        self.root = root
        self.seqnum = f"{seqnum:02d}"
        self.calib_file = os.path.join(self.root, "calib", self.seqnum, "calib.txt")
        self.gt_poses_file = os.path.join(self.root, "poses", f"{self.seqnum}.txt")  # this is the relative transformation wrt to first timestamp camera coordinate system (gt_T_cur_CCS_to_ref_CCS)
        # 0 -> left gray, 1 -> right gray, 2 -> left color, 3 -> right color
        if lr == "l":
            self.img_dirnum = 2 if color else 0
        elif lr == "r":
            self.img_dirnum = 3 if color else 1
        else:
            raise ValueError("lr should be either 'l' or 'r'")
        self.img_dir = os.path.join(self.root, "sequences", self.seqnum, f"image_{self.img_dirnum}")
        self.load()

    def load(self):
        # Load the camera calibration matrix
        matrices = {}  # P0, P1, P2, P3, Tr 3x4 matrices
        with open(self.calib_file, 'r') as file:
            for line in file:
                key, value = line.split(':')
                if key in ['P0', 'P1', 'P2', 'P3', 'Tr']:
                    values = np.fromstring(value, sep=' ')
                    matrices[key] = values.reshape(3, 4)
        self.K = matrices['P0'][:3, :3]  # this is same for all P0, P1, P2, P3, the 4th column is the translation wrt to the left gray camera

        # Load the ground truth poses
        self.gt_poses = []
        with open(self.gt_poses_file, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                self.gt_poses.append(T)

        # Load the images
        image_paths = [os.path.join(self.img_dir, file) for file in sorted(os.listdir(self.img_dir))]
        self.images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        self.H = self.images[0].shape[0]
        self.W = self.images[0].shape[1]
