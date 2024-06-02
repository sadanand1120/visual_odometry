import numpy as np
import cv2
from kitti_run import KITTI_odom_loader
from tqdm.auto import tqdm
from lib.visualization import plotting
from lib.visualization.video import play_trip
from matplotlib import pyplot as plt


class ORB_VO:
    """
    Runs ORB feature matching based VO on cv2 grayscale images and estimates the camera pose
    """

    def __init__(self, K: np.ndarray, W: int, H: int):
        self.K = K
        self.W = W
        self.H = H
        # detector and descriptor
        self.orb = cv2.ORB_create(nfeatures=3000)
        # matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _form_transf(R: np.ndarray, t: np.ndarray):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, prev_frame: np.ndarray, cur_frame: np.ndarray):
        """
        This function detect and compute keypoints and descriptors from the prev_frame and cur_frame using the class orb object

        Returns
        -------
        q_prev (ndarray): The good keypoints matches pixel coords in prev_frame
        q_cur (ndarray): The good keypoints matches pixel coords in cur_frame
        img3 (ndarray): The BGR grayscale image with the good matches drawn
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(cur_frame, None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)  # k=2 to get the two best matches

        # Find the good matches using Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(cur_frame, kp1, prev_frame, kp2, good, None, **draw_params)
        img3 = cv2.resize(img3, (self.W, self.H))

        # Get the image points from the good matches
        q_prev = np.float32([kp1[m.queryIdx].pt for m in good])  # N x 2
        q_cur = np.float32([kp2[m.trainIdx].pt for m in good])  # N x 2
        return q_prev, q_cur, img3

    def get_relative_T(self, q1: np.ndarray, q2: np.ndarray):
        """
        Calculates the transformation matrix (from q1 cam frame to q2 cam frame) using the given pixel coordinates of the good matches

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix


if __name__ == "__main__":
    kitti = KITTI_odom_loader(root="/home/dynamo/Music/visualOdometry/KITTI_odom", seqnum=1, lr="l", color=True)
    orb_feat = ORB_VO(K=kitti.K, W=kitti.W, H=kitti.H)

    # play_trip(kitti.images)

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(kitti.gt_poses[:100], unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q_prev, q_cur, img3 = orb_feat.get_matches(kitti.images[i - 1], kitti.images[i])
            # cv2.imshow("Matches", img3)
            # cv2.waitKey(200)
            transf = orb_feat.get_relative_T(q_cur, q_prev)  # T_cur_CCS_to_prev_CCS
            cur_pose = np.matmul(cur_pose, transf)   # T_cur_CCS_to_ref_CCS = T_prev_CCS_to_ref_CCS * T_cur_CCS_to_prev_CCS
        # X-Z plane (CCS) is the ground plane
        # So basically consider a point at the origin of the camera coordinate system, and we are tracing its path, in the first timestamp CCS frame
        pt_cur_CCS = np.array([0, 0, 0, 1])
        est_pt_ref_CCS = np.matmul(cur_pose, pt_cur_CCS)
        gt_pt_ref_CCS = np.matmul(gt_pose, pt_cur_CCS)
        # No need to unhomogenize as 4th coordinate is 1 already
        estimated_path.append((est_pt_ref_CCS[0], est_pt_ref_CCS[2]))
        gt_path.append((gt_pt_ref_CCS[0], gt_pt_ref_CCS[2]))
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out="orb.html")
