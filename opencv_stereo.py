import cv2
import numpy as np
import glob
import os

# ---------------------------
# Checkerboard configuration
# ---------------------------
CHECKERBOARD = (9, 6)   # inner corners (columns, rows)
SQUARE_SIZE = 0.0186    # meters (adjust to your checkerboard)

# Termination criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ---------------------------
# Prepare object points
# ---------------------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []     # 3D points in real world
imgpoints_l = []   # 2D points in left image
imgpoints_r = []   # 2D points in right image

# ---------------------------
# Load image pairs
# ---------------------------
left_images = sorted(glob.glob("images/left/*.png"))
right_images = sorted(glob.glob("images/right/*.png"))

assert len(left_images) == len(right_images), "Mismatched image pairs"

for left_img, right_img in zip(left_images, right_images):
    img_l = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

    ret_l, corners_l = cv2.findChessboardCorners(img_l, CHECKERBOARD)
    ret_r, corners_r = cv2.findChessboardCorners(img_r, CHECKERBOARD)

    if ret_l and ret_r:
        objpoints.append(objp)

        corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)

        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

print(f"Valid pairs used: {len(objpoints)}")

# ---------------------------
# Single camera calibration
# ---------------------------
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, img_l.shape[::-1], None, None
)

ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, img_r.shape[::-1], None, None
)

# ---------------------------
# Stereo calibration
# ---------------------------
flags = cv2.CALIB_USE_INTRINSIC_GUESS

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    img_l.shape[::-1],
    criteria=criteria,
    flags=flags
)

# ---------------------------
# Stereo rectification
# ---------------------------
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l,
    mtx_r, dist_r,
    img_l.shape[::-1],
    R, T,
    alpha=0
)

# ---------------------------
# Save calibration data
# ---------------------------
np.savez(
    "stereo_calibration.npz",
    mtx_l=mtx_l,
    dist_l=dist_l,
    mtx_r=mtx_r,
    dist_r=dist_r,
    R=R,
    T=T,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q
)

print("Stereo calibration saved to stereo_calibration.npz")

from time import sleep
sleep(1)

# Load the calibration results
data = np.load("stereo_calibration.npz")

mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]
R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]
Q = data["Q"]

# Load an example stereo pair
left = cv2.imread("rect/in/left/0_capture_device0.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("rect/in/right/0_capture_device2.png", cv2.IMREAD_GRAYSCALE)

image_size = left.shape[::-1]

# Build rectification maps
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
)

# Apply rectification
left_rect = cv2.remap(left, left_map1, left_map2, cv2.INTER_LINEAR)
right_rect = cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR)


# StereoSGBM parameters
window_size = 5

min_disp = 0
num_disp = 128  # must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=16
)


disparity = stereo.compute(left_rect, right_rect).astype(np.float32)

# Convert from fixed-point to float disparity (div by 16)
disparity /= 16.0


disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)

cv2.imshow("Left", left_rect)
cv2.imshow("Right", right_rect)
cv2.imshow("Disparity", disp_vis)
cv2.waitKey(0)