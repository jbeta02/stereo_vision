import cv2
import numpy as np
import glob

# ================= USER PARAMETERS =================
checkerboard = (9, 6)        # (columns, rows) INNER CORNERS
square_size = 1.          # meters per square (change to your checker size)

left_path = "images/left/*.png"     # path to LEFT images
right_path = "images/right/*.png"   # path to RIGHT images
# ===================================================


# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on real-world geometry
objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

left_images = sorted(glob.glob(left_path))
right_images = sorted(glob.glob(right_path))

assert len(left_images) == len(right_images), "Left/right image count mismatch!"

print(f"Found {len(left_images)} image pairs")

for left_file, right_file in zip(left_images, right_images):
    imgL = cv2.imread(left_file)
    imgR = cv2.imread(right_file)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)

    retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)

    if retL and retR:
        objpoints.append(objp)

        cornersL = cv2.cornerSubPix(grayL, cornersL, (3, 3), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (3, 3), (-1, -1), criteria)

        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

    else:
        print(f"Checkerboard not detected in pair: {left_file} / {right_file}")

# Ensure correct fisheye shapes
objpoints = [op.reshape(1, -1, 3) for op in objpoints]
imgpoints_left = [ip.reshape(1, -1, 2) for ip in imgpoints_left]
imgpoints_right = [ip.reshape(1, -1, 2) for ip in imgpoints_right]



N_OK = len(objpoints)
print(f"Valid usable pairs: {N_OK}")

if N_OK < 5:
    raise RuntimeError("Not enough valid pairs. Try capturing more images.")


# ======= FISHEYE CALIBRATION FOR EACH CAMERA =======
K1 = np.zeros((3, 3))
D1 = np.zeros((4, 1))
K2 = np.zeros((3, 3))
D2 = np.zeros((4, 1))

img_shape = grayL.shape[::-1]

flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
        cv2.fisheye.CALIB_FIX_SKEW + \
        cv2.fisheye.CALIB_CHECK_COND

rms1, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints_left, img_shape, K1, D1, None, None,
    flags, criteria
)

rms2, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints_right, img_shape, K2, D2, None, None,
    flags, criteria
)

print("Left RMS:", rms1)
print("Right RMS:", rms2)


# ======= STEREO CALIBRATION (EXTRINSICS) =======
R = np.zeros((3, 3))
T = np.zeros((3, 1))

stereo_flags = cv2.fisheye.CALIB_FIX_INTRINSIC

result = cv2.fisheye.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1, D1,
    K2, D2,
    img_shape,
    flags=cv2.fisheye.CALIB_FIX_INTRINSIC,
    criteria=criteria
)

rms_stereo = result[0]
K1 = result[1]
D1 = result[2]
K2 = result[3]
D2 = result[4]
R  = result[5]
T  = result[6]

print("Stereo RMS:", rms_stereo)
print("\nR:\n", R)
print("\nT:\n", T)


# ======= RECTIFICATION =======
# R1 = np.zeros((3, 3))
# R2 = np.zeros((3, 3))
# P1 = np.zeros((3, 4))
# P2 = np.zeros((3, 4))
# Q  = np.zeros((4, 4))

R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
    K1, D1, K2, D2, img_shape, R, T, 
    cv2.CALIB_ZERO_DISPARITY, # Example flag
    img_shape, 
    balance=0.0, 
    fov_scale=1.0
)

# ======= INIT RECTIFICATION MAPS =======
left_map1, left_map2 = cv2.fisheye.initUndistortRectifyMap(
    K1, D1, R1, P1[:, :3], img_shape, cv2.CV_16SC2
)

right_map1, right_map2 = cv2.fisheye.initUndistortRectifyMap(
    K2, D2, R2, P2[:, :3], img_shape, cv2.CV_16SC2
)

# np.savez("stereo_fisheye_calib.npz",
#          K1=K1, D1=D1, K2=K2, D2=D2,
#          R=R, T=T, R1=R1, R2=R2,
#          P1=P1, P2=P2, Q=Q,
#          left_map1=left_map1, left_map2=left_map2,
#          right_map1=right_map1, right_map2=right_map2)

# print("\nCalibration saved to stereo_fisheye_calib.npz")


# data = np.load("stereo_fisheye_calib.npz")
# lm1 = data["left_map1"]
# lm2 = data["left_map2"]
# rm1 = data["right_map1"]
# rm2 = data["right_map2"]

# # capL = cv2.VideoCapture(0)
# # capR = cv2.VideoCapture(1)

# while True:
#     _, frameL = capL.read()
#     _, frameR = capR.read()

#     rectL = cv2.remap(frameL, lm1, lm2, cv2.INTER_LINEAR)
#     rectR = cv2.remap(frameR, rm1, rm2, cv2.INTER_LINEAR)

#     both = np.hstack((rectL, rectR))
#     cv2.imshow("Rectified", both)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break


# Load the calibration results
# data = np.load("stereo_calibration.npz")

# mtx_l = data["mtx_l"]
# dist_l = data["dist_l"]
# mtx_r = data["mtx_r"]
# dist_r = data["dist_r"]
# R = data["R"]
# T = data["T"]
# R1 = data["R1"]
# R2 = data["R2"]
# P1 = data["P1"]
# P2 = data["P2"]
# Q = data["Q"]

# Load an example stereo pair
left = cv2.imread("rect/in/left/0_capture_device0.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("rect/in/right/0_capture_device2.png", cv2.IMREAD_GRAYSCALE)

image_size = left.shape[::-1]

# Build rectification maps
# left_map1, left_map2 = cv2.initUndistortRectifyMap(
#     K1, D1, R1, P1, image_size, cv2.CV_16SC2
# )

# right_map1, right_map2 = cv2.initUndistortRectifyMap(
#     K2, D2, R2, P2, image_size, cv2.CV_16SC2
# )

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

for y in range(0, left_rect.shape[0], 40):
    cv2.line(left_rect, (0,y), (left_rect.shape[1],y), 255, 1)
    cv2.line(right_rect,(0,y), (right_rect.shape[1],y), 255, 1)

cv2.imshow("Left", left_rect)
cv2.imshow("Right", right_rect)
cv2.imshow("Disparity", disp_vis)
cv2.waitKey(0)