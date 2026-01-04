import cv2
import glob
import numpy as np
from pathlib import Path

full_path = Path(__file__).resolve().parent.parent.parent # leads to \stereo_vision\ from opencv_stereo.py

def rectify_images():
    images_names = sorted(glob.glob("calibration_images_left/*"))
    im = cv2.imread(images_names[0], cv2.IMREAD_GRAYSCALE)
    imageSize = im.shape[::-1]

    full_path = Path(__file__).resolve().parent.parent.parent # leads to \stereo_vision\ from opencv_stereo.py

    # load the calibration results
    data = np.load("test_images/out/stereo_calibration_fisheye.npz")

    K1 = data["K1"]
    D1 = data["D1"]
    K2 = data["K2"]
    D2 = data["D2"]
    R = data["R"]
    T = data["T"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q = data["Q"]

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1[:, :3], imageSize, cv2.CV_16SC2
    )

    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2[:, :3], imageSize, cv2.CV_16SC2
    )

    # do remapping based on grid findings
    imgs_l = sorted(glob.glob("test_images/in/left/*"))
    imgs_r = sorted(glob.glob("test_images/in/right/*"))
    i = 0
    for img_l, img_r in zip(imgs_l, imgs_r):
        imL = cv2.imread(img_l, cv2.IMREAD_GRAYSCALE)
        imR = cv2.imread(img_r, cv2.IMREAD_GRAYSCALE)

        left_rect  = cv2.remap(imL,  map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(imR, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imwrite(f"{full_path}/test_images/out/left/left_rectified_img_fisheye_{i}.png", left_rect)
        cv2.imwrite(f"{full_path}/test_images/out/right/right_rectified_img_fisheye_{i}.png", right_rect)
        i+=1

    print("Baseline (m):", np.linalg.norm(T))


def build_disparity_map(
        minDisparity,
        numDisparities,
        blockSize,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=120,
        speckleRange=2
    ):


    # create depth map
    img_path0 = f"{full_path}/test_images/out/left/left_rectified_img_fisheye_0.png" # left
    img_path1 = f"{full_path}/test_images/out/right/right_rectified_img_fisheye_0.png" # right

    img0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

    # stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=15)
    channels = 1 # since using grayscale use 1
    stereo = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * channels * blockSize**2, # recommended openCV formula
        P2=32 * channels * blockSize**2, # recommended openCV formula
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange
    )
    disparity = stereo.compute(img0, img1).astype(np.float32) / 16.0
    map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return map


if __name__ == "__main__":

    rectify_images()

    map = build_disparity_map(0, 64, 5, 1, 10, 120, 2) # decent default values (0, 64, 5, 1, 10, 120, 2)

    cv2.imshow("numDisparities_BM 128, block 15", map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()