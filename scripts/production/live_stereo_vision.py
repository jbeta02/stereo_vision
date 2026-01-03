import cv2
import glob
import numpy as np
from pathlib import Path
import image_capture

while True:

    frame0, frame1, cap0, cap1 = image_capture.capture(0, 2)
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    data = np.load("test_images/out/stereo_calibration_pinhole.npz")

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

    imageSize = frame0.shape[::-1]

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, imageSize, cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, imageSize, cv2.CV_32FC1
    )

    frame0 = cv2.remap(frame0,  map1x, map1y, cv2.INTER_LINEAR)
    frame1 = cv2.remap(frame1,  map1x, map1y, cv2.INTER_LINEAR)


    channels = 1 # since using grayscale use 1
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=15,
        P1=8 * channels * 15**2, # recommended openCV formula
        P2=32 * channels * 15**2, # recommended openCV formula
        disp12MaxDiff=1,
        uniquenessRatio=7,
        speckleWindowSize=100,
        speckleRange=2
    )

    disparity = stereo.compute(frame0, frame1).astype(np.float32) / 16.0
    map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("map", map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()