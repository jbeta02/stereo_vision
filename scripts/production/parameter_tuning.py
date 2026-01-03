import opencv_pinhole_stereo
import cv2
import glob
import numpy as np
from pathlib import Path

opencv_pinhole_stereo.rectify_images()

# tunning strategy
"""
1) Determine a good default state as the base for incremental changes
2) One parameter at a time increment values, select the best choice, and use
    best choice as new current/default. Repeat this step for each parameter
"""

# default
current_minDisparity=0
current_numDisparities=64
current_blockSize=15
current_disp12MaxDiff=1
current_uniquenessRatio=7
current_speckleWindowSize=100
current_speckleRange=2

current_map = opencv_pinhole_stereo.build_disparity_map(
    minDisparity=current_minDisparity,
    numDisparities=current_numDisparities,
    blockSize=current_blockSize,
    disp12MaxDiff=current_disp12MaxDiff,
    uniquenessRatio=current_uniquenessRatio,
    speckleWindowSize=current_speckleWindowSize,
    speckleRange=current_speckleRange
)

title = (
    f"numDisp {current_numDisparities}, " + \
    f"blockSize {current_blockSize}, " + \
    f"MaxDiff {current_disp12MaxDiff}, " + \
    f"uniqRatio {current_uniquenessRatio}, " + \
    f"specWindSize {current_speckleWindowSize}, " + \
    f"specRange {current_speckleRange}"
)


def tune_numDisparities():
    # how far the matcher searches for a match
    # 16 - 512, val must be div by 16
    i = 16
    while i <= 512:
        current_numDisparities = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)
        i += i
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_numDisparities = int(input("best numDisp (int): "))



def tune_blockSize():
    # window size for block matching
    # 1 - 15
    for i in range(1, 16, 2):
        current_blockSize = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_blockSize = int(input("best blockSize (int): "))



def tune_MaxDiff():
    # sets the maximum allowed difference between left and right image window
    # 0 - 10
    for i in range(1, 11):
        current_disp12MaxDiff = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_disp12MaxDiff = int(input("best MaxDiff (int): "))


def tune_uniquenessRatio():
    # rejects ambiguous matches
    # 1 - 15
    for i in range(1, 16, 2):
        current_uniquenessRatio = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_uniquenessRatio = int(input("best uniqRatio (int): "))


def tune_speckleWindowSize():
    # minimum connected region size to keep
    # 50 - 200
    for i in range(50, 200, 25):
        current_speckleWindowSize = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_speckleWindowSize = int(input("best specWindSize (int): "))


def tune_speckleRange():
    # minimum connected region size to keep
    # 1 - 5
    for i in range(1, 5):
        current_speckleRange = i

        current_map = opencv_pinhole_stereo.build_disparity_map(
            minDisparity=current_minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize,
            disp12MaxDiff=current_disp12MaxDiff,
            uniquenessRatio=current_uniquenessRatio,
            speckleWindowSize=current_speckleWindowSize,
            speckleRange=current_speckleRange
        )

        title = (
        f"numDisp {current_numDisparities}, " + \
        f"blockSize {current_blockSize}, " + \
        f"MaxDiff {current_disp12MaxDiff}, " + \
        f"uniqRatio {current_uniquenessRatio}, " + \
        f"specWindSize {current_speckleWindowSize}, " + \
        f"specRange {current_speckleRange}"
        )
        cv2.imshow(title, current_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    current_speckleRange = int(input("best specRange (int): "))


if __name__ == "__main__":
    
    # 1
    cv2.imshow("default " + title, current_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2
    tune_numDisparities()
    tune_blockSize()
    tune_MaxDiff()
    tune_uniquenessRatio()
    tune_speckleWindowSize()
    tune_speckleRange()

    cv2.imshow("result " + title, current_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()