import cv2
import subprocess
import os
import numpy as np
from time import sleep

# capture and save test image
def capture_test_image(device): # device is device number ex: 0
    # open a video device number x
    cap = cv2.VideoCapture(device) # NOTE: find correct video using v4l2-ctl --list-devices (get first videoX of usb cam list)

    # enable auto focus
    subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=focus_automatic_continuous=1"]) # 1 = Enable, 0 = disable

    # disable auto exposure
    subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=auto_exposure=1"]) # 1 = Manual Mode, 3 = Aperture Priority Mode

    # disable auto white balance
    subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=white_balance_automatic=0"]) # 1 = on, 0 = off

    # Capture frame-by-frame
    ret, frame = cap.read()

    return frame


def draw_chess_single_cam(frame):

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 6 #number of checkerboard rows.
    columns = 9 #number of checkerboard columns.
    world_scaling = 0.0186    # meters
    shape = (rows, columns)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.

    gray = frame
 
    #find the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, shape, None)

    if ret == True:

        #Convolution size used to improve corner detection. Don't make this too large.
        conv_size = (11, 11)

        #opencv can attempt to improve the checkerboard coordinates
        corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, shape, corners, ret)

    return frame

if __name__ == "__main__":

    deviceL = 0
    deviceR = 2

    pathL = "/home/dell-user/Software/robo_sub/D0/"
    pathR = "/home/dell-user/Software/robo_sub/D2/"

    image_count = 20

    for i in range(image_count):
        imgL = None
        imgR = None

        # position good image
        while True:
            imgL = capture_test_image(0)
            imgR = capture_test_image(2)

            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            cv2.imshow('imgL', imgL)
            cv2.imshow('imgR', imgR)
            
            k = cv2.waitKey(1) & 0xFF

            if k == ord('c'):
                break # Exit the loop

        cv2.destroyAllWindows()


        # save image if good
        while True:

            viewL = imgL.copy()
            viewR = imgR.copy()

            cv2.imshow('imgL', draw_chess_single_cam(viewL))
            cv2.imshow('imgR', draw_chess_single_cam(viewR))

            # Wait for a key press for 1 millisecond
            # The result is masked with 0xFF to get the lower 8 bits, 
            # which is necessary for cross-platform compatibility
            k = cv2.waitKey(1) & 0xFF

            # Check if the pressed key's ASCII value is 'w' (which is 119)
            if k == ord('y'):
                img_nameL = f"{pathL}{str(i)+"_"}capture_device{deviceL}.png"
                cv2.imwrite(img_nameL, imgL)
                print(f"Image saved as {os.path.abspath(img_nameL)}")

                img_nameR = f"{pathR}{str(i)+"_"}capture_device{deviceR}.png"
                cv2.imwrite(img_nameR, imgR)
                print(f"Image saved as {os.path.abspath(img_nameR)}")

                break

            if k == ord('n'):
                break # Exit the loop

        # Close all OpenCV windows
        cv2.destroyAllWindows()