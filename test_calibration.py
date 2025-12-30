import cv2
import subprocess
import os
import numpy as np


# capture and save test image
def capture_test_image(device, label=None, path=None): # device is device number ex: 0
    # open a video device number x
    cap = cv2.VideoCapture(device) # NOTE: find correct video using v4l2-ctl --list-devices (get first videoX of usb cam list)

    # enable auto focus
    subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=focus_automatic_continuous=1"]) # 1 = Enable, 0 = disable

    # disable auto exposure
    subprocess.run(["v4l2-ctl", "-d", "/dev/video" + str(device), "--set-ctrl=auto_exposure=1"]) # 1 = Manual Mode, 3 = Aperture Priority Mode

    # Capture frame-by-frame
    ret, frame = cap.read()
    # save the resulting frame
    img_name = f"{path}{label}capture_device{device}.png"
    cv2.imwrite(img_name, frame)
    print(f"Image saved as {os.path.abspath(img_name)}")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# build A matrix which refers to system of equations containing camera and world coordinates
# use this video for reference https://www.youtube.com/watch?v=GUbWsXU1mac&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=4
def build_A_matrix(camera_points, world_points):
    A = []

    if len(camera_points) == len(world_points):
        for c, w in zip(camera_points, world_points):
            first_row = [w[0], w[1], w[2], 1, 0, 0, 0, 0, -c[0] * w[0], -c[0] * w[1], -c[0] * w[2], -c[0]]
            second_row = [0, 0, 0, 0, w[0], w[1], w[2], 1, -c[1] * w[0], -c[1] * w[1], -c[1] * w[2], -c[1]]
            A.append(first_row)
            A.append(second_row)

    return np.array(A)

# calculate projection matrix using Eigen Decomposition (EVD) method
def calc_projection_matrix_eigen(camera_points, world_points): 
    A = build_A_matrix(camera_points, world_points)

    A_final = np.matmul(A.T, A)

    # Eigen Decomposition (EVD) which squares condition numbers and can lead to more noise / increase error
    eigen_values, eigen_vectors = np.linalg.eigh(A_final)

    # get index of smallest eigen value
    index = np.argmin(eigen_values)

    # grab eigen vector with smallest eigen value at index (syntax correclty grabs eigen vector)
    P = eigen_vectors[:, index].reshape(3, 4) # reshape to 3 rows and 4 columns for proper P format

    # normalize so that P[2,3] = 1
    P = P / P[-1, -1]

    return P

# calculate projection matrix using SVD (Singular Value Decomposition) method (preferred)
def calc_projection_matrix(camera_points, world_points):
    A = build_A_matrix(camera_points, world_points)

    # use SVD instead (more stable than eigen-decomposition of A.T @ A)
    # better at containing noise, used more standard in the industry
    U, S, Vt = np.linalg.svd(A)

    # last row of Vt gives the solution (smallest singular value)
    P = Vt[-1, :].reshape(3, 4)

    # normalize so that P[2,3] = 1 or ||P|| = 1
    P = P / P[-1, -1]

    return P


def test_projection_matrix(camera_coord, world_coord, P):
    w = np.array([world_coord[0], world_coord[1], world_coord[2], 1])
    output = np.matmul(P, w)

    # normalize homogeneous coordinates
    u = output[0] / output[2]
    v = output[1] / output[2]

    print("Expected:", camera_coord)
    print("Predicted:", (u, v))

    return np.allclose([u, v], camera_coord, atol=1.0) # are the values close with given tolerance

# decouple P using QR Factorization
def decouple_projection_matrix(P):
    A = np.array([[P[0, 0], P[0, 1], P[0, 2]],
                  [P[1, 0], P[1, 1], P[1, 2]],
                  [P[2, 0], P[2, 1], P[2, 2]]])
    
    R, K = np.linalg.qr(A) # K is calibration, R is rotation matrix

    print("calibration matrix, K\n", K) # contains fx, fy, ox, oy
    print("rotation matrix, R\n", R) # contains rotations and tranlation vector

    # calc translation vector t
    t = np.linalg.inv(K) @ np.array([P[0, 3], P[1, 3], P[2, 3]]) # @ for matrix multiplication instead of np.matmul()
    print("translation vector, t", t)

    return K, R, t


def get_camera_parameters(K):
    return K[0, 0], K[1, 2], K[0, 2], K[1, 2] # return fx, fy, ox, oy



if __name__ == "__main__":
    # ### device0 callibration
    # # camera test points
    # blue_c = (189, 287)
    # green_c = (191, 404)
    # red_c = (237, 298)
    # yellow_c = (239, 445)
    # lime_c = (369, 295)
    # navy_c = (366, 429)

    # # world test points
    # blue_w = (0, 2.008, 2.016)
    # green_w = (0, 2.0101, 0)
    # red_w = (0, 0, 2.017)
    # yellow_w = (0, 0, 0)
    # lime_w = (2.006, 0, 2.017)
    # navy_w = (2.007, 0, 0)

    # camera_points = (blue_c, green_c, red_c, yellow_c, lime_c, navy_c)
    # world_points = (blue_w, green_w, red_w, yellow_w, lime_w, navy_w)

    # P = calc_projection_matrix(camera_points, world_points)

    # # print(test_projection_matrix(blue_c, blue_w, P))

    # K, R, t = decouple_projection_matrix(P)

    # print("fx, fy, ox, oy", get_camera_parameters(K))

    import time
    for i in range(4):
        time.sleep(2)
        capture_test_image(0, str(i)+"_", "/home/dell-user/Software/robo_sub/D0/")
        capture_test_image(2, str(i)+"_", "/home/dell-user/Software/robo_sub/D2/")