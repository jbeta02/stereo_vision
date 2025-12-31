import test_calibration
import cv2
import numpy as np

# calc Sum of Squared Differences (SSD)
def calc_ssd(img, template, h):
    img_h = img.shape[0]
    img_w = img.shape[1]
    temp_h = template.shape[0]
    temp_w = template.shape[1]

    # template = template.astype(np.int32)
    # img = img.astype(np.int32)

    ssd_values = []

    for x in range(img_w - temp_w + 1):  # slide horizontally
        ssd = 0
        for i in range(temp_h):
            for j in range(temp_w):
                diff = int(template[i, j]) - int(img[h + i, x + j])
                ssd += diff * diff
        ssd_values.append(ssd)

    # print("sum_lst", ssd_values)

    # print("min", min(ssd_values))

    # # print("pixel x loc of template match", ssd_values.index(min(ssd_values)))

    # print("len", len(ssd_values))

    # print("val of x", ssd_values[ssd_values.index(min(ssd_values))])

    # print("val of custom", ssd_values[328])

    top_left = ssd_values.index(min(ssd_values)), h
    w, h = template.shape[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # print("top left", top_left)

    # Draw a rectangle around the matched area
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # Display the result (example, requires display environment)
    # img = np.clip(img, 0, 255).astype(np.uint8)
    # template = np.clip(template, 0, 255).astype(np.uint8)
    # cv2.imshow("Match Result", img)
    # cv2.imshow("template", template)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return top_left[::-1]


# use open cv to calc Sum of Squared Differences (SSD)
def calc_ssd_opencv(img, template):
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) # using Correlation Coefficient instead of ssd for better results

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # print("Best SSD value:", min_val)
    # print("Best match (x, y):", min_loc)

    # The min_loc gives the top-left corner of the best match
    top_left = min_loc
    # Get template dimensions to define the bounding box
    w, h = template.shape[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the matched area
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # Display the result (example, requires display environment)
    # cv2.imshow("Match Result", img)
    # cv2.imshow("template", template)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return min_loc


# solve correspondence problem
def find_depth(K, baseline, disparity):
    fx, fy, ox, oy = test_calibration.get_camera_parameters(K)

    z = baseline * fx / disparity

    return z


# same source code as build_disparity_map() but added x, y, z calculation and build world coord map as well
def get_maps(K, baseline, left_img, right_img, block_size, max_disparity=64):
    fx, fy, ox, oy = test_calibration.get_camera_parameters(K)

    height, width = left_img.shape

    depth_map = np.zeros((height, width, 3), dtype=np.float32)
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    # convert to int32 to avoid overflow
    left_img_i32 = left_img.astype(np.int32)
    right_img_i32 = right_img.astype(np.int32)

    half_block = block_size // 2 # using will center block around target pixel (round down to nearest whole number)

    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):
            best_ssd = float('inf') # treat as positive infinit for first loop iteration
            best_disp = 0

            # create template with block centered at x, y
            template = left_img_i32[y - half_block:y + half_block + 1,
                                     x - half_block: x + half_block + 1]
            
            # Shift max with x, shift min when x greater than half_block + max_disparity.
            # This will help with matching when obj is very close to left edge of screen.
            # After x reaches half_block + max_disparity min will shift with max
            min_x = max(half_block, x - max_disparity)
            max_x = x

            for right_x in range(min_x, max_x + 1):

                block = right_img_i32[y - half_block:y + half_block + 1,
                                     right_x - half_block: right_x + half_block + 1]
                
                # calc Sum of Squard differences
                # https://ieeexplore.ieee.org/document/7449303
                diff = template - block
                ssd = np.sum(diff ** 2)

                if ssd < best_ssd:
                    best_ssd = ssd
                    best_disp = x - right_x

            # scale disparity for visualization
            disparity_map[y, x] = int(best_disp * 255 / max_disparity)

            # ul, vl -> x, y
            # ur, vr -> right_z, ()
            x_world = baseline *  (x - ox) / best_disp
            y_world = baseline * fx * (y - oy) / fy * best_disp
            z_world = baseline * fx / best_disp

            depth_map[y, x] = [x_world, y_world, z_world]

    return disparity_map, depth_map



def build_disparity_map(left_img, right_img, block_size, max_disparity=64):

    height, width = left_img.shape
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    # convert to int32 to avoid overflow
    left_img_i32 = left_img.astype(np.int32)
    right_img_i32 = right_img.astype(np.int32)

    half_block = block_size // 2 # using will center block around target pixel (round down to nearest whole number)

    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):
            best_ssd = float('inf') # treat as positive infinit for first loop iteration
            best_disp = 0

            # create template with block centered at x, y
            template = left_img_i32[y - half_block:y + half_block + 1,
                                     x - half_block: x + half_block + 1]
            
            # Shift max with x, shift min when x greater than half_block + max_disparity.
            # This will help with matching when obj is very close to left edge of screen.
            # After x reaches half_block + max_disparity min will shift with max
            min_x = max(half_block, x - max_disparity)
            max_x = x

            for right_x in range(min_x, max_x + 1):

                block = right_img_i32[y - half_block:y + half_block + 1,
                                     right_x - half_block: right_x + half_block + 1]
                
                # calc Sum of Squard differences
                # https://ieeexplore.ieee.org/document/7449303
                diff = template - block
                ssd = np.sum(diff ** 2)

                if ssd < best_ssd:
                    best_ssd = ssd
                    best_disp = x - right_x

            # scale disparity for visualization
            disparity_map[y, x] = int(best_disp * 255 / max_disparity)

    return disparity_map


def build_disparity_map_opencv(left_img, right_img, block_size, max_disparity=64):
    stereo = cv2.StereoBM_create(numDisparities=max_disparity, blockSize=block_size)
    disparity = stereo.compute(left_img, right_img)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return disparity_normalized
                

if __name__ == "__main__":

    ### device0 callibration (left)
    # camera test points
    blue_c = (189, 287)
    green_c = (191, 404)
    red_c = (237, 298)
    yellow_c = (239, 445)
    lime_c = (369, 295)
    navy_c = (366, 429)

    # world test points
    blue_w = (0, 2.008, 2.016)
    green_w = (0, 2.0101, 0)
    red_w = (0, 0, 2.017)
    yellow_w = (0, 0, 0)
    lime_w = (2.006, 0, 2.017)
    navy_w = (2.007, 0, 0)

    camera_points = (blue_c, green_c, red_c, yellow_c, lime_c, navy_c)
    world_points = (blue_w, green_w, red_w, yellow_w, lime_w, navy_w)

    P = test_calibration.calc_projection_matrix(camera_points, world_points)

    # print(test_projection_matrix(blue_c, blue_w, P))

    K, R, t = test_calibration.decouple_projection_matrix(P)

    print("cam left: fx, fy, ox, oy", test_calibration.get_camera_parameters(K))


    ### device2 calibration (right)
    # camera test points
    blue_c = (255, 300)
    green_c = (253, 425)
    red_c = (278, 316)
    yellow_c = (279, 464)
    lime_c = (425, 311)
    navy_c = (421, 425)

    # world test points
    blue_w = (0, 2.008, 2.016)
    green_w = (0, 2.0101, 0)
    red_w = (0, 0, 2.017)
    yellow_w = (0, 0, 0)
    lime_w = (2.006, 0, 2.017)
    navy_w = (2.007, 0, 0)

    camera_points = (blue_c, green_c, red_c, yellow_c, lime_c, navy_c)
    world_points = (blue_w, green_w, red_w, yellow_w, lime_w, navy_w)

    P = test_calibration.calc_projection_matrix(camera_points, world_points)

    # print(test_projection_matrix(blue_c, blue_w, P))

    K, R, t = test_calibration.decouple_projection_matrix(P)

    print("cam right: fx, fy, ox, oy", test_calibration.get_camera_parameters(K))

    img_path0 = "/home/dell-user/Software/robo_sub/capture_device0.png" # left
    img_path1 = "/home/dell-user/Software/robo_sub/capture_device2.png" # right
    # img_path0 = "/home/dell-user/Software/robo_sub/t1.png"
    # img_path1 = "/home/dell-user/Software/robo_sub/t2.png"
    img0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

    # test different block sizes
    out_image0 = build_disparity_map(img0, img1, 5 )
    out_image1 = build_disparity_map(img0, img1, 10)
    out_image2 = build_disparity_map(img0, img1, 25)
    out_image3 = build_disparity_map(img0, img1, 35)
    out_image4 = build_disparity_map(img0, img1, 50)