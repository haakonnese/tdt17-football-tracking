import cv2
import numpy as np
def focalLength_to_camera_matrix(focalLenght, image_size):
    w, h = image_size[0], image_size[1]
    K = np.array([
        [focalLenght, 0, w/2],
        [0, focalLenght, h/2],
        [0, 0, 1],
    ])
    return K
points_3d = np.array([[52.5, 70-25.85, 0], [52.5, 70-44.15, 0], [52.5, 70-70, 0], [43.35, 70-35, 0], [61.65, 70-35, 0]], dtype=np.float32)
pos_2d = np.array([[938.07, 593.91], [935.97, 372.24], [935.37, 218.08], [585.81, 463.86], [1301.51, 474.43]], dtype=np.float32)

camera_matrix_guess = np.eye(3)

# Use solvePnPRansac to estimate the camera matrix
retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, pos_2d, camera_matrix_guess, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)

# Print the results
print("Success:", retval)
print("Rotation Vector:", rvec)
print("Translation Vector:", tvec)

# The camera matrix is not directly returned, but you can obtain it using cv2.Rodrigues
rotation_matrix, _ = cv2.Rodrigues(rvec)
camera_matrix = np.hstack((rotation_matrix, tvec))
print("Estimated Camera Matrix:")
print(camera_matrix)
img = cv2.imread("/cluster/work/haaknes/tdt17/yolov8/data/train/1_000001.jpg")
success, rotation_vector, translation_vector = cv2.solvePnP(points_3d, pos_2d, camera_matrix, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
point, _  = cv2.projectPoints(np.array([[52.5, 35, 0]]), rotation_vector, translation_vector, camera_matrix, None, None, None, None)
# cv2.line(img, (int(pos_2d[2][0]), int(pos_2d[2][1])), (int(pos_2d[2][0] - normal[0]), int(pos_2d[2][1] - normal[1])), (0, 0, 255), 2)
# cv2.line(img, (int(pos_2d[2][0]), int(pos_2d[2][1])), (int(pos_2d[2][0] + normal[0]), int(pos_2d[2][1] + normal[1])), (0, 0, 255), 2)

distance1 = np.sqrt((pos_2d[1][0] - pos_2d[0][0])**2 + (pos_2d[1][1] - pos_2d[0][1])**2) / (44.15 - 25.85)
distance2 = np.sqrt((pos_2d[2][0] - pos_2d[1][0])**2 + (pos_2d[2][1] - pos_2d[1][1])**2) / (70-44.14)
# print(distance1, distance2)
# for i in range(5):
#     for j in range(i, 5):
#         cv2.line(img, (int(pos_2d[i][0]), int(pos_2d[i][1])), ((int(pos_2d[j][0]), int(pos_2d[j][1]))), (0, 255, 0), 2)
print(point)
cv2.circle(img, (int(point[0][0][0]), int(point[0][0][1])), 5, (0, 0, 255), -1)
# cv2.circle(img, (int(pos_2d[1][0]), int(pos_2d[1][1])), 5, (255, 0, 255), -1)
# cv2.circle(img, (int(pos_2d[2][0]), int(pos_2d[2][1])), 5, (0, 0, 255), -1)
# cv2.circle(img, (int(pos_2d[3][0]), int(pos_2d[3][1])), 5, (0, 0, 255), -1)
# cv2.circle(img, (int(pos_2d[4][0]), int(pos_2d[4][1])), 5, (0, 0, 255), -1)


# save img to file
cv2.imwrite("arrowedLine.jpg", img)
# input_pts = np.array([[pos_2d[0][0], pos_2d[0][1]], [pos_2d[2][0], pos_2d[2][1]], [pos_2d[3][0], pos_2d[3][1]], [pos_2d[4][0], pos_2d[4][1]]], dtype=np.float32)
# print(input_pts)
# output_pts = np.array([[points_3d[0][0], points_3d[0][1]], [points_3d[2][0], points_3d[2][1]], [points_3d[4][0], points_3d[4][1]], [points_3d[5][0], points_3d[5][1]]], dtype=np.float32)
# print(output_pts)
# M = cv2.getPerspectiveTransform(input_pts,output_pts*30)
# point = cv2.perspectiveTransform(np.array([[[935.97, 372.24], [1,2]]]), M)
# print(point.shape)
# print(point[0][0])
# out = cv2.warpPerspective(img,M,(105*30, 70*30))
# cv2.circle(out, center=(int(point[0][0][0]), int(point[0][0][1])), radius=5, color=(0, 0, 255), thickness=-1)
# cv2.imwrite("warped.jpg", out)

