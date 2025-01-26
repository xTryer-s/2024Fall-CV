from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2] + I1.size[0], matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()

def normalize_points(pts):
    # Normalize points
    # 1. calculate mean and std
    # 2. build a transformation matrix
    # :return normalized_pts: normalized points
    # :return T: transformation matrix from original to normalized points

    # input: pts=[[x11,y11],[x12,y12]....]
    pts_mean = np.mean(pts,axis=0)
    pts_std = np.std(pts,axis=0)

    trans_matrix=np.array([
        [1/pts_std[0],0,-pts_mean[0]/pts_std[0]],
        [0,1/pts_std[1],-pts_mean[1]/pts_std[1]],
        [0,0,1]
    ])
    tmp_array = np.ones((1,np.shape(pts)[0]))
    homo_pts = np.vstack((pts.T,tmp_array))
    # like:
    # [[1. 2. 5.]
    # [2. 3. 6.]
    # [1. 1. 1.]]
    normalized_pts=(np.dot(trans_matrix,homo_pts).T)[:,:2]
    return normalized_pts, trans_matrix

def fit_fundamental(matches):
    # Calculate fundamental matrix from ground truth matches
    # 1. (normalize points if necessary)
    # 2. (x2, y2, 1) * F * (x1, y1, 1)^T = 0 -> AX = 0
    # X = (f_11, f_12, ..., f_33) 
    # build A(N x 9) from matches(N x 4) according to Eight-Point Algorithm
    # 3. use SVD (np.linalg.svd) to decomposite the matrix
    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    # 5. use SVD to decomposite F, set the smallest eigenvalue as 0, and recalculate F
    # 6. Report your fundamental matrix results

    # Normalized:
    pic1_pts = matches[:,:2]
    pic2_pts = matches[:,2:]

    normalized_pic1_pts,T_matrix_1 = normalize_points(pic1_pts)
    normalized_pic2_pts,T_matrix_2 = normalize_points(pic2_pts)

    N_ = matches.shape[0]
    A_matrix = np.zeros((N_, 9))
    for i in range(matches.shape[0]):
        x1, y1 = normalized_pic1_pts[i]
        x2, y2 = normalized_pic2_pts[i]
        A_matrix[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Use SVD to decomposite the matrix
    U_,S_,V_ = np.linalg.svd(np.transpose(A_matrix).dot(A_matrix))
    smallest_index = np.argmin(S_)
    get_col = V_[smallest_index] # find smallest eigenvalue
    F_matrix_init = np.reshape(get_col, (3, 3))
    
    f_U, f_S, f_V = np.linalg.svd(F_matrix_init)
    f_S_mat=np.diag(f_S)
    f_S_mat[np.argmin(f_S)]=0
    final_F = np.dot(f_U, np.dot(f_S_mat, f_V))
    
    final_F = np.dot(T_matrix_2.T, np.dot(final_F, T_matrix_1))
    
    print("Fundamental Matrix:\n", final_F)


    # # unnormalized
    # pic1_pts = matches[:,:2]
    # pic2_pts = matches[:,2:]


    # N_ = matches.shape[0]
    # A_matrix = np.zeros((N_, 9))
    # for i in range(matches.shape[0]):
    #     x1, y1 = pic1_pts[i]
    #     x2, y2 = pic2_pts[i]
    #     A_matrix[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # # Use SVD to decomposite the matrix
    # U_,S_,V_ = np.linalg.svd(np.transpose(A_matrix).dot(A_matrix))
    # smallest_index = np.argmin(S_)
    # get_col = V_[smallest_index] # find smallest eigenvalue
    # F_matrix_init = np.reshape(get_col, (3, 3))
    
    # f_U, f_S, f_V = np.linalg.svd(F_matrix_init)
    # f_S_mat=np.diag(f_S)
    # f_S_mat[2]=0
    # final_F = np.dot(f_U, np.dot(f_S_mat, f_V))
    
    # print("Task1 Fundamental Matrix:\n", final_F)
    
    return final_F


def visualize_fundamental(matches, F, I1, I2):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())   # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis = 1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2],np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10    # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2],matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()

def evaluate_fundamental(matches, F):
    N = len(matches)
    points1, points2 = matches[:, :2], matches[:, 2:]
    points1_homogeneous = np.concatenate([points1, np.ones((N, 1))], axis=1)
    points2_homogeneous = np.concatenate([points2, np.ones((N, 1))], axis=1)
    product = np.dot(np.dot(points2_homogeneous, F), points1_homogeneous.T)
    diag = np.diag(product)
    residual = np.mean(diag ** 2)
    return residual

## Task 0: Load data and visualize
## load images and match files for the first example
## matches[:, :2] is a point in the first image
## matches[:, 2:] is a corresponding point in the second image

library_image1 = Image.open('data/library1.jpg')
library_image2 = Image.open('data/library2.jpg')
library_matches = np.loadtxt('data/library_matches.txt')

lab_image1 = Image.open('data/lab1.jpg')
lab_image2 = Image.open('data/lab2.jpg')
lab_matches = np.loadtxt('data/lab_matches.txt')

# # Visualize matches
# visualize_matches(library_image1, library_image2, library_matches)
# visualize_matches(lab_image1, lab_image2, lab_matches)


# Task 1: Fundamental matrix
# display second image with epipolar lines reprojected from the first image

# first, fit fundamental matrix to the matches
# Report your fundamental matrices, visualization and evaluation results

library_F = fit_fundamental(library_matches) # this is a function that you should write
visualize_fundamental(library_matches, library_F, library_image1, library_image2)
print(evaluate_fundamental(library_matches, library_F))
assert evaluate_fundamental(library_matches, library_F) < 0.5

lab_F = fit_fundamental(lab_matches) # this is a function that you should write
visualize_fundamental(lab_matches, lab_F, lab_image1, lab_image2) 
print("evaluate:",evaluate_fundamental(lab_matches, lab_F))
assert evaluate_fundamental(lab_matches, lab_F) < 0.5
tmp_ = input("Task1 over! Press enter to continue~\n")



## Task 2: Camera Calibration

def calc_projection(points_2d, points_3d):
    # Calculate camera projection matrices
    # 1. Points_2d = P * Points_3d -> AX = 0
    # X = (p_11, p_12, ..., p_34) is flatten of P
    # build matrix A(2*N, 12) from points_2d
    # 2. SVD decomposite A
    # 3. take the eigen vector(12, ) of smallest eigen value
    # 4. return projection matrix(3, 4)
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return P: projection matrix

    # print(np.shape(points_2d),np.shape(points_3d))
    point_pair_num = np.shape(points_2d)[0]
    A_mat = np.zeros((2*point_pair_num,12))
    for i in range(0,point_pair_num):
        x_i,y_i=points_2d[i,:2]
        X_i,Y_i,Z_i=points_3d[i,:3]
        A_mat[2*i]=[X_i,Y_i,Z_i,1,0,0,0,0,-x_i*X_i,-x_i*Y_i,-x_i*Z_i,-x_i]
        A_mat[2*i+1]=[0,0,0,0,X_i,Y_i,Z_i,1,-y_i*X_i,-y_i*Y_i,-y_i*Z_i,-y_i]

    U_,S_,V_ = np.linalg.svd(np.transpose(A_mat).dot(A_mat))
    smallest_index = np.argmin(S_)
    get_col = V_[smallest_index] # find smallest eigenvalue
    final_P_mat = np.reshape(get_col, (3, 4))
    print("Projection_matrix:\n",final_P_mat)
    return final_P_mat

def rq_decomposition(P):
    # Use RQ decomposition to calculte K, R, T
    # 1. perform QR decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    # 2. normalize to set K[2, 2] = 1
    # 3. calculate T by P = K[R|T]
    # :param P: projection matrix
    # :return K, R, T: camera matrices

    # order of step2 & step3 should be changed
    # perform RQ decomposition

    # print(type(P),np.shape(P))
    R_mat,Q_mat = scipy.linalg.rq(P[:,:3])
    K=R_mat
    R=Q_mat
    T=np.linalg.solve(np.dot(K,R),P[:,3])
    K=R_mat/R_mat[2,2]
    # print("temp",R_mat[2,2])
    print('K_matrix:\n',K)
    print('R_matrix:\n',R)
    print('T_matrix:\n',T)
    return K, R, T

def evaluate_points(P, points_2d, points_3d):
    # Visualize the actual 2D points and the projected 2D points calculated from
    # the projection matrix
    # You do not need to modify anything in this function, although you can if you
    # want to
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def triangulate_points(P1, P2, point1, point2):
    # Use linear least squares to triangulation 3d points
    # 1. Solve: point1 = P1 * point_3d
    #           point2 = P2 * point_3d
    # 2. use SVD decomposition to solve linear equations
    # :param P1, P2 (3 x 4): projection matrix of two cameras
    # :param point1, point2: points in two images
    # :return point_3d: 3D points calculated by triangulation
    A_mat = np.zeros((4,4))
    A_mat[0]=P1[0]-point1[0]*P1[2]
    A_mat[1]=P1[1]-point1[1]*P1[2]
    A_mat[2]=P2[0]-point2[0]*P2[2]
    A_mat[3]=P2[1]-point2[1]*P2[2]
    
    U_,S_,V_ = np.linalg.svd(np.transpose(A_mat).dot(A_mat))
    smallest_index = np.argmin(S_)
    get_col = V_[smallest_index] # find smallest eigenvalue
    # print(get_col)
    point_3d = get_col[:3]/get_col[3]

    return point_3d

lab_points_3d = np.loadtxt('data/lab_3d.txt')

projection_matrix = dict()
for key, points_2d in zip(["lab_a", "lab_b"], [lab_matches[:, :2], lab_matches[:, 2:]]):
    P = calc_projection(points_2d, lab_points_3d)
    points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
    distance = np.mean(np.linalg.norm(points_2d - points_3d_proj))
    # Check: residual should be < 20 and distance should be < 4
    print(f'{key}: residual:{residual},distance:{distance}')
    assert residual < 20.0 and distance < 4.0
    projection_matrix[key] = P

tmp_ = input("Task2 over! Press enter to continue~\n")

## Task 3
## Camera Centers
projection_library_a = np.loadtxt('data/library1_camera.txt')
projection_library_b = np.loadtxt('data/library2_camera.txt')
projection_matrix["library_a"] = projection_library_a
projection_matrix["library_b"] = projection_library_b

# for P in projection_matrix.values():
#     # Paste your K, R, T results in your report
#     K, R, T = rq_decomposition(P)

# print("\n\nmy_check")
for key_,P in projection_matrix.items():
    print(f'{key_}:')
    # Paste your K, R, T results in your report
    K, R, T = rq_decomposition(P)

str_ = input("Task 3 over! Press enter to continue~\n")

## Task 4: Triangulation
lab_points_3d_estimated = []
for point_2d_a, point_2d_b, point_3d_gt in zip(lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d):
    point_3d_estimated = triangulate_points(projection_matrix['lab_a'], projection_matrix['lab_b'], point_2d_a, point_2d_b)

    # Residual between ground truth and estimated 3D points
    residual_3d = np.sum(np.linalg.norm(point_3d_gt - point_3d_estimated))
    assert residual_3d < 0.1
    lab_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
lab_points_3d_estimated = np.stack(lab_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['lab_a'], lab_matches[:, :2], lab_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['lab_b'], lab_matches[:, 2:], lab_points_3d_estimated)
print(f'residual_lab_a:{residual_a},residual_lab_b:{residual_b}')
assert residual_a < 20 and residual_b < 20

library_points_3d_estimated = []
for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
    point_3d_estimated = triangulate_points(projection_matrix['library_a'], projection_matrix['library_b'], point_2d_a, point_2d_b)
    library_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
library_points_3d_estimated = np.stack(library_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['library_a'], library_matches[:, :2], library_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['library_b'], library_matches[:, 2:], library_points_3d_estimated)
print(f'residual_library_a:{residual_a},residual_library_b:{residual_b}')
assert residual_a < 30 and residual_b < 30

str_ = input('Task 4 over! Press enter to continue~\n')
## Task 5: Fundamental matrix estimation without ground-truth matches
import cv2

def fit_fundamental_for_task5(matches):
    # Normalized:
    pic1_pts = matches[:,:2]
    pic2_pts = matches[:,2:]

    normalized_pic1_pts,T_matrix_1 = normalize_points(pic1_pts)
    normalized_pic2_pts,T_matrix_2 = normalize_points(pic2_pts)

    N_ = matches.shape[0]
    A_matrix = np.zeros((N_, 9))
    for i in range(matches.shape[0]):
        x1, y1 = normalized_pic1_pts[i]
        x2, y2 = normalized_pic2_pts[i]
        A_matrix[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Use SVD to decomposite the matrix
    U_,S_,V_ = np.linalg.svd(np.transpose(A_matrix).dot(A_matrix))
    smallest_index = np.argmin(S_)
    get_col = V_[smallest_index] # find smallest eigenvalue
    F_matrix_init = np.reshape(get_col, (3, 3))
    
    f_U, f_S, f_V = np.linalg.svd(F_matrix_init)
    f_S_mat=np.diag(f_S)
    f_S_mat[np.argmin(f_S)]=0
    final_F = np.dot(f_U, np.dot(f_S_mat, f_V))
    
    final_F = np.dot(T_matrix_2.T, np.dot(final_F, T_matrix_1))
    return final_F

def convert_to_homogeneous_coordinates(pixels):
    pinxels_array = np.array(pixels)
    homo_coori = np.ones((np.shape(pinxels_array)[0],3))
    homo_coori[:,0:2]=pixels
    return homo_coori

def align_pair_with_RANSAC(pixels_1, pixels_2):
    pixels_pair_num = len(pixels_1)
    assert pixels_pair_num>=4,"To compute need more pixels-pair\n"

    final_fundamental_matrix=np.zeros((3,3)) # initial matrix
    final_inliers_num = -1
    final_matches =[]
    final_res_ = 0
    pixels_num =len(pixels_1)
    itration_times = 1200
    threshold_ = 0.02

    homogenenous_pixels_1 = convert_to_homogeneous_coordinates(pixels_1)
    homogenenous_pixels_2 = convert_to_homogeneous_coordinates(pixels_2)
    # print("pixels_num:",pixels_num)
    range_num = [i for i in range(pixels_num)]
    for it_ in range(itration_times):
        random_samples_indices = np.random.choice(range_num,int(pixels_num/50),replace=False)
        assert int(pixels_num/50)>=4
        sample_pixels_1=[]
        sample_pixels_2=[]
        for index_ in random_samples_indices:
            sample_pixels_1.append(pixels_1[index_])
            sample_pixels_2.append(pixels_2[index_])

        sample_fundamental_matrix = fit_fundamental_for_task5(np.hstack((sample_pixels_1,sample_pixels_2)))

        sample_inlier_cnt = 0
        sample_res_sum=0
        sample_final_matches = []
        for match_index in range(pixels_num):
            check_point1 = homogenenous_pixels_1[match_index]
            check_point2 = homogenenous_pixels_2[match_index]
            res_ = np.abs(np.dot(check_point2.transpose(),np.dot(sample_fundamental_matrix,check_point1)))
            
            if res_<threshold_:
                # print(res_)
                sample_inlier_cnt+=1
                sample_res_sum+=res_
                sample_final_matches.append(np.hstack((pixels_1[match_index],pixels_2[match_index])))
                


        if sample_inlier_cnt>final_inliers_num:
            final_fundamental_matrix=sample_fundamental_matrix
            final_inliers_num=sample_inlier_cnt
            final_res_=sample_res_sum
            final_matches=sample_final_matches

    return final_fundamental_matrix,final_inliers_num,final_res_,final_matches


def fit_fundamental_without_gt(image1, image2):
    # Calculate fundamental matrix without groundtruth matches
    # 1. convert the images to gray
    # 2. compute SIFT keypoints and descriptors
    # 3. match descriptors with Brute Force Matcher
    # 4. select good matches
    # 5. extract matched keypoints
    # 6. compute fundamental matrix with RANSAC
    # :param image1, image2: two-view images
    # :return fundamental_matrix
    # :return matches: selected matched keypoints 
    gray_img1 = image1
    gray_img2 = image2
    if gray_img1.shape[2]==3: 
        gray_img1 = cv2.cvtColor(gray_img1,cv2.COLOR_BGR2GRAY)
    if gray_img2.shape[2]==3: 
        gray_img2 = cv2.cvtColor(gray_img2,cv2.COLOR_BGR2GRAY)

    # cv2.sift
    my_sift = cv2.SIFT_create()
    keypoints_1,description_1 = my_sift.detectAndCompute(gray_img1,None)
    keypoints_2,description_2 = my_sift.detectAndCompute(gray_img2,None)

    # cv2.BFMatcher
    my_bf = cv2.BFMatcher()
    # use knnmatch to get two match pair
    get_matches = my_bf.knnMatch(description_1,description_2,k=2)

    selected_matchs =[]
    for match_ in get_matches:
        if match_[0].distance<0.8*match_[1].distance:
            selected_matchs.append(match_[0])

    points_selected_1 = np.array([keypoints_1[pair_.queryIdx].pt for pair_ in selected_matchs]).astype(np.float32)
    points_selected_2 = np.array([keypoints_2[pair_.trainIdx].pt for pair_ in selected_matchs]).astype(np.float32)
    
    fundamental_matrix, inliers_num,residuals_sum,final_matches = align_pair_with_RANSAC(points_selected_1,points_selected_2)

    final_matches=np.array(final_matches)

    mean_residuals = residuals_sum/inliers_num

    print("get inliers_num:",inliers_num)
    print("get average residual for the inliers:",mean_residuals)

    #print(np.shape(final_matches))
    return fundamental_matrix, final_matches


house_image1 = Image.open('data/library1.jpg')
house_image2 = Image.open('data/library2.jpg')

house_F, house_matches = fit_fundamental_without_gt(np.array(house_image1), np.array(house_image2))
visualize_fundamental(house_matches, house_F, house_image1, house_image2)
