# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt

from utils import *

#可调超参:pair_threshold , homo_matrix threshold,homo_matrix iteration num
IMGDIR = 'Problem2Images'


######################################################################################################

def gradient_x(img):
    # TODO
    gray_img_cv=img
    if gray_img_cv.shape[2]==3: #三通道BGR2一通道Gray
        gray_img_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray_img_cv=np.float64(gray_img_cv)
    blur_gray_img_array = ndimage.gaussian_filter(gray_img_cv,sigma=3.0,mode='reflect') #高斯模糊 增加连续性

    kernel_x = np.array([[1,0,-1],
                         [2,0,-2],
                         [1,0,-1]]) # sobel operation
    
    grad_x_img_array = ndimage.convolve(blur_gray_img_array,kernel_x)
    
    return grad_x_img_array


def gradient_y(img):
    # TODO
    gray_img_cv=img
    if gray_img_cv.shape[2]==3: #三通道BGR2一通道Gray
        gray_img_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img_cv=np.float64(gray_img_cv)
    blur_gray_img_array = ndimage.gaussian_filter(gray_img_cv,sigma=3.0,mode='reflect') #高斯模糊 增加连续性
    kernel_y = np.array([[1,2,1],
                         [0,0,0],
                         [-1,-2,-1]]) # sobel operation
    grad_y_img_array = ndimage.convolve(blur_gray_img_array,kernel_y)
    
    return grad_y_img_array


def harris_response(img, alpha, win_size):
    # TODO
    img_x_grad_array = gradient_x(img)
    img_y_grad_array = gradient_y(img)
    ###用ndimage gaussin
    Ix2 = gaussian_blur_2d(img_x_grad_array*img_x_grad_array,k_size=win_size,k_sigma=5)
    Iy2 = gaussian_blur_2d(img_y_grad_array*img_y_grad_array,k_size=win_size,k_sigma=5)
    IxIy = gaussian_blur_2d(img_x_grad_array*img_y_grad_array,k_size=win_size,k_sigma=5)

    detM = Ix2*Iy2-IxIy**2
    trM = Ix2+Iy2

    output_harris_array=detM-alpha*(trM**2)

    return output_harris_array


def corner_selection(R, thresh, min_dist):
    # TODO
    R_ = ndimage.maximum_filter(R,size=min_dist)
    pixels_lst = []
    select_points  = np.argwhere(R_>thresh) # get pixel where local maximum

    expand_=min_dist//2
    R_height,R_width = np.shape(R_)
    for point_index in range(len(select_points)):
        center_y = select_points[point_index][0]
        center_x = select_points[point_index][1]

        if center_y<expand_ or center_y>R_height-expand_-1 or center_x<expand_ or center_x>R_width-expand_-1:
            continue
        
        box_up = max(center_y-expand_,0)
        box_down = min(center_y+expand_+1,R_height)
        box_left = max(center_x-expand_,0)
        box_right = min(center_x+expand_+1,R_width)
        
        if(np.unique(R_[box_up:box_down,box_left:box_right]).size==1 and np.sum(R_[box_up:box_down,box_left:box_right])!=0):
            pixels_lst.append(tuple((center_x,center_y)))

    return pixels_lst


def histogram_of_gradients(img, pix):
    # TODO
    features=[]
    img_grad_x = gradient_x(img)
    img_grad_y = gradient_y(img)
    img_angle_array = np.arctan2(img_grad_y,img_grad_x)*180/np.pi+180 # 0,360
    img_grad_array = np.sqrt(img_grad_x**2+img_grad_y**2)

    for pixel_position in pix:
        pix_outer_angle_box = get_surrounding_box_array(img_angle_array,pixel_position,9)
        
        pix_divided_angle_boxs = divide_box_array(pix_outer_angle_box,divide_n=2) #divided into n*n


        pix_outer_grad_box = get_surrounding_box_array(img_grad_array,pixel_position,9)
        pix_divided_grad_boxs = divide_box_array(pix_outer_grad_box,divide_n=2)

        block_features=[]
        dir_num = 8
        for cell_index in range(len(pix_divided_angle_boxs)):
            cell_histogram = angle_grad_histogram(pix_divided_angle_boxs[cell_index],pix_divided_grad_boxs[cell_index],dir_num) # 分成8个direction
            block_features.extend(item_ for item_ in cell_histogram)

        
        principle_agnle_index = block_features.index(max(block_features))
        principle_agnle = (principle_agnle_index%dir_num)*(360//dir_num)   #找到主方向


        pix_new_angle_box = np.array((pix_outer_angle_box-principle_agnle+360)%360)
        pix_new_divided_angle_boxs = divide_box_array(pix_new_angle_box,divide_n=2)

        new_block_features=[]
        for new_cell_index in range(len(pix_new_divided_angle_boxs)):
            new_cell_histogram = angle_grad_histogram(pix_new_divided_angle_boxs[new_cell_index],pix_divided_grad_boxs[new_cell_index],dir_num) # 分成8个direction
            new_block_features.extend(item_ for item_ in new_cell_histogram)
        
        features.append(new_block_features)
    
    return np.array(features)


def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.06, 9)
    R2 = harris_response(img_2, 0.06, 9)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    # print(len(cor1))
    
    
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')

    threshold = 0.5 # 0.3 for q1  0.5 for test2
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    # print(p1," ",p2)
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p]) <= threshold:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])
    rate = np.shape(pixels_1)[0]/min_len

    print('final_choose_num:',len(pixels_1))
    assert rate >= 0.03, "Fail to Match!"
    return pixels_1, pixels_2

def test_matching():    
    img_1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/1_2.jpg')

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_1, img_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        plt.plot([x1, x2+W_1], [y1, y2])
        #plt.plot([y1, y2+W_1], [x1, x2])

    # plt.show()
    plt.savefig('test_match_new.jpg')


def compute_homography(pixels_1, pixels_2):
    # TODO

    pixels_pair_num = len(pixels_1)
    assert pixels_pair_num>=4,"To compute homography need more pixels-pair\n"

    pixel_matrix1=convert_to_homogeneous_coordinates(pixels_1)
    pixel_matrix2=convert_to_homogeneous_coordinates(pixels_2)
    pixel_matrix_ = np.zeros((pixels_pair_num*2,9))
    pixel_matrix_[::2,0:3]=-pixel_matrix1
    pixel_matrix_[1::2,3:6]=-pixel_matrix1
    pixel_matrix_[::2,6:9]=pixel_matrix1*pixel_matrix2[:,0:1]
    pixel_matrix_[1::2,6:9]=pixel_matrix1*pixel_matrix2[:,1:2]

    U_,S_,V_ = np.linalg.svd(np.transpose(pixel_matrix_).dot(pixel_matrix_))
    smallest_index = np.argmin(S_)
    get_col = V_[smallest_index] # find smallest eigenvalue
    ret_matrix = np.reshape(get_col, (3, 3))

    return ret_matrix

def align_pair(pixels_1, pixels_2):
    pixels_pair_num = len(pixels_1)
    assert pixels_pair_num>=4,"To compute need more pixels-pair\n"

    final_homo_matrix=np.zeros((3,3)) # initial matrix
    final_inliers_num = -1
    pixels_num =len(pixels_1)
    itration_times = 1200
    threshold_ = 2
    homo_coor_pixels_1=convert_to_homogeneous_coordinates(pixels_1)
    homo_coor_pixels_2=convert_to_homogeneous_coordinates(pixels_2)

    range_num = [i for i in range(pixels_num)]
    for it_ in range(itration_times):
        random_samples_indices = np.random.choice(range_num,4,replace=False)
        sample_pixels_1=[]
        sample_pixels_2=[]
        for index_ in random_samples_indices:
            sample_pixels_1.append(pixels_1[index_])
            sample_pixels_2.append(pixels_2[index_])

        sample_homo_matrix = compute_homography(sample_pixels_1,sample_pixels_2)
        sample_matrix1_dot_homo = sample_homo_matrix.dot(homo_coor_pixels_1.transpose())
        sample_distance_matrix_norm_transpose = (sample_matrix1_dot_homo/sample_matrix1_dot_homo[2]).transpose()
        
        sample_distance_matrix = (sample_distance_matrix_norm_transpose-homo_coor_pixels_2)[:,0:2]
        sample_distance_norm = np.array([np.linalg.norm(tmp_) for tmp_ in sample_distance_matrix])

        sample_inlier_cnt = np.sum([sample_distance_norm<threshold_])

        if sample_inlier_cnt>final_inliers_num:
            final_homo_matrix=sample_homo_matrix
            final_inliers_num=sample_inlier_cnt

    return final_homo_matrix

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    # trans_x = (trans_x/trans_z)
    # trans_y = (trans_y/trans_z)
    trans_x = (trans_x/trans_z).astype(np.float32)
    trans_y = (trans_y/trans_z).astype(np.float32)
    # print(trans_x)
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)

    
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1 + est_img_2*alpha2
    return est_img

def my_stitch_blend(img_1,img_2,est_homo):
    img_1 = img_1.astype(np.float32)
    img_2 = img_2.astype(np.float32)
    
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    
    corners_1 = np.array([[0, 0,1], [0, h1,1], [w1, 0,1], [w1, h1,1]], dtype=np.float32)
    corners_1_homogeneous = est_homo.dot(corners_1.transpose())
    
    corners_1_change = []
    for i in range(2):
        tmp_ = corners_1_homogeneous[:,i]
        corners_1_change.append(tmp_[0:2]/tmp_[2])

    corners_1_change=np.array(corners_1_change)
    
    corners_2 = np.array([[0, 0], [0, h2], [w2, 0], [w2, h2]], dtype=np.float32)
    all_corners = np.concatenate((corners_1_change,corners_2),axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))
    
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x=x.astype(np.float32)
    y=y.astype(np.float32)


    homo_inv = np.linalg.pinv(est_homo).astype(np.float32)
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    trans_x = (homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2])/trans_z
    trans_y = (homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2])/trans_z
    
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    est_img_2 = cv2.remap(img_2, x,y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)

    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1 + est_img_2*alpha2

    est_img = np.clip(est_img, 0, 255).astype(np.uint8)
    
    return est_img

def generate_panorama(ordered_img_seq):
    len = np.shape(ordered_img_seq)[0]
    mid = int(len/2) # middle anchor
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < len):
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano



if __name__ == '__main__':
    # # make image list
    # # call generate panorama and it should work well
    # # save the generated image following the requirements
    #test_matching()
    
    # an example
   
    # parrington 0 1 2 3 4 threshold=0.5
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn00.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn01.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn02.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn03.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn04.jpg')
    # img_6 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn05.jpg')
    # img_7 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn06.jpg')
    # img_8 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn07.jpg')

    # library 3 4 5  threshold=0.5
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/library/4.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/library/5.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/library/6.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/library/7.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/library/8.jpg')

    # grail  1 2 3 4 threshold=0.5
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail02.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail03.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail04.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail05.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail06.jpg')

    # Xue-Mountain 1 2 3 4 5 threshold=0.5
    #img_1 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0171.jpg')
    #img_2 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0172.jpg')
    #img_3 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0173.jpg')
    #img_4 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0174.jpg')
    #img_5 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0175.jpg')


    #img_list=[]
    #img_list.append(img_1)
    #img_list.append(img_2)
    #img_list.append(img_3)
    #img_list.append(img_4)
    #img_list.append(img_5)

    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements

    #test_matching() #threshold=0.3
    
    # an example
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn00.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn01.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn02.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn03.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn04.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    #img_list.append(img_4)
    #img_list.append(img_5)
    pano = generate_panorama(img_list) #threshold=0.5
    cv2.imwrite("outputs/panorama_new.jpg", pano)