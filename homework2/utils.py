import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt

def gauss_2d_calculate(k_sigma,center_point,tar_point):
    distance_ = np.linalg.norm(center_point-tar_point)
    return np.exp(-(distance_*distance_)/(2*k_sigma*k_sigma))

def gaussian_blur_kernel_2d(k_size,k_sigma=1):
    gauss_kernel = np.zeros((k_size,k_size), dtype=np.float32)
    mean_ = k_size//2
    for i in range(0,k_size):
        for j in range(0,k_size):
            gauss_kernel[i,j] = gauss_2d_calculate(k_sigma,np.array([mean_,mean_]),np.array([i,j]))
    return gauss_kernel/np.sum(gauss_kernel)

def gaussian_blur_2d(img_array,k_size,k_sigma=1):
    return ndimage.convolve(img_array,gaussian_blur_kernel_2d(k_size,k_sigma=k_sigma))

def get_surrounding_box_array(feature_array,center_position,box_size=9):
    center_x = center_position[0]
    center_y = center_position[1]
    
    array_width = feature_array.shape[1] # 水平方向 
    array_height = feature_array.shape[0] # 竖直方向
    # box_size 建议为奇数
    box_expand = box_size//2
    box_up = max(center_y-box_expand,0)
    box_down = min(center_y+box_expand,array_height)
    box_left = max(center_x-box_expand,0)
    box_right = min(center_x+box_expand,array_width)

    ret_box = feature_array[box_up:box_down,box_left:box_right]
    return ret_box

def divide_box_array(box_array,divide_n=2):
    box_height = box_array.shape[0]
    box_width = box_array.shape[1]
    divide_height_step = box_height//divide_n
    divide_width_step = box_width//divide_n
    ret_arr =[]
    for i in range(divide_n):
        for j in range(divide_n):
            divide_box_up = i*divide_height_step
            divide_box_down = (i+1)*divide_height_step #if i!=divide_n else box_height
            divide_box_left = j*divide_width_step
            divide_box_right = (j+1)*divide_width_step #if j !=divide_n else box_width

            divide_box =box_array[divide_box_up:divide_box_down,divide_box_left:divide_box_right].flatten()
            ret_arr.append(divide_box)
    
    # print(ret_arr)
    return np.array(ret_arr)

def angle_grad_histogram(angle_array,grad_array,n):
    # print(angle_array)
    
    lower_bound = 0
    upper_bound = 360
    divide_step = (upper_bound-lower_bound)//n

    histogram_lst = np.zeros(n)

    # print(grad_array)
    # print(angle_array)
    for pix_index in range(len(angle_array)):
        select_index = int(angle_array[pix_index]//divide_step)
        # print(select_index)
        down_ = select_index*divide_step
        l_ = (angle_array[pix_index]-down_)/divide_step

        histogram_lst[select_index]+=grad_array[pix_index]*(1-l_)
        histogram_lst[(select_index+1)%n]+=grad_array[pix_index]*(l_) #线性加权

    if np.sum(grad_array)==0:
        return np.zeros(n)
    
    return histogram_lst/np.linalg.norm(histogram_lst)

def convert_to_homogeneous_coordinates(pixels):
    pinxels_array = np.array(pixels)
    homo_coori = np.ones((np.shape(pinxels_array)[0],3))
    homo_coori[:,0:2]=pixels
    return homo_coori