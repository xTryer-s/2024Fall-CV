import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple

import time


def normalize_disparity_map(disparity_map):
    '''Normalize disparity map for visualization 
    disparity should be larger than zero
    '''
    return np.maximum(disparity_map, 0.0) / (disparity_map.max() + 1e-10)


def visualize_disparity_map(disparity_map, gt_map, save_path=None):
    '''Visualize or save disparity map and compare with ground truth
    '''
    # mine
    np.set_printoptions(threshold=np.inf)
    # print(gt_map)
    # print(disparity_map)
    # Normalize disparity maps
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize or save to file
    if save_path is None:
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imshow(concat_map, 'gray')
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imsave(save_path, concat_map, cmap='gray')


def task1_compute_disparity_map_simple(
    ref_img: np.ndarray,        # shape (H, W)
    sec_img: np.ndarray,        # shape (H, W)
    window_size: int, 
    disparity_range: Tuple[int, int],   # (min_disparity, max_disparity)
    matching_function: str      # can be 'SSD', 'SAD', 'normalized_correlation'
):
    '''Assume image planes are parallel to each other
    Compute disparity map using simple stereo system following the steps:
    1. For each row, scan all pixels in that row
    2. Generate a window for each pixel in ref_img
    3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img 
    4. Select the best disparity that minimize window difference between ref_img[row, col] and sec_img[row, col - d]
    '''
    task1_time_start = time.time()
    img_H,img_W=np.shape(ref_img)
    min_disparity,max_disparity = disparity_range

    # pad the img to scan
    padding_ = window_size//2
    pad_ref_img = np.pad(ref_img,((padding_,padding_),(padding_,padding_)),'edge').astype(np.float64)
    pad_sec_img = np.pad(sec_img,((padding_,padding_),(padding_,padding_)),'edge').astype(np.float64)

    disparity_map = np.zeros(np.shape(ref_img),dtype=np.float32)
    for row_ in range(padding_,padding_+img_H):
        # scan each row's all piexls
        for col_ in range(padding_,padding_+img_W):
            # generate a window
            pixel_window_ref = pad_ref_img[row_-padding_:row_+padding_,col_-padding_:col_+padding_]
            cur_disparity=min_disparity
            cur_match_result=0

            for d_ in range(min_disparity,max_disparity+1):
                if col_-d_<padding_ or col_-d_>img_W:
                    continue
                # get window of sec_img
                pixel_window_sec = pad_sec_img[row_-padding_:row_+padding_,col_-d_-padding_:col_-d_+padding_]

                # print(pixel_window_ref)
                # print(pixel_window_sec)
                # print("#")

                tmp_match_result = 0
                if matching_function=='SSD':
                    tmp_window_distance = pixel_window_ref-pixel_window_sec
                    tmp_match_result=np.sum((tmp_window_distance**2))
                elif matching_function=='SAD':
                    tmp_window_distance = pixel_window_ref-pixel_window_sec
                    tmp_match_result=np.sum(np.abs(tmp_window_distance))
                elif matching_function=='normalized_correlation':
                    ref_window_mean = np.mean(pixel_window_ref)
                    sec_window_mean = np.mean(pixel_window_sec)

                    ref_window_sub_mean = pixel_window_ref-ref_window_mean
                    sec_window_sub_mean = pixel_window_sec-sec_window_mean

                    tmp_match_result=np.sum(ref_window_sub_mean*sec_window_sub_mean)/(np.sqrt(np.sum(ref_window_sub_mean**2))*np.sqrt(np.sum(sec_window_sub_mean**2))+1e-10)
                else:
                    raise ValueError("Invalid matching_function")
                    

                if matching_function=='normalized_correlation':
                    if tmp_match_result>cur_match_result or d_==min_disparity:
                        cur_match_result=tmp_match_result
                        cur_disparity=d_
                else:
                    if tmp_match_result<cur_match_result or d_==min_disparity:
                        cur_match_result=tmp_match_result
                        cur_disparity=d_

            disparity_map[row_-padding_,col_-padding_]=cur_disparity
        # print(disparity_map[row_-padding_])
                    
    task1_time_end=time.time()
    output_str= f"Task1 time cost(window_size:{window_size},disparity_range:{disparity_range},matching:function:{matching_function}):{task1_time_end-task1_time_start}"
    print(output_str)
    with open("log.txt",'a',encoding='utf-8') as write_file:
        write_file.write(output_str+'\n')
    return disparity_map

def task1_simple_disparity(ref_img, sec_img, gt_map, img_name='tsukuba'):
    '''Compute disparity maps for different settings
    '''
    # window_sizes = [2,3,4,5,6]  # Try different window sizes
    # disparity_range = (0, 20)  # Determine appropriate disparity range
    # matching_functions = ['SSD', 'SAD']#, 'normalized_correlation']  # Try different matching functions
    
    window_sizes = [15]#[13]#[2,5,7,11,13,15,20]  # Try different window sizes
    disparity_ranges = [(0,15)]#[(0,20),(0,30),(0,40),(0,50),(0,100),(0,150)]#[(0,5),(0, 10),(0,15),(0,20),(0,30)]#[(0,15),(0,40)]#(0,20),(0,50),(0,100),(0,150),(0,200)]  # Determine appropriate disparity range
    matching_functions = ['SSD']#,'SAD', 'normalized_correlation']  # Try different matching functions
    
    disparity_maps = []
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            for disparity_range in disparity_ranges:
                print(f"Computing disparity map for window_size={window_size}, disparity_range={disparity_range}, matching_function={matching_function}")
                disparity_map = task1_compute_disparity_map_simple(
                    ref_img, sec_img, 
                    window_size, disparity_range, matching_function)
                disparity_maps.append((disparity_map, window_size, matching_function, disparity_range))
                dmin, dmax = disparity_range
                visualize_disparity_map(
                    disparity_map, gt_map, 
                    save_path=f"output/task1_{img_name}_{window_size}_{dmin}_{dmax}_{matching_function}.png")
    return disparity_maps


def task2_compute_depth_map(disparity_map, baseline, focal_length):
    '''Compute depth map by z = fB / (x - x')
    Note that a disparity less or equal to zero should be ignored (set to zero) 
    '''
    depth_map = np.zeros(np.shape(disparity_map))
    img_H,img_W = np.shape(disparity_map)
    for row_ in range(img_H):
        for col_ in range(img_W):
            if disparity_map[row_,col_]<=0:
                continue
            depth_map[row_,col_]=baseline*focal_length/disparity_map[row_,col_]

    return depth_map


def task2_visualize_pointcloud(
    ref_img: np.ndarray,        # shape (H, W, 3) 
    disparity_map: np.ndarray,  # shape (H, W)
    save_path: str = 'output/task2_tsukuba.ply'
):
    '''Visualize 3D pointcloud from disparity map following the steps:
    1. Calculate depth map from disparity
    2. Set pointcloud's XY as image's XY and and pointcloud's Z as depth
    3. Set pointcloud's color as ref_img's color
    4. Save pointcloud to ply files for visualizationh. We recommend to open ply file with MeshLab
    5. Adjust the baseline and focal_length for better performance
    6. You may need to cut some outliers for better performance
    '''
    baseline = 10
    focal_length = 10
    depth_map = task2_compute_depth_map(disparity_map, baseline, focal_length)

    new_depth_map=depth_map.copy()

    mean_depth = np.mean(depth_map)
    std_depth = np.std(depth_map)
    threshold_=1.3

    depth_distance_map = np.abs(depth_map-mean_depth)
    new_depth_map[depth_distance_map>threshold_*std_depth]=0
    # cut some outliers

    img_H,img_W,_ = np.shape(ref_img)
    # Points
    new_depth_map = (new_depth_map/np.max(new_depth_map))*150  #归一化并放缩


    points=[]
    colors=[]
    for i in range(img_H):
        for j in range(img_W):
            tmp_depth=new_depth_map[i,j]
            if tmp_depth==0:
                continue
            points.append([j,i,tmp_depth])
            colors.append(ref_img[i,j]/255.0)

    # Save pointcloud to ply file
    pointcloud = trimesh.PointCloud(points, colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pointcloud.export(save_path, file_type='ply')


def task3_compute_disparity_map_dp(ref_img, sec_img):
    ''' Conduct stereo matching with dynamic programming
    '''
    task3_time_begin = time.time()
    print("task3 start!")
    img_H,img_W =np.shape(ref_img)
    # print(f"ref_img,shape:H{img_H},W{img_W}")

    disparity_range = (0, 14)
    disparity_max = disparity_range[1]
    occlusionConstant=80.0
    window_size = 1
    padding_=window_size//2
    # calculate cost_matrix cost[i][j][k] means:
    # line i : ref_img[col_j] distance to sec_img[col_k]
    
    cal_ref_img =np.array(ref_img).astype(np.float32)
    cal_sec_img =np.array(sec_img).astype(np.float32)

    pad_ref_img = np.pad(ref_img,((padding_,padding_),(padding_,padding_)),'edge').astype(np.float64)
    pad_sec_img = np.pad(sec_img,((padding_,padding_),(padding_,padding_)),'edge').astype(np.float64)

    # cost_matrix_time = time.time()
    # print(f"cost_matrix over,cost time:{cost_matrix_time-task3_time_begin}")

    disparity_map_dp = np.zeros((img_H,img_W),dtype=np.float32)

    for row_ in range(img_H):
        # calculate disparity by row
        
        # 动态规划：从第一列开始走，只能往下，往右，往右下，mininize map_dp[img_W-1][img_W-1]
        # 恢复路径的方式是记录每个点的前继方式
        # map_dp[i][j]表示已经考虑过sec 前i+1个像素点 和 ref 前j+1个像素点 的匹配,记录最小的cost
        # print(np.shape(row_matrix))
        dp_H,dp_W = img_W,img_W
        dp_matrix = np.full((dp_W,dp_W),float('inf'),dtype=np.float32)
        dp_record_former_matrix = np.full((dp_W,dp_W),4.0,dtype=np.int32)

        dp_matrix[0,0]=min(np.sum(pad_ref_img[row_:row_+window_size,0:0+window_size]-pad_sec_img[row_:row_+window_size,0:0+window_size]**2),occlusionConstant) # initialize [0,0]
        dp_record_former_matrix[0,0]=-1

        # sec为右侧视角 正确的匹配sec在ref上的对应点应该在sec左侧
        for i in range(1,disparity_max):# initialize row0
            dp_matrix[0,i]=dp_matrix[0,i-1]+occlusionConstant # 横着走表示 ref[i]像素无法与sec[1]像素匹配,直接惩罚
            dp_record_former_matrix[0,i]=1

        for dp_row in range(1,dp_H): # 从上往下 从左往右dp
            for dp_col in range(dp_row,min(dp_row+disparity_max,dp_W)): # 只考虑disparity_range内的点，且对于sec[dp_row],目标ref只会出现在右侧
                # print(cal_ref_img[row_,dp_col],cal_sec_img[row_,dp_row])
                min1 = dp_matrix[dp_row-1,dp_col-1]+np.sum((pad_ref_img[row_:row_+window_size,dp_col:dp_col+window_size]-pad_sec_img[row_:row_+window_size,dp_row:dp_row+window_size])**2) # 从左上,正确匹配
                min2 = dp_matrix[dp_row,dp_col-1]+occlusionConstant # 从左 无法匹配 表示跳过ref[dp_col]
                min3 = dp_matrix[dp_row-1,dp_col]+occlusionConstant # 从上 无法匹配 表示跳过sec[dp_row]

                dp_min_lst =np.array([min1,min2,min3])
                dp_min_index = np.argmin(dp_min_lst)
                dp_matrix[dp_row,dp_col]=dp_min_lst[dp_min_index]
                dp_record_former_matrix[dp_row,dp_col]=dp_min_index
        
        # print(dp_matrix)
        # print(dp_record_former_matrix)
        # 恢复path
        cur_col = dp_W-1
        cur_row = dp_H-1

        dp_best_disparity=np.zeros((img_W))
        # print("begin recover best path!")
        while dp_record_former_matrix[cur_row,cur_col]!=-1:# when [0,0] break
            if dp_record_former_matrix[cur_row,cur_col]==0: #从左上
                dp_best_disparity[cur_col]=cur_col-cur_row
                cur_col-=1
                cur_row-=1   
            elif dp_record_former_matrix[cur_row,cur_col]==1: # 从左
                dp_best_disparity[cur_col]=-1
                cur_col-=1
            elif dp_record_former_matrix[cur_row,cur_col]==2: #从上
                cur_row-=1
            else:
                raise ValueError("Invalid dp_record_former_matrix value")
        
        for j in range(1,img_W):
            if dp_best_disparity[j]==-1:
                dp_best_disparity[j]=dp_best_disparity[j-1]
        disparity_map_dp[row_]=dp_best_disparity

        # print(np.shape(dp_best_path))
        # print(dp_best_disparity)

    # for i in range(img_H):
    #     print(disparity_map_dp[i])
    task3_time_end=time.time()
    print("task3 end!")
    print("task3 time cost:",task3_time_end-task3_time_begin)
    return disparity_map_dp

def main(tasks): 
    
    # Read images and ground truth disparity maps
    moebius_img1 = cv2.imread("data/moebius1.png")
    moebius_img1_gray = cv2.cvtColor(moebius_img1, cv2.COLOR_BGR2GRAY)
    moebius_img2 = cv2.imread("data/moebius2.png")
    moebius_img2_gray = cv2.cvtColor(moebius_img2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_img1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_img1_gray = cv2.cvtColor(tsukuba_img1, cv2.COLOR_BGR2GRAY)
    tsukuba_img2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_img2_gray = cv2.cvtColor(tsukuba_img2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Task 0: Visualize cv2 Results
    if '0' in tasks:   
        # Compute disparity maps using cv2
        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
        moebius_disparity_cv2 = stereo.compute(moebius_img1_gray, moebius_img2_gray)
        visualize_disparity_map(moebius_disparity_cv2, moebius_gt)
        tsukuba_disparity_cv2 = stereo.compute(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt)
        
        if '2' in tasks:
            print('Running task2 with cv2 results ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_cv2, save_path='output/task2_tsukuba_cv2.ply')

    ######################################################################
    # Note. Running on moebius may take a long time with your own code   #
    # In this homework, you are allowed only to deal with tsukuba images #
    ######################################################################

    # Task 1: Simple Disparity Algorithm
    if '1' in tasks:
        print('Running task1 ...')
        disparity_maps = task1_simple_disparity(tsukuba_img1_gray, tsukuba_img2_gray, tsukuba_gt, img_name='tsukuba')
        
        #####################################################
        # If you want to run on moebius images,             #
        # parallelizing with multiprocessing is recommended #
        #####################################################
        # task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task1 ...')
            for (disparity_map, window_size, matching_function, disparity_range) in disparity_maps:
                dmin, dmax = disparity_range
                task2_visualize_pointcloud(
                    tsukuba_img1, disparity_map, 
                    save_path=f'output/task2_tsukuba_{window_size}_{dmin}_{dmax}_{matching_function}.ply')      
        
    # Task 3: Non-local constraints
    if '3' in tasks:
        print('----------------- Task 3 -----------------')
        tsukuba_disparity_dp = task3_compute_disparity_map_dp(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_dp, tsukuba_gt, save_path='output/task3_tsukuba.png')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task3 ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_dp, save_path='output/task2_tsukuba_dp.ply')

if __name__ == '__main__':
    # Set tasks to run
    parser = argparse.ArgumentParser(description='Homework 4')
    parser.add_argument('--tasks', type=str, default='0123')
    args = parser.parse_args()

    main(args.tasks)
