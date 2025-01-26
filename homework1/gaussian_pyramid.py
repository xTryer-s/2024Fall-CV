from PIL import Image
import numpy as np

def cross_correlation_2d(img_array,kernel_array):
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    kernel_height,kernel_width = kernel_array.shape
    output_img_array = np.zeros_like(img_array,dtype=np.float32)
    k_h = kernel_height//2
    k_w = kernel_width//2

    for i in range(0,img_height):
        for j in range(0,img_width):
            for k_i in range(-k_h,k_h+1):
                for k_j in range(-k_w,k_w+1):
                    cur_i = i+k_i
                    cur_j = j+k_j
                    if 0 <= cur_i < img_height and 0 <= cur_j < img_width:
                        output_img_array[i][j] += kernel_array[k_h+k_i][k_w+k_j]*img_array[cur_i][cur_j]
    return output_img_array


def convolve_2d(img_array,kernel_array):
    return cross_correlation_2d(img_array,np.flip(kernel_array))

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

def gaussian_blur_2d(img_array,k_size):
    return convolve_2d(img_array,gaussian_blur_kernel_2d(k_size))

def low_pass(img_array,k_size,k_sigma=1): 
    gauss_kernel = gaussian_blur_kernel_2d(k_size,k_sigma)
    return convolve_2d(img_array,gauss_kernel)

def image_subsampling(img_array):
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    img_channels_num = img_array.shape[2]

    tar_height = img_height//2
    tar_width = img_width//2
    ret_img = np.zeros((tar_height,tar_width,img_channels_num))

    for i in range(0,tar_height):
        for j in range(0,tar_width):
            ret_img[i,j]=img_array[i*2,j*2]

    return ret_img

def gaussian_pyramid(img_array,pyramid_level):
    pyramid =[img_array]
    cur_img = img_array
    for i in range(0,pyramid_level):
        cur_img = low_pass(cur_img,3)
        cur_img = image_subsampling(cur_img)
        pyramid.append(cur_img)
    
    return pyramid


def main():
    print("my cv work") # 取消注释即可运行各部分代码。

    
    # cross_correlation
    # img_read = Image.open('Lena.png')
    # img_pixel_array = np.array(img_read)
    # kernel1 = np.array([[1/9,1/9,1/9],
    #                     [1/9,1/9,1/9],
    #                     [1/9,1/9,1/9]])
    # cross_correlation_img_array = cross_correlation_2d(img_pixel_array,kernel1)
    # cross_correlation_img = Image.fromarray(np.uint8(cross_correlation_img_array))
    # cross_correlation_img.save('lena_cross_correlation.png')


    # cross_convolve
    # img_read = Image.open('Lena.png')
    # kernel1 = np.array([[1/9,1/9,1/9],
    #                     [1/9,1/9,1/9],
    #                     [1/9,1/9,1/9]])
    # img_pixel_array = np.array(img_read)
    # cross_convolve_img_array = cross_correlation_2d(img_pixel_array,kernel1)
    # cross_convolve_img = Image.fromarray(np.uint8(cross_convolve_img_array))
    # cross_convolve_img.save('lena_convolve.png')


    # Low Pass Filter
    # img_read = Image.open('Lena.png')
    # img_pixel_array = np.array(img_read)
    # low_pass_img_array = low_pass(img_pixel_array,3)
    # low_pass_img = Image.fromarray(np.uint8(low_pass_img_array))
    # low_pass_img.save('lena_lowpass.png')


    # subsampling
    # img_read = Image.open('Vangogh.png')
    # img_pixel_array = np.array(img_read)
    # subsampling_img_array=image_subsampling(img_pixel_array)
    # subsampling_img = Image.fromarray(np.uint8(subsampling_img_array))
    # subsampling_img.save('Vangogh_downsample.png')

    # Gaussian Pyramid
    img_read = Image.open('Vangogh.png')
    img_pixel_array = np.array(img_read)
    pyramid_level = 3
    gaussian_pyramid_img_array = gaussian_pyramid(img_pixel_array,pyramid_level)
    for i in range(0,pyramid_level+1):
        pyramid_img_ =Image.fromarray(np.uint8(gaussian_pyramid_img_array[i]))
        pyramid_img_.save(f'Vangogh_pyramid_{pow(2,i)}.png')

if __name__=="__main__":
    main()