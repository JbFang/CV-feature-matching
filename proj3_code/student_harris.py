import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb

def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################
    
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.dot(kernel, kernel.T)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################
    
    widthx, widthy = np.shape(filt)
    width_top = int(widthx / 2)
    width_bot = widthx - width_top - 1
    width_left = int(widthy / 2)
    width_right = widthy - width_left - 1

    image = cv2.copyMakeBorder(image, width_top, width_bot, width_left,\
                               width_right, cv2.BORDER_CONSTANT)
    m, n = np.shape(image)

    conv_image = np.zeros(np.shape(image))
    
    for imgx in range(width_left, m - width_right):
        for imgy in range(width_top, n - width_bot):
            F = image[(imgx - width_left):(imgx + width_right + 1),\
                      (imgy - width_top):(imgy + width_bot + 1)]
            conv_image[imgx, imgy] = np.sum(F[::-1, ::-1] * filt)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image[width_left:m-width_right,width_top:n-width_bot]

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    sobelx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobely = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    ix = my_filter2D(image, sobelx)
    iy = my_filter2D(image, sobely)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing). treat the center point of this window as the bottom right
        of the center most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   c: numpy nd-array of dim (N,) containing the confidences of each pixel
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    m, n = np.shape(image)
    
    width_top = int(window_size / 2)
    width_bot = window_size - width_top - 1
    width_left = int(window_size / 2)
    width_right = window_size - width_left - 1

    xx = np.array([])
    yy = np.array([])
    cc = np.array([])

    for i in range(len(x)):
        if x[i] >= width_left and y[i] >= width_top\
           and x[i] < n - width_right and y[i] < m - width_bot:
            xx = np.append(xx, x[i])
            yy = np.append(yy, y[i])
            cc = np.append(cc, c[i])
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return xx, yy, cc

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    filt = get_gaussian_kernel(ksize, sigma)

    sx2 = my_filter2D(ix*ix, filt)
    sy2 = my_filter2D(iy*iy, filt)
    sxsy = my_filter2D(ix*iy, filt)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha = 0.05):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    m, n = np.shape(sx2)
    M = np.zeros((m,n,2,2))
    M2 = np.array([[sx2, sxsy], [sxsy, sy2]])

    for i in range(m):
        for j in range(n):
            M[i,j,0,0] = sx2[i,j]
            M[i,j,0,1] = sxsy[i,j]
            M[i,j,1,0] = sxsy[i,j]
            M[i,j,1,1] = sy2[i,j]

    R = np.linalg.det(M) - alpha * np.trace(M2)**2
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    filt = maximum_filter(R, size = neighborhood_size)

    R_local_pts = np.where(R == filt, R, 0)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None
    

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################
    
    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy)
    R_local_pts = non_max_suppression(R)
    x, y = np.unravel_index(np.argsort((-R_local_pts).ravel())[:n_pts],\
                             np.shape(R_local_pts))
    
    confidences = np.zeros((len(x)))
    for i in range(len(x)):
        confidences[i] = R_local_pts[x[i], y[i]]

    x, y, confidences = remove_border_vals(image, x, y, confidences, window_size = 16)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return y,x, R_local_pts, confidences


