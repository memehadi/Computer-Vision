import math

import numpy as np
from PIL import Image
from skimage import color, io
from skimage.color import rgb2hsv


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out= io.imread(image_path)
     
    pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out =np.zeros((300, 300, 3))

    ### YOUR CODE HERE
    for i in range(300):
        for j in range(300):
            for k in range(3):
                square= image[i][j][k]* image[i][j][k]
                out[i][j][k]= 0.5*square
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    #out = ImageOps.grayscale(image)
    out= color.rgb2gray(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".
     Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
  

    ### YOUR CODE HERE
    out =np.zeros((300, 300, 3))
    if channel=="R":
        for i in range(300):
            for j in range(300):
                r,g,b = image[i,j]
                out[i,j] = 0,g,b
    elif channel=="G":
        for i in range(300):
            for j in range(300):
                r,g,b = image[i,j]
                out[i,j] = r,0,b
    elif channel=="B":
        for i in range(300):
            for j in range(300):
                r,g,b = image[i,j]
                out[i,j] = r,g,0
    
### END YOUR CODE
    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    #hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    hsv_img = rgb2hsv(image)
    if channel=="H":
        out = hsv_img[:,:,0]
    elif channel=="S":
        out = hsv_img[:,:,1]
    elif channel=="V":
        out = hsv_img[:,:,2]
    
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = np.zeros((300, 300, 3))
    ### YOUR CODE HERE
    # image11=np.zeros((300, 150, 3))
    # image22=np.zeros((300, 150, 3))
    # for i in range(300):
    #     for j in range(150):
    #         for k in range(3):
    #             image11[i][j][k]= image1[i][j][k]
    #             image22[i][j][k]= image2[i][j][k]

    # image11=rgb_exclusion(image1,channel1)
    # image22=rgb_exclusion(image2,channel2)

    #out= np.concatenate((image11, image22), axis=1)
    image1=rgb_exclusion(image1,channel1)
    image2=rgb_exclusion(image2,channel2)
    for i in range(300):
        for j in range(300):
            for k in range(3):
                if j<150:
                    out[i][j][k]= image1[i][j][k]
                   
                elif j>=150:
                    out[i][j][k]= image2[i][j][k]
                    

                
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
 
    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


