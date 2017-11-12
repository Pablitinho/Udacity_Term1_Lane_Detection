#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio as imio
# --------------------------------------------------------------------------------
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------------------------------------
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
# --------------------------------------------------------------------------------
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
# --------------------------------------------------------------------------------
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
# --------------------------------------------------------------------------------
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
# --------------------------------------------------------------------------------
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines
# --------------------------------------------------------------------------------
def get_scores(image, lines, num_lines,lines_equations,min_y):

    score = np.zeros(len(lines))
    new_coordinates = np.zeros(2)

    for i in range(num_lines):
        equation = lines_equations[i]
        score[i] = 0
        if equation[0] != 0 and math.isinf(equation[0]) == False and math.isinf(equation[1]) == False:
            for y in range(min_y, image.shape[1], 1):

                if equation[0] == 0:
                    new_coordinates[0] = equation[1]
                else:
                    new_coordinates[0] = int(round((y - equation[1]) / (equation[0]+0.0001)))
                new_coordinates[1] = int(round(y))
                for x in range(-5, 5, 1):
                    y_ = int(new_coordinates[1])
                    x_ = int(new_coordinates[0] + x)
                    if x_ > 0 and x_< image.shape[1] and y_ > 0 and y_< image.shape[0]:
                        if image[y_, x_] > 0:
                            score[i] = score[i] + 1

    return score
# --------------------------------------------------------------------------------
def get_line_equation(line):

    line_equation = np.zeros(2)

    xA = line[0, 0]
    yA = line[0, 1]
    xB = line[0, 2]
    yB = line[0, 3]
    line_equation[0] = (yA - yB)/(xA - xB) # Slope
    if line_equation[0] == 0:
        pp = 0


    line_equation[1] = yA - line_equation[0]*xA #Bias
    return line_equation
# --------------------------------------------------------------------------------
def new_coordinate_line(image,line_equation,min_y):

    new_coordinates = np.zeros(4)

    if line_equation[0] == 0:
        new_coordinates[0] = line_equation[1]
    else:
        new_coordinates[0] = (min_y - line_equation[1]) / (line_equation[0]+0.0001)

    new_coordinates[1] = min_y

    if line_equation[0] == 0:
        new_coordinates[0] = line_equation[1]
    else:
        new_coordinates[2] = (image.shape[1] - line_equation[1]) / (line_equation[0]+0.0001)
    new_coordinates[3] = image.shape[1]

    return new_coordinates
# --------------------------------------------------------------------------------
def get_best_line_score(lines,num_lines,score):

    best_score_idx = -1
    best_score = -1

    for i in range(num_lines):
        if score[i]>best_score:
            best_score = score[i]
            best_score_idx = i

    return lines[best_score_idx], best_score_idx
# --------------------------------------------------------------------------------
def detect_left_right_line(img,lines):

    min_y = int(img.shape[1]/3)
    l_r_lines = np.zeros([1,2, 4])
    l_r_lines = l_r_lines.astype(int)

    left_score = np.zeros(len(lines))
    right_score = np.zeros(len(lines))
    left_candidates = np.copy(lines)*0
    num_left_candidates = 0
    right_candidates = np.copy(lines)*0
    num_right_candidates = 0

    left_line_equation = np.zeros([len(lines), 2])
    right_line_equation = np.zeros([len(lines), 2])

    # Get the center of the image, in the proper way it should be the Y vehicle coordinates respect to the camera
    center_x = (img.shape[1])/2

    for i in range(len(lines)):
        line = lines[i]

        x_coordinate = 0

        if line[0, 1] > line[0, 3]:
            x_coordinate = line[0, 0]
        else:
            x_coordinate = line[0, 2]

        if x_coordinate >= center_x:
            line_equation = get_line_equation(line)
            right_line_equation[num_right_candidates] = line_equation

            right_candidates[num_right_candidates] = lines[i]
            #

            num_right_candidates = num_right_candidates+1
        else:
            line_equation = get_line_equation(line)
            left_line_equation[num_left_candidates] = line_equation

            left_candidates[num_left_candidates] = lines[i]

            #left_candidates[num_left_candidates] = new_coordinate_line(img, line_equation, min_y)

            num_left_candidates = num_left_candidates + 1

    left_score = get_scores(img, left_candidates, num_left_candidates-1, left_line_equation, min_y)
    right_score = get_scores(img, right_candidates, num_right_candidates-1, right_line_equation, min_y)

    l_r_lines[0, 0], best_score_left = get_best_line_score(left_candidates, num_left_candidates-1, left_score)
    l_r_lines[0, 1], best_score_right = get_best_line_score(right_candidates, num_right_candidates-1, right_score)

    l_r_lines[0, 0] = new_coordinate_line(img, left_line_equation[best_score_left], min_y)
    l_r_lines[0, 1] = new_coordinate_line(img, right_line_equation[best_score_right], min_y)

    return l_r_lines
# --------------------------------------------------------------------------------
def hough_lines_road(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

    l_r_lines = detect_left_right_line(img, lines)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, l_r_lines)
    return line_img, l_r_lines
# --------------------------------------------------------------------------------
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def detect_lines_on_road(image,kernel_size = 5, low_threshold = 50, high_threshold = 150, rho = 2, threshold = 40, min_line_length = 50, max_line_gap = 30):

    # Convert to Gray scale
    gray = grayscale(image)
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold, high_threshold)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on

    theta = (np.pi / 180)  # angular resolution in radians of the Hough grid
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    hough_image, lines = hough_lines_road(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw the lines on the edge image
    lines_edges = weighted_img(hough_image, image, 0.9, 1., 0.)

    return lines_edges
# --------------------------------------------------------------------------------
# Read the images in the folder test_image.
image_list = os.listdir("test_images/")
# --------------------------------------------------------------------------------
# Set the parameters to detect the lines.
# --------------------------------------------------------------------------------
kernel_size = 5 #Gaussina kernel size
low_threshold = 50 # low thr in the canny method
high_threshold = 150 # high thr in canny method
rho = 2  # distance resolution in pixels of the Hough grid
threshold = 40  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Loop to process all the images.
# --------------------------------------------------------------------------------
for i in range(len(image_list)):

    print('Image:', image_list[i])  # reading in an image
    image = mpimg.imread('test_images/' + image_list[i])
    image_lines = detect_lines_on_road(image, kernel_size, low_threshold, high_threshold, rho, threshold,
                                       min_line_length, max_line_gap)

    cv2.imshow('Lane Detection', image_lines)
    cv2.waitKey(0)

video_list = os.listdir("test_videos/")
for i in range(len(video_list)):

    imio.plugins.ffmpeg.download()
    white_output = 'test_videos_output/' + video_list[i]
    clip1 = VideoFileClip('test_videos/' + video_list[i])
    white_clip = clip1.fl_image(detect_lines_on_road) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    HTML("""
    <video width="960" height="540" controls>
    <source src="{0}">
    </video>
    """.format(white_output))