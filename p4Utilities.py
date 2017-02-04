import numpy as np
import cv2
import glob
import os
import pickle

# Calibrating Cmaera
# parameters: image_path - where images used to calibrate camera# pickle_file - saved calibrated data
# pickle_file - saved calibrated data file 
# return: mtx, dist, img_size
def calibrate_camera(image_path, pickle_file):
	# check if pickle file is not valid file then calibrate the camera
	# and save the calibrated data in pickle_file
	if not os.path.isfile(pickle_file):
		print('picke file does not exist')
		# initialize objpoints (3D real world space) and imgpoints (2D image plane)
		objpoints = []
		imgpoints = []

		# check dimension of chessboard images to initialize individual object point
		# the camera images in camera_cal folder does not seems to capture 
		# constant chessboard dimension and I am not getting treturn true from 
		# find chessboard corenr from every image so I have to change the dimension
		# of chessboard images to find the corners [(9,6), (8,6), (9,5), (9,4) (7,6), (5,6)]

		objp1 = np.zeros((9*6,3), np.float32)
		objp1[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
		objp2 = np.zeros((8*6,3), np.float32)
		objp2[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
		objp3 = np.zeros((9*5,3), np.float32)
		objp3[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)
		objp4 = np.zeros((9*4,3), np.float32)
		objp4[:,:2] = np.mgrid[0:9, 0:4].T.reshape(-1,2)
		objp5 = np.zeros((7*6,3), np.float32)
		objp5[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
		objp6 = np.zeros((5*6,3), np.float32)
		objp6[:,:2] = np.mgrid[0:5, 0:6].T.reshape(-1,2)

		images = glob.glob(image_path+'/calibration*.jpg')

		for idx, fname in enumerate(images):
			# read in image file
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# find the chessboard corners through all those dimensions
			ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
			objp = objp1
			if not ret:
				ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
				objp = objp2
			if not ret:
				ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
				objp = objp3
			if not ret:
				ret, corners = cv2.findChessboardCorners(gray, (9,4), None)
				objp = objp4
			if not ret:
				ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
				objp = objp5
			if not ret:
				ret, corners = cv2.findChessboardCorners(gray, (5,6), None)
				objp = objp6

			# If found add object points and image points
			if ret == True:
				# print('found object points and image points')
				objpoints.append(objp)
				imgpoints.append(corners)
				img_size = (img.shape[1], img.shape[0])

				# cv2.drawChessboardCorners(img, (corners.shape[1],corners.shape[0]), corners, ret)

				# calibrate camera
				ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
																	imgpoints, 
																	img_size, 
																	None, 
																	None)

		# save the data in pickel file for later use (we don't worry about rvecs and tvecs)
		dist_pickle = {}
		dist_pickle["img_size"] = img_size
		dist_pickle["mtx"] = mtx
		dist_pickle["dist"] = dist
		pickle.dump(dist_pickle, open(pickle_file, "wb"))
		# print('saved distortion data to pickled file')

	else:
		# use the saved data 
		print('picke file exists')
		with open(pickle_file, 'rb') as pkf:
			data = pickle.load(pkf)
			img_size = data["img_size"]
			mtx = data["mtx"]
			dist = data["dist"]
			# print('loaded pickle distortion file')

	return mtx, dist, img_size


# undistort the camera image
# parameters: img - images to undistort
# mtx, dist - calibrated data returned from calibrate_camera()
# return: undistorted image
def undistort_image(img, mtx, dist):
	img = cv2.imread(img)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst


# Gradient Orientation Threshold using Sobel Operators
# parameters: image, gradient orientation, kernel size and threshold min/max
# return: binary_output of the gradient oreintation image
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
	    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
	    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	# binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	# using cv2.threshold function
	ret, binary_output = cv2.threshold(scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)

	# Return the result
	return binary_output

# Magnitude of gradient Threshold 
# parameters: image, kernel size and magnitude threshold min/max
# return: binary_output of the magnitude gradient image
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
	
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(img[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(img[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
	# Return the binary image
	return binary_output

# Directional gradient Threshold 
# parameters: image, kernel size and directional threshold range
# return: binary_output of the directional gradient image
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(img[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(img[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output

# S-channel thresholds of HLS
# parameters: image, and threshold min/max
def hls_select(img, thresh=(0, 255)):

	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

	return binary_output

# hsv yelloe and white thrshold
# parameter: image 
def hsv_threshold(img):
	# convet to hsv space
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	# yellow mask
	yellow_min = np.array([15, 100, 120], np.uint8)
	yellow_max = np.array([80, 255, 255], np.uint8)
	yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

	# white mask
	white_min = np.array([0, 0, 200], np.uint8)
	white_max = np.array([255, 30, 255], np.uint8)
	white_mask = cv2.inRange(img, white_min, white_max)

	binary_output = np.zeros_like(img[:, :, 0])
	binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

	# filtered = img
	# filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

	return binary_output

# combining all the thresholding binary 
# parameter: img 
def combined_threshold(img):

	ksize = 15
	# directional gradient
	dir_binary = dir_threshold(img, ksize, thresh=(0.7, 1.2))
	# magnitude gradient
	mag_binary = mag_thresh(img, ksize, mag_thresh=(50, 255))
	# s channel binary
	hls_binary = hls_select(img, (90,255))

	hsv_binary = hsv_threshold(img)

	# combine together
	combined = np.zeros_like(dir_binary)
	# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
	# combined[((gradx > 0) & (grady > 0)) | ((mag_binary > 0) & (dir_binary > 0)) | (hls_binary > 0)] = 1
	# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	# combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	combined[((hsv_binary == 1) & ((mag_binary == 1) | (dir_binary == 1)))] = 1

	return combined

# compute perspective transform and warp image
# parameters: img - undistorted image
# src - four source points
# dst - foour dst points
def warp_image(img, src=SOURCE, dst=DESTINATION):
	# need to convert gray scale?
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

	return warped, M

# compute inverse perspective transform and unwarp image
# parameters: img - undistorted image
# src - four source points
# dst - foour dst points 
def unwarp_image(img, dst=DESTINATION, src=SOURCE):
	# need to convert gray scale?
	Minv = cv2.getPerspectiveTransform(dst, src)
	unwarped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

	return unwarped, Minv


# fitting the left and right lane lines using histogram and polyfit sliding window methods
# support code is ported from udacity lecture
# parameters: img - warped image
# return: left_fit, right_fit - second order polynomial
def fit_polynomial(img):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
	
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((img, img, img))*255
	
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(img.shape[0]/nwindows)

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = img.shape[0] - (window+1)*window_height
	    win_y_high = img.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
	    	& (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
	    	& (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	        
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return left_fit, right_fit

# Drawing the polynomial on the original image
# parameters: img - image to be drawn on
# left_fit, right_fit - generated by the polynomial fit 
# Minv - inverse perspective transform 
def draw_polygon(img, left_fit, right_fit, Minv):
    blank = np.zeros_like(img).astype(np.uint8)

    fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_pts = np.array([np.transpose(np.vstack([left_fitx, fity]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
    # stack left_pts and right_pts to be a single array
    pts = np.hstack((left_pts, right_pts))
    # saved in np array
    pts = np.array(pts, dtype=np.int32)
    # draw the plygon
    cv2.fillPoly(blank, pts, (0, 255, 0))

    # using inverse perspective matrix (Minv)
    unwapred = cv2.warpPerspective(blank, Minv, (img.shape[1], img.shape[0]))
    # Combine with the original image
    new_img = cv2.addWeighted(img, 1, unwapred, 0.3, 0)

    return new_img

# Calculating curvature of the lane line
def calulate_curvature(image, left_fit, right_fit):
    # calculating curvature radius in pixel and meter
    ploty = np.linspace(0, 719, num=720)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # print(y_eval)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    # print((ploty*ym_per_pix).shape)
    # print (left_fit)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # calculating the vehicle position w.r.t the lane lines
    center_pixel = image.shape[1]/2
    # assuming camera is mounted at the center of the vehicle
    vp_center = int((left_fitx[0]+right_fitx[0])/2)
    # print(vp_center)
    from_center = center_pixel - vp_center
    # print(from_center)
    vp_center_in_meter = xm_per_pix * from_center
    # print(vp_center_in_meter,'m')

    return left_curverad, right_curverad, vp_center_in_meter

    