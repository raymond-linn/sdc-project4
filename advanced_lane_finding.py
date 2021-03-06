import numpy as np
import cv2
import pickle
import os
import p4Utilities as p4Util
from moviepy.video.io.VideoFileClip import VideoFileClip


pickle_file_path = 'camera_cal/dist_pickle.p'
camera_images_dir = 'camera_cal'

# define the pipeline using the utility functions


# 2. undistort the image frame using calibrated data
# 3. apply combined binary thresholding (with color thresholding)
# 4. transofrm perspective
# 5. fit polynomial lane line
# 6. calculate cuvature radius
# 7. plot back to image frame
# 8. output video frames
# 9. discussion
def main():

	# 1. read in video frame
	video_in = 'project_video.mp4'
	video_out = 'project_video_out.mp4'

	# 2. Set up the fl_image
	clip = VideoFileClip(video_in)
	video_clip = clip.fl_image(pipeline)

	# 3. Output the video frames
	video_clip.write_videofile(video_out, audio=False)

'''
# test function for notebook without the p4Util import
def pipeline(image):
	# 1) undistort the frame image
	# 1.1) read in the calculated coefficients from pickle file
	if os.path.isfile(pickle_file):
		with open('camera_cal/dist_pickle.p', 'rb') as pkf:
			data = pickle.load(pkf)
			mtx = data["mtx"]
			dist = data["dist"]
			img_size = data["img_size"]
	else:
		# calibrate the camera using given chessboard images, store in 'dist_pickle.p'
		mtx, dist, img_size = p4Util.calibrate_camera('camera_cal', 'camera_cal/dist_pickle.p')

	# print(img_size)
	# undistort the image
	img = cv2.undistort(image, mtx, dist, None, mtx)
	# 2) apply threshold
	img = combined_threshold(img)
	# 3) perspective transform and warp image
	warped, M = warp_image(img)
	# to get inverse perspective transform to use to drawback on original image
	unwarped, Minv = unwarp_image(img)
	# 4) poly fit for left lane and right lane
	left_fit, right_fit = fit_polynomial(warped)    
	# 5) draw left and right with polygon on the original image
	new_img = draw_polygon(image, left_fit, right_fit, Minv)

	# return the new image 
	return new_img
'''

def pipeline(image):

	# 1) undistort the frame image
	# 1.1) read in the calculated coefficients from pickle file
	if os.path.isfile(pickle_file_path):
		with open(pickle_file_path, 'rb') as pkf:
			data = pickle.load(pkf)
			mtx = data["mtx"]
			dist = data["dist"]
			img_size = data["img_size"]
	else:
		# calibrate the camera using given chessboard images, store in 'dist_pickle.p'
		mtx, dist, img_size = p4Util.calibrate_camera(camera_images_dir, pickle_file_path)

	# print(img_size)
	# undistort the image
	img = cv2.undistort(image, mtx, dist, None, mtx)
	# 2) apply threshold
	img = p4Util.combined_threshold(img)
	# 3) perspective transform and warp image
	src = np.float32(
            [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
            [((img_size[0] / 6)  + 47), img_size[1] - 40 ],
            [(img_size[0] * 5 / 6) - 26, img_size[1] - 40],
            [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    dst = np.float32(
            [[(img_size[0] / 4) - 60, 0],
            [(img_size[0] / 4) - 60, img_size[1]],
            [(img_size[0] * 3 / 4) + 80, img_size[1]],
            [(img_size[0] * 3 / 4) + 80, 0]])
	warped, M = p4Util.warp_image(img, src, dst)
	# to get inverse perspective transform to use to drawback on original image
	unwarped, Minv = p4Util.unwarp_image(img, dst, src)
	# 4) poly fit for left lane and right lane
	left_fit, right_fit = p4Util.fit_polynomial(warped)    
	# 5) draw left and right with polygon on the original image
	new_img = p4Util.draw_polygon(image, left_fit, right_fit, Minv)

	# 6) calculate curvature radius and vehcile position then draw on the image
    left_curve, right_curve, vp_from_center = p4Util.calulate_curvature(new_img, left_fit, right_fit)
    
    # 7) draw the text on the image for left and right curve in meter
    cv2.putText(new_img, "Left curve: {}m".format(left_curve.round()) + " and " + "Right curve: {}m".format(right_curve.round()), 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)
    
    # 8) prepare the vp text and draw on the image
    if vp_from_center > 0:
        vp_text = '{:.2f}m right from the center'.format(vp_from_center)
    else:
        vp_text = '{:.2f}m left from the center'.format(vp_from_center)
        
    cv2.putText(new_img, "VP is {}".format(vp_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)


	# return the new image 
	return new_img


if __name__ == '__main__':
	# maybe use argument to run this for different videos-TODO
	main()

