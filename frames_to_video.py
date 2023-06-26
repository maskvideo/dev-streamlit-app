import cv2
import os

import extract_frames
import retina

# Path to the directory containing the frames
frame_folder = retina.MASKED_FRAMES_DIR

# Get all the file names in the directory
frame_files = extract_frames.sorted_frames_files(frame_folder)

# Get the dimensions of the frames
frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
height, width, layers = frame.shape

# Create a VideoWriter object to write the frames to a video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('blurred_result.mp4', fourcc, extract_frames.SAVING_FRAMES_PER_SECOND, (width,height))

# Loop through all the frames and add them to the video
for filename in frame_files:
    img = cv2.imread(os.path.join(frame_folder, filename))
    video.write(img)

# Release the VideoWriter and close all windows
video.release()
cv2.destroyAllWindows()