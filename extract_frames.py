import re
from datetime import timedelta
import cv2
import numpy as np
import os
import retina
import sys
import time
import multiprocessing
import aws_client


UNMASKED_FRAMES_DIR = "frames_from_video"
SAVING_FRAMES_PER_SECOND = 30


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def extract_frames_from_video(video_url):
    # make a folder by the name of the video file

    # read the video file
    cap = cv2.VideoCapture(video_url)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration,
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            file_name = f"frame{frame_duration_formatted}.jpg"
            s3_path = "unmasked_frames" + "/" + file_name  # path is the folder path within the S3 bucket where you want to store the frames
            print(file_name, file_name, "\n")
            try:
                response = aws_client.upload_unmasked_frame(frame, s3_path)
            except FileNotFoundError:
                print("The file was not found")
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1
    return saving_frames_per_second



def frame_sort_key(frame):
    match = re.match(r"frame0-(\d+)-(\d+)\.(\d+).jpg", frame)
    if match:
        hour = match.group(1).zfill(2)
        minute = match.group(2).zfill(2)
        second = match.group(3).zfill(3)
        return hour, minute, second
    return frame


def sorted_frames_files(bucket_name, prefix):
    s3 = aws_client.s3_client
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.jpg'):
            files.append(obj['Key'])
    sorted_files = sorted(files, key=frame_sort_key)
    return sorted_files


def masked_frame_group(frames_files, kernel, epsilon):
    for frame in frames_files:
        faces_locations = retina.all_faces_locations(frame)
        retina.update_parameters(frame, (kernel, kernel), epsilon, faces_locations)

