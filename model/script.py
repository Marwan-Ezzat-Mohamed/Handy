import os
import json
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from tqdm import tqdm
from mediapipeHelper import *


def convert_to_avcmp4(video_path):
    # Split the path and extension of the input video file
    input_path, input_ext = os.path.splitext(video_path)
    # Define the output path and extension for the converted mp4 file
    output_path = input_path + "_avc.mp4"
    # Load the input video file with moviepy
    clip = VideoFileClip(video_path)
    # Define the ffmpeg codec parameters for the output video
    codec_params = ['-vcodec', 'libx264', '-preset', 'medium', '-profile:v',
                    'main', '-tune', 'film', '-crf', '23', '-movflags', '+faststart']
    # Use moviepy to write the output video with the specified codec parameters
    clip.write_videofile(output_path, codec='libx264', preset='medium',
                         ffmpeg_params=codec_params)
    # Replace the original video file with the converted mp4 file
    os.replace(output_path, video_path)


def get_all_vids_paths(folder_name):
    # get all the paths of the videos (.mp4) in folder or subfolders
    vids_paths = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.mp4'):
                vids_paths.append(os.path.join(root, file))
    return vids_paths


def extract_keypoints_dict(results):
    keypoints_map = {
        "poseLandmarks": [],
        "leftHandLandmarks": [],
        "rightHandLandmarks": [],
        "faceLandmarks": []
    }
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": 1
            })
            # make it equal a copy of the keypoints list
        keypoints_map['poseLandmarks'] = list(keypoints)
        keypoints.clear()
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": 1
            })
            # make it equal a copy of the keypoints list
        keypoints_map['leftHandLandmarks'] = list(keypoints)
        keypoints.clear()

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": 1
            })
        # make it equal a copy of the keypoints list
        keypoints_map['rightHandLandmarks'] = list(keypoints)
        keypoints.clear()

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": 1
            })
        keypoints_map['faceLandmarks'] = list(keypoints)
        keypoints.clear()
    return keypoints_map


def make_keypoints_json(videos_paths):
    # get all the videos paths
    keypoints_map = {}
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for video in videos_paths:
            # get the first video of the action
            video_path = video
            cap = cv2.VideoCapture(video_path)
            action_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    cv2.imshow('OpenCV Feed', image)
                    # sleep for 1 second
                    cv2.waitKey(100)
                    action_keypoints.append(extract_keypoints_dict(results))
                else:
                    break
            cap.release()
            # show the video

            action = video_path.split(".")[0].split("\\")[-1]
            keypoints_map[action] = action_keypoints

    with open(os.path.join('keypoints_reshaped' + '.json'), 'w') as f:
        json.dump(keypoints_map, f)


# DATASETS/DATASET/ACTION/VIDEO.mp4
all_vids = get_all_vids_paths('C:\\Users\\Marwan\\Downloads\\videos')
# convert all videos to mp4 format regardless of their original format
# replace the original videos with the converted ones
pbar = tqdm(total=len(all_vids))
limit = 2
make_keypoints_json = make_keypoints_json(all_vids[:limit])
