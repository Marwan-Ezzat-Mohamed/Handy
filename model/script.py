import os
import json
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import shutil
from tqdm import tqdm
from mediapipeHelper import *
import multiprocessing


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

    # if results.face_landmarks:
    #     for lm in results.face_landmarks.landmark:
    #         keypoints.append({
    #             "x": lm.x,
    #             "y": lm.y,
    #             "z": lm.z,
    #             "visibility": 1
    #         })
    #     keypoints_map['faceLandmarks'] = list(keypoints)
    #     keypoints.clear()
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
                    # draw_landmarks(image, results)
                    # cv2.imshow('OpenCV Feed', image)
                    # # sleep for 1 second
                    # cv2.waitKey(100)
                    action_keypoints.append(extract_keypoints_dict(results))
                else:
                    break
            cap.release()
            # show the video

            action = video_path.split(".")[0].split("\\")[-1]
            keypoints_map[action] = action_keypoints

    with open(os.path.join('keypoints_reshaped' + '.json'), 'w') as f:
        json.dump(keypoints_map, f)


if __name__ == '__main__':
    # videos = get_all_vids_paths('DATA')
    # for video in videos:
    #     with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    #         pool.map(convert_to_avcmp4, videos)

    #########################
    # datasets_path = 'DATASETS'
    unique_videos_path = "unique_videos"
    # # create a folder to store the unique videos
    # if not os.path.exists(unique_videos_path):
    #     os.mkdir(unique_videos_path)

    # # get all the videos paths
    # # DATASETS/DATASET/ACTION_NAME/VIDEO.mp4
    # videos_path = get_all_vids_paths(datasets_path)
    # all_actions_names = [video.split(os.sep)[-2] for video in videos_path]
    # all_actions = list(set(all_actions_names))
    # # copy the highest resolution video of each action to videos folder and rename it to the action name
    # # DATASETS/DATASET/ACTION/VIDEO.mp4
    # # videos/VIDEO.mp4
    # # videos/ACTION.mp4
    # for action in all_actions:
    #     print(action)
    #     action_videos = [video for video in videos_path if action in video]
    #     # get the highest resolution video
    #     highest_res_video = max(action_videos, key=os.path.getsize)
    #     # copy the video to the unique_videos folder
    #     shutil.copy(highest_res_video, unique_videos_path)
    #     # rename the video to the action name
    #     # highest_res_video DATASETS/DATASET/ACTION/VIDEO.mp4

    #     # rename the video to the action name in the unique_videos folder
    #     old_name = os.path.join(
    #         unique_videos_path, highest_res_video.split(os.sep)[-1])
    #     new_name = os.path.join(unique_videos_path, action + '.mp4')
    #     # print(old_name, new_name)
    #     os.rename(old_name, new_name)

    # # convert all videos to mp4 format regardless of their original format
    # # replace the original videos with the converted ones
    # videos_path = get_all_vids_paths(unique_videos_path)
    # pbar = tqdm(total=len(videos_path))
    # for video in videos_path:
    #     convert_to_avcmp4(video)
    #     pbar.update(1)

    # # DATASETS/DATASET/ACTION/VIDEO.mp4
    all_vids = get_all_vids_paths(unique_videos_path)
    # convert all videos to mp4 format regardless of their original format
    # replace the original videos with the converted ones
    # pbar = tqdm(total=len(all_vids))

    make_keypoints_json(all_vids)
