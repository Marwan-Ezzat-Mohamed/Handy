import os
import shutil
from tqdm import tqdm
import numpy as np
from mediapipeHelper import *
import math
import json
import time


def make_test_val_data(limit=10,
                       min_videos_per_action=50):
    # count the number of videos that are greater than min_videos_per_action
    number_of_videos = [f for f in os.listdir(
        'MP_Data') if len(os.listdir(os.path.join('MP_Data', f))) >= min_videos_per_action]
    pbar = tqdm(total=min(len(number_of_videos), limit),
                desc='Generating test and val data')
    words_folder = os.listdir('MP_Data')
    # sort the words folder by the folder that has the most videos
    words_folder.sort(key=lambda x: len(os.listdir(
        os.path.join('MP_Data', x))), reverse=True)
    for word_folder in words_folder:
        if limit == 0:
            break
        action_path = os.path.join('MP_Data', word_folder)
        number_of_videos = len(os.listdir(action_path))
        if number_of_videos < min_videos_per_action:
            continue
        limit -= 1
        number_of_videos_for_test = max(int(number_of_videos * 0.1), 2)
        number_of_videos_for_train = number_of_videos - number_of_videos_for_test

        # val data is used for generating the new data (will be used for training)
        # copy videos to train folder
        for video in os.listdir(action_path)[:number_of_videos_for_train]:
            video_path = os.path.join(action_path, video)
            new_video_path = os.path.join('MP_Val', word_folder)
            os.makedirs(new_video_path, exist_ok=True)
            shutil.copy(video_path, new_video_path)

        # copy videos to test folder
        for video in os.listdir(action_path)[number_of_videos_for_train:]:
            video_path = os.path.join(action_path, video)
            new_video_path = os.path.join('MP_Test', word_folder)
            os.makedirs(new_video_path, exist_ok=True)
            shutil.copy(video_path, new_video_path)

        pbar.update(1)


def get_all_vids_paths(folder_name):
    # get all the paths of the videos (.mp4) in folder or subfolders
    vids_paths = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.mp4'):
                vids_paths.append(os.path.join(root, file))
    return vids_paths


if __name__ == '__main__':
    make_test_val_data()
    # videos = get_all_vids_paths('MSASL')
    # # calculate time taken to process all the videos
    # start = time.time()

    # remove_not_moving_frames(videos)
    # end = time.time()
    # print("Time taken to process all the videos: ", end - start)
