from itertools import repeat
import math
import multiprocessing
import os
import shutil
from tqdm import tqdm
import numpy as np
from mediapipeHelper import *
import cv2
import time
import json


def adjust_video_total_frames_using_interpolation(video_path, output_path, num_frames):

    # read the video
    cap = cv2.VideoCapture(video_path)
    # get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # calculate the ratio
    ratio = total_frames / float(num_frames)
    # create a new video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, num_frames, (width, height))
    # loop through the frames
    for i in range(num_frames):
        # get the frame number
        frame_number = int(i * ratio)
        if frame_number >= total_frames:
            frame_number = total_frames - 1
        if frame_number < 0:
            frame_number = 0
        # set the frame number

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # read the frame
        ret, frame = cap.read()
        retry_count = 0
        while not ret:
            print("Can't receive frame. Retrying ...")
            cap.release()
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            retry_count += 1
            if retry_count > 10:
                break
        # write the frame
        out.write(frame)
    # release the video
    cap.release()
    # release the video writer
    out.release()
    # close all windows
    cv2.destroyAllWindows()


def save_video_features(video_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # get the number of frames in the video
        cap = cv2.VideoCapture(os.path.join(video_path))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Loop through frames
        to_save = []
        for frame_num in range(length):
            # Read feed
            success, image = cap.read()
            # Make detections
            if image is None:
                continue
            image, results = mediapipe_detection(image, holistic)
            # Export keypoints
            keypoints = extract_keypoints(results)
            to_save.append(keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # output path will be the same as input but with changing first folder to MP_DATA_NEW and adding .npy
        # split using the os separator
        output_path = video_path.split(os.sep)
        output_path[0] = 'MP_DATA_NEW'
        output_path[-1] = output_path[-1].split('.')[0] + '.npy'
        output_path = '/'.join(output_path)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, to_save)
        cap.release()
        cv2.destroyAllWindows()


def save_videos_features(videos_paths):
    print('Number of cores:', multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        with tqdm(total=len(videos_paths), desc='Saving videos features') as pbar:
            for _ in p.imap_unordered(save_video_features, videos_paths):
                pbar.update()


def random_rotation(frame):
    """
    Applies a random rotation to a video frame with a maximum rotation angle of 20 degrees left or right.
    """
    # Generate a random angle between -20 and 20 degrees
    max_angle = 5
    angle = np.random.randint(-max_angle, max_angle)

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    # Apply the rotation to the frame
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

    return rotated_frame


def rotate_video(input_path):
    # output_path save as input_path but adding _rotated to the name and the angle
    output_path = input_path[:-4] + '_rotated_' + '.mp4'

    # Open input video file
    cap = cv2.VideoCapture(input_path)

    # Get the codec information of input video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Get the frame rate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the dimensions of input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to write the rotated video
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    while cap.isOpened():
        # Read a frame from input video
        ret, frame = cap.read()

        if ret:
            # Rotate the frame
            rotated_frame = random_rotation(frame)
            # Write the rotated frame to output video
            out.write(rotated_frame)

        else:
            break

    # Release input and output video
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()


def resize_video(input_path,  multiplier=2.0):
    # output_path save as input_path but adding _resized to the name and the multiplier
    output_path = input_path[:-4] + '_resized_' + str(multiplier) + 'x.mp4'

    # Open input video file
    cap = cv2.VideoCapture(input_path)

    # Get the codec information of input video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Get the frame rate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the dimensions of input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = math.ceil(frame_width * multiplier)
    height = math.ceil(frame_height * multiplier)
    # Create VideoWriter object to write the resized video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # Read a frame from input video
        ret, frame = cap.read()

        if ret:
            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))
            # Write the resized frame to output video
            out.write(resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def increase_data():
    actions = os.listdir('all_vids_regulated')
    pbar = tqdm(total=len(actions), desc='Resizing videos')
    for action in actions:
        for video in os.listdir(os.path.join('all_vids_regulated', action)):
            if not os.path.exists(os.path.join('all_vids_regulated', action)):
                os.makedirs(os.path.join(
                    'all_vids_regulated', action))
            video_path = os.path.join(
                'all_vids_regulated', action, video)
            resize_video(video_path,  2.0)
            resize_video(video_path,  0.5)
            resize_video(video_path,  1.5)
        pbar.update(1)

    pbar = tqdm(total=len(os.listdir('all_vids_regulated')),
                desc='Rotating videos')
    actions = os.listdir('all_vids_regulated')
    for action in actions:
        for video in os.listdir(os.path.join('all_vids_regulated', action)):
            if not os.path.exists(os.path.join('all_vids_regulated', action)):
                os.makedirs(os.path.join(
                    'all_vids_regulated', action))
            video_path = os.path.join(
                'all_vids_regulated', action, video)
            rotate_video(video_path)
        pbar.update(1)


def detect_non_moving_frames(video_path,  threshold=0.1):
    if video_path.endswith('_moving.mp4'):
        return
    # Initialize the video capture object
    output_path = video_path[:-4] + '_moving.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize the previous frame
    ret, prev_frame = cap.read()
    if not ret:
        # retry 10 times
        for i in range(10):
            ret, prev_frame = cap.read()
            if ret:
                break
        if not ret:
            cap.release()
            return

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            # Read the current frame
            ret, frame = cap.read()
            if not ret:
                # retry 10 times
                for i in range(10):
                    ret, frame = cap.read()
                    if ret:
                        break
                if not ret:
                    break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            image, results = mediapipe_detection(frame, holistic)

            if not results.left_hand_landmarks:
                continue
            if not results.right_hand_landmarks:
                continue

            # Calculate the difference between the current and previous frames
            diff = cv2.absdiff(gray, prev_gray)

            # Check if the difference falls below the threshold
            if cv2.mean(diff)[0] < threshold:
                continue

            # Write the frame to the output video
            out.write(frame)

            # Update the previous frame
            prev_gray = gray

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Delete the old video
    os.remove(video_path)


def remove_none_moving_frames_from_video(video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    if video_path.endswith('_moving.mp4'):
        return
    cap = cv2.VideoCapture(video_path)
    # Get the codec information of input video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Get the frame rate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the dimensions of input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to write the resized video
    output_path = video_path[:-4] + '_moving.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence,
                              min_tracking_confidence=min_tracking_confidence) as holistic:
        while cap.isOpened():
            # Read a frame from input video
            ret, frame = cap.read()

            if ret:
                image, results = mediapipe_detection(frame, holistic)
                # Export keypoints
                keypoints = extract_keypoints(results)
                # check if the keypoints are moving by checking if all the keypoints are 0
                is_moving = np.sum(keypoints) != 0
                if is_moving:
                    out.write(frame)

            else:
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # delete the old video
        # os.remove(video_path)

        return None


def remove_none_moving_frames_from_videos(videos_paths):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        with tqdm(total=len(videos_paths), desc='Removing none moving frames from videos') as pbar:
            for _ in p.imap_unordered(detect_non_moving_frames, videos_paths):
                pbar.update()

            # wait for all the processes to finish
            pbar.close()
            p.close()
            p.join()


def get_all_vids_paths(folder_name):
    video_extensions = [
        "3g2",
        "3gp",
        "aaf",
        "asf",
        "avchd",
        "avi",
        "drc",
        "flv",
        "m2v",
        "m3u8",
        "m4p",
        "m4v",
        "mkv",
        "mng",
        "mov",
        "mp2",
        "mp4",
        "mpe",
        "mpeg",
        "mpg",
        "mpv",
        "mxf",
        "nsv",
        "ogg",
        "ogv",
        "qt",
        "rm",
        "rmvb",
        "roq",
        "svi",
        "vob",
        "webm",
        "wmv",
        "yuv"
    ]
    # get all the paths of the videos (.mp4) in folder or subfolders
    vids_paths = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.split('.')[-1] in video_extensions:
                vids_paths.append(os.path.join(root, file))
    return vids_paths


def adjust_videos_total_frames(videos_paths):
    inputs = [(video_path, video_path.replace(
        'all_vids', 'all_vids_regulated'), 15) for video_path in videos_paths]
    # create the folders for the regulated videos
    for input in inputs:
        if not os.path.exists(os.path.dirname(input[1])):
            os.makedirs(os.path.dirname(input[1]))
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        with tqdm(total=len(videos_paths), desc='Adjusting videos total frames') as pbar:
            # call adjust_video_total_frames_using_interpolation for each video with 3 arguments
            # output_path should be the same as the input_path but with parent changed to 'all_vids_regulated'
            for _ in p.starmap(adjust_video_total_frames_using_interpolation, inputs):
                pbar.update()


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
                "visibility": lm.visibility
            })
        keypoints_map['poseLandmarks'] = keypoints
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        keypoints_map['leftHandLandmarks'] = keypoints

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        keypoints_map['rightHandLandmarks'] = keypoints
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        keypoints_map['faceLandmarks'] = keypoints
    return keypoints_map


def make_keypoints_json():
    # get all the videos paths
    keypoints_map = {}
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in os.listdir(os.path.join('all_vids_regulated')):
            # get the first video of the action
            video_path = os.path.join('all_vids_regulated', action, os.listdir(
                os.path.join('all_vids_regulated', action))[0])
            # loop over the video frames
            cap = cv2.VideoCapture(video_path)
            action_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    action_keypoints.append(extract_keypoints_dict(results))
                else:
                    break
            cap.release()
            keypoints_map[action] = action_keypoints

    with open(os.path.join('keypoints_reshaped' + '.json'), 'w') as f:
        json.dump(keypoints_map, f)


def is_video_corrupted(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # try 10 times to read a frame
    for i in range(10):
        if ret:
            return False
        ret, frame = cap.read()
    return True


def remove_corrupted_videos(videos_paths):
    number_of_deleted_videos = 0
    pbar = tqdm(total=len(videos_paths), desc='Removing corrupted videos')
    for video_path in videos_paths:
        if is_video_corrupted(video_path):
            number_of_deleted_videos += 1
            os.remove(video_path)
        pbar.update()
    print(f'{number_of_deleted_videos} videos were deleted')


def save_videos_per_actions_json():
    videos_per_actions = {}
    for action in os.listdir('all_vids_regulated'):
        videos_per_actions[action] = len(os.listdir(
            os.path.join('all_vids_regulated', action)))
    # sort from largest to smallest
    videos_per_actions = dict(
        sorted(videos_per_actions.items(), key=lambda item: item[1], reverse=True))
    with open(os.path.join('videos_per_actions.json'), 'w') as f:
        json.dump(videos_per_actions, f)


if __name__ == '__main__':
    # make_keypoints_json()
    # remove_generated_videos()

    # vids = get_all_vids_paths('all_vids_regulated')
    # remove_corrupted_videos(vids)

    # remove_none_moving_frames_from_videos(vids)

    # # put all videos in one folder
    # folder_name = 'all_vids'
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    # vids = get_all_vids_paths('DATASETS')
    # # video in vids can be DATASETS/DATASET_NAME/ACTIVITY_NAME/VIDEO_NAME.mp4
    # # save all videos in one folder all_vids/ACTIVITY_NAME/VIDEO_NAME.mp4
    # for video in vids:
    #     action = video.split(os.sep)[-2]
    #     if not os.path.exists(os.path.join(folder_name, action)):
    #         os.makedirs(os.path.join(folder_name, action))
    #     shutil.copy(video, os.path.join(folder_name, action))

    # vids = get_all_vids_paths(folder_name)
    # adjust_videos_total_frames(vids)
    # make_keypoints_json()
    # save_videos_per_actions_json()
    # # increase_data()
    vids = get_all_vids_paths('all_vids_regulated')
    save_videos_features(vids)
