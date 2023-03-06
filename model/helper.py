import os
import shutil
from tqdm import tqdm
import numpy as np
from mediapipeHelper import *
import math
import json
import time


def make_test_val_data(limit=40,
                       min_videos_per_action=40):
    # count the number of videos that are greater than min_videos_per_action
    number_of_videos = [f for f in os.listdir(
        'MP_Data') if len(os.listdir(os.path.join('MP_Data', f))) >= min_videos_per_action]
    pbar = tqdm(total=min(len(number_of_videos), limit),
                desc='Generating test and val data')
    for word_folder in os.listdir('MP_Data'):
        if limit == 0:
            break
        action_path = os.path.join('MP_Data', word_folder)
        number_of_videos = len(os.listdir(action_path))
        if number_of_videos < min_videos_per_action:
            continue
        limit -= 1
        number_of_videos_for_test = max(int(number_of_videos * 0.01), 2)
        number_of_videos_for_train = number_of_videos - number_of_videos_for_test

        # val data is used for generating the new data (will be used for training)
        # copy videos to train folder
        for video in os.listdir(action_path)[:number_of_videos_for_train]:
            video_path = os.path.join(action_path, video)
            new_video_path = os.path.join('MP_Val', word_folder, video)
            os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
            for frame in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame)
                new_frame_path = os.path.join(new_video_path, frame)
                os.makedirs(os.path.dirname(new_frame_path), exist_ok=True)
                shutil.copy(frame_path, new_frame_path)

        # copy videos to test folder
        for video in os.listdir(action_path)[number_of_videos_for_train:]:
            video_path = os.path.join(action_path, video)
            new_video_path = os.path.join('MP_Test', word_folder, video)
            os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
            for frame in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame)
                new_frame_path = os.path.join(new_video_path, frame)
                os.makedirs(os.path.dirname(new_frame_path), exist_ok=True)
                shutil.copy(frame_path, new_frame_path)
        pbar.update(1)


def save_one_features(video_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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

            # Draw landmarks
            draw_styled_landmarks(image, results)
            # Export keypoints
            keypoints = extract_keypoints(results)
            # Reshape keypoints
            to_save.append(keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # save as text file
        # get the name of the video
        #
        np.save(video_path[:-4], to_save)
        cap.release()
        cv2.destroyAllWindows()


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
    print(output_path)

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


def resize_video(input_path, multiplier=2.0):
    # output_path save as input_path but adding _resized to the name and the multiplier
    output_path = input_path[:-4] + '_resized_' + str(multiplier) + 'x.mp4'
    print(output_path)

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
            resized_frame = random_rotation(resized_frame)
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


def save_one_keypoints_for_words():
    # make a json file for each word with the keypoints of one video
    for word_folder in os.listdir('MP_Data'):
        for video in os.listdir(os.path.join('MP_Data', word_folder)):
            video_path = os.path.join('MP_Data', word_folder, video)
            keypoints = []
            for frame in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame)
                # load all the keypoints of the video
                keypoints.append(np.load(frame_path))
        # append the keypoints of the video in a json file
        left_hand = []
        right_hand = []

        break


def remove_not_moving_frames(videos_paths):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, tqdm(total=len(videos_paths)) as pbar:
        for video_path in videos_paths:
            cap = cv2.VideoCapture(video_path)

            # Get the codec information of input video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Get the frame rate of input video
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Get the dimensions of input video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create VideoWriter object to write the resized video
            output_path = video_path[:-4] + '_moving.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    # check if the keypoints are moving by checking if any of the keypoints are non-zero
                    is_moving = np.any(keypoints)
                    if is_moving:
                        out.write(frame)

            # Release resources
            cap.release()
            out.release()

            # delete the old video
            os.remove(video_path)

            pbar.update(1)

    cv2.destroyAllWindows()


def get_all_vids_paths(folder_name):
    # get all the paths of the videos (.mp4) in folder or subfolders
    vids_paths = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.mp4'):
                vids_paths.append(os.path.join(root, file))
    return vids_paths


if __name__ == '__main__':
    # make_test_val_data()
    videos = get_all_vids_paths('MSASL')
    # calculate time taken to process all the videos
    start = time.time()

    remove_not_moving_frames(videos)
    end = time.time()
    print("Time taken to process all the videos: ", end - start)


def get_total_frames(video_path):
    # get the total number of frames in a video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames
