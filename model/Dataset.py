from itertools import repeat
import math
import multiprocessing
import os
import shutil
from typing import List
from tqdm import tqdm
import numpy as np
from mediapipeHelper import *
import cv2
import time
import json
import istarmap


class Dataset:

    # static variables
    resize_values = [1.2, 0.5, 1.4, 1.6, 1.8]
    features_output_folder_name = 'MP_DATA_NEW',

    def __init__(
            self,
            dataset_folder_path='DATASETS',
            num_frames_per_video=20,
            features_output_folder_name='MP_DATA_NEW',
            resize_values=[1.2, 0.5, 1.4, 1.6, 1.8],
            save_new_videos=True,
    ):
        if features_output_folder_name is not None:
            Dataset.features_output_folder_name = features_output_folder_name
        if resize_values is not None:
            Dataset.resize_values = resize_values
        self.dataset_folder_path = dataset_folder_path
        self.num_frames_per_video = num_frames_per_video
        self.all_videos_folder_name = 'all_vids'
        self.all_regulated_videos_folder_name = f'all_vids_regulated_{num_frames_per_video}'

        self.save_new_videos = save_new_videos

    def process(self):
        # self._put_all_videos_in_one_folder()
        # self._remove_corrupted_videos()
        # self._remove_none_moving_frames_from_videos()
        # self._adjust_videos_total_frames()
        self._increase_videos_by_data_augmentation()
        self._save_videos_per_actions_json()
        self._save_videos_features()

    def _put_all_videos_in_one_folder(self) -> None:
        if not os.path.exists(self.all_videos_folder_name):
            os.makedirs(self.all_videos_folder_name, exist_ok=True)
        # ['whatever//whatever//action//video.mp4']
        vids: List[str] = self.get_all_vids_paths(self.dataset_folder_path)
        pbar: tqdm = tqdm(
            total=len(vids), desc='Putting all videos in one folder')
        for video in vids:
            action: str = video.split(os.sep)[-2]
            video_name: str = video.split(os.sep)[-1]
            # check if video already exists
            if not os.path.exists(os.path.join(self.all_videos_folder_name, action, video_name)):
                # create the folder for the action if it doesn't exist
                if not os.path.exists(os.path.join(self.all_videos_folder_name, action)):
                    os.makedirs(os.path.join(
                        self.all_videos_folder_name, action), exist_ok=True)
                # copy the video to the folder
                shutil.copy(video, os.path.join(
                    self.all_videos_folder_name, action))
            pbar.update()

    def _remove_corrupted_videos(self) -> None:
        vids: List[str] = Dataset.get_all_vids_paths(
            self.all_videos_folder_name)
        Dataset.remove_corrupted_videos(vids)

    def _remove_none_moving_frames_from_videos(self) -> None:
        vids = Dataset.get_all_vids_paths(self.all_videos_folder_name)
        Dataset.remove_none_moving_frames_from_videos(vids)

    def _adjust_videos_total_frames(self) -> None:
        vids: List[str] = Dataset.get_all_vids_paths(
            self.all_videos_folder_name)
        Dataset.regulate_videos_total_frames(
            vids, self.num_frames_per_video, self.all_regulated_videos_folder_name)

    def _increase_videos_by_data_augmentation(self) -> None:
        Dataset.increase_videos_by_data_augmentation(
            self.all_regulated_videos_folder_name, self.save_new_videos)

    def _save_videos_per_actions_json(self) -> None:
        Dataset.save_videos_per_actions_json(
            self.all_regulated_videos_folder_name)

    def _save_videos_features(self) -> None:
        vids = Dataset.get_all_vids_paths(
            self.all_regulated_videos_folder_name)
        Dataset.save_videos_features(vids)

    @staticmethod
    def regulate_videos_total_frames(videos_paths, num_frames_per_video, output_folder_name):
        inputs = [(video_path, video_path.replace(
            'all_vids', output_folder_name), num_frames_per_video) for video_path in videos_paths]
        # create the folders for the regulated videos
        for input in inputs:
            if not os.path.exists(os.path.dirname(input[1])):
                os.makedirs(os.path.dirname(input[1]))
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            with tqdm(total=len(videos_paths), desc='Adjusting videos total frames') as pbar:
                # call adjust_video_total_frames_using_interpolation for each video with 3 arguments
                # output_path should be the same as the input_path but with parent changed to 'all_vids_regulated'
                for _ in p.istarmap(Dataset.adjust_video_total_frames_using_interpolation, inputs):
                    pbar.update()

    @staticmethod
    def save_video_features(video_path: str):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # get the number of frames in the video
            cap: cv2.VideoCapture = cv2.VideoCapture(os.path.join(video_path))
            length: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            output_path[0] = Dataset.features_output_folder_name
            output_path[-1] = output_path[-1].split('.')[0] + '.npy'
            output_path = '/'.join(output_path)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, to_save)
            cap.release()
            cv2.destroyAllWindows()

    @staticmethod
    def save_videos_features(vids: List[str]):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            with tqdm(total=len(vids), desc='Saving videos features') as pbar:
                for _ in p.imap_unordered(Dataset.save_video_features, vids):
                    pbar.update()

    @staticmethod
    def get_all_vids_paths(folder_path):
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
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.split('.')[-1] in video_extensions:
                    vids_paths.append(os.path.join(root, file))
        return vids_paths

    @staticmethod
    def adjust_videos_total_frames(videos_paths, num_frames=15, new_folder_name='all_vids_regulated'):
        try:
            shutil.rmtree(new_folder_name)
        except:
            pass

        inputs = [(video_path, video_path.replace(
            'all_vids', new_folder_name), num_frames) for video_path in videos_paths]
        # create the folders for the regulated videos
        for input in inputs:
            if not os.path.exists(os.path.dirname(input[1])):
                os.makedirs(os.path.dirname(input[1]))
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            with tqdm(total=len(videos_paths), desc='Adjusting videos total frames') as pbar:
                # call adjust_video_total_frames_using_interpolation for each video with 3 arguments
                # output_path should be the same as the input_path but with parent changed to 'all_vids_regulated'
                for _ in p.starmap(Dataset.adjust_video_total_frames_using_interpolation, inputs):
                    pbar.update()

    @staticmethod
    def adjust_video_total_frames_using_interpolation(video_path, output_path, num_frames):
        # read the video
        cap = cv2.VideoCapture(video_path)
        # get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # check if the video in the output path already exists
        if os.path.exists(output_path):
            # check if the video has the same number of frames
            cap2 = cv2.VideoCapture(output_path)
            total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames2 == num_frames:
                # if the video has the same number of frames, return
                cap.release()
                cap2.release()
                return

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

    @staticmethod
    def random_rotation(frame):
        """
        Applies a random rotation to a video frame with a maximum rotation angle of 20 degrees left or right.
        """
        # Generate a random angle between -7 and 7 degrees
        max_angle = 7
        angle = np.random.randint(-max_angle, max_angle)

        # Get the height and width of the frame
        height, width = frame.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            (width/2, height/2), angle, 1)

        # Apply the rotation to the frame
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        return rotated_frame

    @staticmethod
    def rotate_video(input_path, vid_num="", save_video=True):
        # output_path save as input_path but adding _rotated to the name and the angle
        output_path = input_path[:-4] + '_rotated_' + str(vid_num) + '.mp4'
        if os.path.exists(output_path):
            return

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
                rotated_frame = Dataset.random_rotation(frame)
                # Write the rotated frame to output video
                if save_video:
                    out.write(rotated_frame)

            else:
                break

        # Release input and output video
        cap.release()
        out.release()

        # Close all windows
        cv2.destroyAllWindows()

    @staticmethod
    def resize_video(input_path,  multiplier=2.0, save_video=True):
        # output_path save as input_path but adding _resized to the name and the multiplier
        output_path = input_path[:-4] + '_resized_' + str(multiplier) + 'x.mp4'
        if os.path.exists(output_path):
            return

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

    @staticmethod
    def increase_videos_by_data_augmentation(folder_path='all_vids_regulated', save_videos=True):
        actions = os.listdir(folder_path)
        pbar = tqdm(total=len(actions), desc='Resizing videos')
        for action in actions:
            for video in os.listdir(os.path.join(folder_path, action)):
                if not os.path.exists(os.path.join(folder_path, action)):
                    os.makedirs(os.path.join(
                        folder_path, action))
                video_path = os.path.join(
                    folder_path, action, video)

                for multiplier in Dataset.resize_values:
                    Dataset.resize_video(
                        video_path, multiplier, save_videos)

            pbar.update(1)

        pbar = tqdm(total=len(os.listdir(folder_path)),
                    desc='Rotating videos')
        actions = os.listdir(folder_path)
        for action in actions:
            for video in os.listdir(os.path.join(folder_path, action)):
                if not os.path.exists(os.path.join(folder_path, action)):
                    os.makedirs(os.path.join(
                        folder_path, action))
                video_path = os.path.join(
                    folder_path, action, video)
                Dataset.rotate_video(video_path, 0, save_videos)
                Dataset.rotate_video(video_path, 1, save_videos)
                Dataset.rotate_video(video_path, 2, save_videos)
            pbar.update(1)

    @staticmethod
    def detect_non_moving_frames(video_path,  threshold=0.08):
        moving_video_suffix = '_moving.mp4'
        if video_path.endswith(moving_video_suffix):
            return
        # Initialize the video capture object
        output_path = video_path[:-4] + moving_video_suffix
        cap = cv2.VideoCapture(video_path)

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize the video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        def read_retry(cap, n=10):
            for i in range(n):
                ret, frame = cap.read()
                if ret:
                    return ret, frame
            return False, None

        # Initialize the previous frame
        ret, prev_frame = read_retry(cap)
        if not ret:
            return

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        # Convert the frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            while True:
                # Read the current frame
                ret, frame = read_retry(cap)
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

    @staticmethod
    def remove_none_moving_frames_from_video(video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        moving_video_suffix = '_moving.mp4'
        if video_path.endswith(moving_video_suffix):
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
        output_path = video_path[:-4] + moving_video_suffix
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

    @staticmethod
    def remove_none_moving_frames_from_videos(videos_paths):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            with tqdm(total=len(videos_paths), desc='Removing none moving frames from videos') as pbar:
                for _ in p.imap_unordered(Dataset.detect_non_moving_frames, videos_paths):
                    pbar.update()

                # wait for all the processes to finish
                pbar.close()
                p.close()
                p.join()

    @staticmethod
    def is_video_corrupted(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        # try 10 times to read a frame
        for i in range(10):
            if ret:
                return False
            ret, frame = cap.read()
        return True

    @staticmethod
    def remove_corrupted_videos(videos_paths):
        number_of_deleted_videos = 0
        pbar = tqdm(total=len(videos_paths), desc='Removing corrupted videos')
        for video_path in videos_paths:
            if Dataset.is_video_corrupted(video_path):
                number_of_deleted_videos += 1
                os.remove(video_path)
            pbar.update()
        print(f'{number_of_deleted_videos} videos were deleted')

    @staticmethod
    def save_videos_per_actions_json(videos_path='all_vids_regulated', json_path='videos_per_actions.json'):
        videos_per_actions = {}
        for action in os.listdir(videos_path):
            videos_per_actions[action] = len(os.listdir(
                os.path.join(videos_path, action)))
        # sort from largest to smallest
        videos_per_actions = dict(
            sorted(videos_per_actions.items(), key=lambda item: item[1], reverse=True))
        with open(os.path.join(json_path), 'w') as f:
            json.dump(videos_per_actions, f)


d = Dataset()
d.process()
