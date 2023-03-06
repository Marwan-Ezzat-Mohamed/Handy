import os
import random
import shutil
from natsort import natsorted
from tqdm import tqdm
import numpy as np
# specify the path to the parent directory containing the ASL word folders


def make_train_data(num_new_folders=1, use_val=False):
    parent_dir = os.path.join('MP_Val')
    # if use_val is True, then the val data will be used as the training data
    if use_val:
        # copy the all files in MP_Val to the MP_Train
        try:
            shutil.rmtree('MP_Train')
        except:
            pass

        # os.makedirs('MP_Train')

        shutil.copytree('MP_Val', 'MP_Train')
        return

    pbar = tqdm(total=len(os.listdir(parent_dir)),
                desc='Generating train data')
    # specify the number of new video feature folders you want to create
    # iterate through each ASL word folder
    for word_folder in os.listdir(parent_dir):
        # if the word has less than 7 videos, skip it

        word_path = os.path.join(parent_dir, word_folder)
        # get the list of video feature folders
        video_folders = [f for f in os.listdir(
            word_path) if os.path.isdir(os.path.join(word_path, f))]
        # create a list of all numpy array files in the video feature folders
        npy_files = []
        for video_folder in video_folders:
            video_path = os.path.join(word_path, video_folder)
            npy_files.extend([os.path.join(video_path, f)
                              for f in os.listdir(video_path) if f.endswith('.npy')])

        # generate 100 new video feature folders and make sure that each video is unique
        new_videos_seq = set()
        while len(new_videos_seq) < num_new_folders:
            # randomly select 15 numpy array files and make sure that we have 15 unique files numbered 0-14
            selected_npy_files = set()
            filenames = set()
            while len(selected_npy_files) < 15:
                selected_npy_file = random.choice(npy_files)
                filename = os.path.basename(selected_npy_file)
                if filename not in filenames:
                    filenames.add(filename)
                    selected_npy_files.add(selected_npy_file)

            # convert selected_npy_files to string and add it to the set
            # sort the set
            selected_npy_files = natsorted(selected_npy_files)
            selected_npy_files = ','.join(selected_npy_files)
            new_videos_seq.add(selected_npy_files)

        # convert the set to a list
        new_videos_seq = list(new_videos_seq)

        # copy the selected numpy array files to the new video feature folder
        for i in range(num_new_folders):
            new_video_folder = os.path.join('MP_Train', word_folder, str(i))
            os.makedirs(new_video_folder, exist_ok=True)
            folder_list = new_videos_seq[i].split(',')
            for npy_file in folder_list:
                # print("copying", npy_file, "to", new_video_folder)
                shutil.copy(npy_file, new_video_folder)
            # print('\n')
        pbar.update(1)


# a function to generate train data by concatenating the keypoints of left hand of one video and the keypoints of right hand of another video
def make_train_data_concat(num_new_folders=70):
    new_folder_cnt = num_new_folders
    parent_dir = os.path.join('MP_Val')
    try:
        shutil.rmtree('MP_Train')
    except:
        pass

    # os.makedirs('MP_Train')
    print("Copying MP_Val to MP_Train")
    shutil.copytree('MP_Val', 'MP_Train')
    print("Done copying")

    pbar = tqdm(total=len(os.listdir(parent_dir)),
                desc='Generating train data')
    # specify the number of new video feature folders you want to create
    # iterate through each ASL word folder
    for word_folder in os.listdir(parent_dir):
        # loop on every video in the word folder and concatenate the keypoints of left hand of one video and the keypoints of right hand of another video
        num_new_folders = new_folder_cnt - \
            len(os.listdir(os.path.join(parent_dir, word_folder)))
        num_new_folders = max(num_new_folders, 0)
        for first_vid in os.listdir(os.path.join(parent_dir, word_folder)):
            for second_vid in os.listdir(os.path.join(parent_dir, word_folder)):
                if first_vid != second_vid:
                    if num_new_folders <= 0:
                        continue
                    num_new_folders -= 1
                    # load the keypoints of the first video
                    first_vid_path = os.path.join(
                        parent_dir, word_folder, first_vid)

                    first_vid_keypoints = []
                    for npy_file in os.listdir(first_vid_path):
                        first_vid_keypoints.append(np.load(os.path.join(
                            first_vid_path, npy_file)))

                    # load the keypoints of the second video
                    second_vid_path = os.path.join(
                        parent_dir, word_folder, second_vid)

                    second_vid_keypoints = []
                    for npy_file in os.listdir(second_vid_path):
                        second_vid_keypoints.append(np.load(os.path.join(
                            second_vid_path, npy_file)))

                    # left hand is the first 21*3 keypoints and right hand is the last 21*3 keypoints
                    # concatenate the keypoints of left hand of one video and the keypoints of right hand of another video
                    new_vid_keypoints = []
                    for i in range(len(first_vid_keypoints)):
                        new_vid_keypoints.append(np.concatenate(
                            (first_vid_keypoints[i][:63], second_vid_keypoints[i][63:]), axis=0))

                    # save the new video keypoints to a new folder
                    new_video_folder = os.path.join(
                        'MP_Train', word_folder, first_vid + '_' + second_vid)
                    os.makedirs(new_video_folder, exist_ok=True)
                    for i in range(len(new_vid_keypoints)):
                        np.save(os.path.join(new_video_folder, str(i) + '.npy'),
                                new_vid_keypoints[i])

        pbar.update(1)


if __name__ == '__main__':
    # # create an np array that have first 63 values as 1 and last 63 values as -1
    # array = np.ones((63))
    # array = np.concatenate((array, -1*array))

    # os.makedirs('MP_Val/0', exist_ok=True)
    # os.makedirs('MP_Val/1', exist_ok=True)
    # np.save('MP_Val/0/array1.npy', array)
    # np.save('MP_Val/0/array2.npy', array)
    # # create an np array that have first 63 values as 2 and last 63 values as -2
    # array = np.ones((63))
    # array = np.concatenate((array*2, -1*array*2))

    # np.save('MP_Val/1/array1.npy', array)
    # np.save('MP_Val/1/array2.npy', array)
    make_train_data_concat()
