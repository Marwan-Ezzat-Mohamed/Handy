import os
import random
import shutil
from natsort import natsorted
# specify the path to the parent directory containing the ASL word folders


def make_train_data(num_new_folders=50):
    parent_dir = os.path.join('MP_Val')
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
