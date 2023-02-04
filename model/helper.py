import os
import shutil
limit = 40
min_videos_per_action = 30
for word_folder in os.listdir('MP_Data'):
    if limit == 0:
        break
    action_path = os.path.join('MP_Data', word_folder)
    number_of_videos = len(os.listdir(action_path))
    if number_of_videos < min_videos_per_action:
        continue
    limit -= 1
    number_of_videos_for_test = max(int(number_of_videos * 0.15), 20)
    number_of_videos_for_train = number_of_videos - number_of_videos_for_test
        

    # copy videos to train folder 
    for video in os.listdir(action_path)[:number_of_videos_for_train]:
        video_path = os.path.join(action_path, video)
        new_video_path = os.path.join('MP_Train', word_folder, video)
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            new_frame_path = os.path.join(new_video_path, frame)
            os.makedirs(os.path.dirname(new_frame_path), exist_ok=True)
            shutil.copy(frame_path, new_frame_path)



    #copy videos to test folder
    for video in os.listdir(action_path)[number_of_videos_for_train:]:
        video_path = os.path.join(action_path, video)
        new_video_path = os.path.join('MP_Test', word_folder, video)
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            new_frame_path = os.path.join(new_video_path, frame)
            os.makedirs(os.path.dirname(new_frame_path), exist_ok=True)
            shutil.copy(frame_path, new_frame_path)

    


    