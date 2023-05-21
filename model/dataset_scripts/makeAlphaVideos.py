import os
import cv2
path = "D:/Downloads/ASL Alphabet - Copy/asl_alphabet_train/"
letters = os.listdir(path)
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for letter in letters:
    out_path = f"{path[:-1]}_videos/{letter}/"
    os.makedirs(out_path)
    count = 0
    for i in range(1, 3000, 20):
        count += 1
        out_video_full_path = f"{out_path}video{count}.mp4"
        seq = [f"{path}{letter}/{letter}{i}.jpg" for i in range(i,i+20)]

        frame = cv2.imread(seq[0])
        size = list(frame.shape)
        del size[2]
        size.reverse()

        video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 20, size) #output video name, fourcc, fps, size

        for j in range(len(seq)): 
            video.write(cv2.imread(seq[j]))
            print('frame ', j+1, ' of ', len(seq))

        video.release()
        print('outputed video to ', out_path)