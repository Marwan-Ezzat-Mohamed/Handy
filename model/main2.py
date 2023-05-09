import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gc


resize_width = 480
resize_height = 360


def resize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (resize_width, resize_height))
    # overwrite the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(video_path, fourcc, 30,
                          (resize_width, resize_height))
    out.write(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resize_width, resize_height))
        out.write(frame)
    cap.release()
    out.release()


def extract_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_list = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(  # calculate optical flow
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_list.append(flow)
        prev_gray = next_gray

    cap.release()
    return np.array(flow_list)


def preprocess_dataset(data_dir):
    X = []
    y = []
    for i, word_dir in enumerate(sorted(os.listdir(data_dir))):
        word = word_dir.split("_")[0]
        print(f"Processing word {word} ({i+1}/{len(os.listdir(data_dir))})")
        for video_path in os.listdir(os.path.join(data_dir, word_dir)):
            # resize_video(os.path.join(data_dir, word_dir, video_path))
            flow = extract_optical_flow(
                os.path.join(data_dir, word_dir, video_path))

            X.append(flow)
            y.append(word)

    X = np.array(X)
    y = np.array(y)
    return X, y


# X, y = np.load("X.npy", allow_pickle=True), np.load("y.npy", allow_pickle=True)

data_dir = "dataset"
# check if data is already saved
if os.path.exists("X.npy") and os.path.exists("y.npy"):
    X = np.load("X.npy")
    y = np.load("y.npy")

else:
    X, y = preprocess_dataset(data_dir)
    np.save("X.npy", X)
    np.save("y.npy", y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


print("X_train shape:", X_train[0].shape)

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train model
X_train_tensor = tf.convert_to_tensor(X_train)
X_test_tensor = tf.convert_to_tensor(X_test)
y_train_tensor = tf.convert_to_tensor(y_train)
y_test_tensor = tf.convert_to_tensor(y_test)

model.fit(X_train_tensor, y_train_tensor, batch_size=32, epochs=10,
          validation_data=(X_test, y_test))

# Evaluate model on test set
score = model.evaluate(X_test_tensor, y_test_tensor, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
