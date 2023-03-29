import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling1D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

import shutil

from helper import make_test_val_data


# mkan el folder elly 3mlna fe save ll features
TRAIN_DATA_PATH = os.path.join('MP_Train')
TEST_DATA_PATH = os.path.join('MP_Test')
VAL_DATA_PATH = os.path.join('MP_Val')

LABEL_MAP_PATH = os.path.join('label_map.json')

MIN_VIDEOS = 10  # minimum number of videos for each action
FRAMES_PER_VIDEO = 15  # number of frames per video (wont be changed)


# def save_features(actions, actions_path):
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

#         pbar = tqdm(total=len(actions),
#                     desc="Processing actions", unit="action")
#         for action in actions:

#             # loop through every video in the folder of the name action
#             videos_path = os.path.join(actions_path, action)
#             videos = os.listdir(videos_path)

#             for video in videos:
#                 # get the number of frames in the video
#                 cap = cv2.VideoCapture(os.path.join(videos_path, video))
#                 length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 # Loop through frames
#                 for frame_num in range(length):
#                     # Read feed
#                     success, image = cap.read()
#                     # Make detections
#                     if image is None:
#                         continue
#                     image, results = mediapipe_detection(image, holistic)

#                     # Draw landmarks
#                     draw_styled_landmarks(image, results)
#                     # Export keypoints
#                     keypoints = extract_keypoints(results)
#                     # Reshape keypoints

#                     # Export to numpy array
#                     npy_path = os.path.join(
#                         DATA_PATH, action, str(start_folder), "{}.npy".format(frame_num))
#                     # if the path doesn't exist, create it
#                     if not os.path.exists(os.path.dirname(npy_path)):
#                         os.makedirs(os.path.dirname(npy_path))

#                     np.save(npy_path, keypoints)
#                     # Show to screen
#                     # cv2.imshow('OpenCV Feed', image)
#                     # Break gracefully
#                     if cv2.waitKey(10) & 0xFF == ord('q'):
#                         break
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 start_folder += 1
#             start_folder = 1
#             pbar.update(1)


def load_features(actions: list, label_map: dict, data_type: str = "train") -> tuple:
    """
    This function loads the features from the npy files and returns them as numpy arrays,
    it also saves the features as npy files for future use
    """
    allowed_data_types = ['train', 'test', 'val']
    if data_type not in allowed_data_types:
        raise ValueError(
            f"Invalid value for 'param', allowed values are: {allowed_data_types}")
    data_path = None
    if data_type == "train":
        data_path = TRAIN_DATA_PATH
        # check if the data is already saved as npy file
        if os.path.exists('x_' + data_type + '.npy') and os.path.exists('y_' + data_type + '.npy'):
            x = np.load('x_' + data_type + '.npy')
            y = np.load('y_' + data_type + '.npy')
            return x, y

    elif data_type == "test":
        data_path = TEST_DATA_PATH
        if os.path.exists('x_' + data_type + '.npy') and os.path.exists('y_' + data_type + '.npy'):
            x = np.load('x_' + data_type + '.npy')
            y = np.load('y_' + data_type + '.npy')
            return x, y
    elif data_type == "val":
        data_path = VAL_DATA_PATH
        if os.path.exists('x_' + data_type + '.npy') and os.path.exists('y_' + data_type + '.npy'):
            x = np.load('x_' + data_type + '.npy')
            y = np.load('y_' + data_type + '.npy')
            return x, y

    sequences, labels = [], []
    for action in actions:
        for sequence in os.listdir(os.path.join(data_path, action)):

            res = np.load(os.path.join(
                data_path, action, sequence))

            sequences.append(res)
            labels.append(label_map[action])
    sequences, labels = np.array(sequences), to_categorical(labels).astype(int)
    # save the data as npy file
    np.save('x_' + data_type + '.npy', sequences)
    np.save('y_' + data_type + '.npy', labels)
    return sequences, labels


def create_model(actions):

    multi = 1
    model = Sequential()
    model.add(Conv1D(128*multi, kernel_size=2, activation='elu',
              input_shape=(FRAMES_PER_VIDEO, 126)))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(256*multi, kernel_size=2, activation='elu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(512*multi, kernel_size=2, activation='elu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512*multi, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(256*multi, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=16):
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=300)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", 'test'))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './models/action{val_loss:.2f}-accuracy{val_accuracy:.2f}.h5', monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max', save_freq='epoch', save_weights_only=False
    )

    callbacks_list = [checkpoint, tensorboard, early_stopping]
    history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1, callbacks=callbacks_list, workers=4)
    return history


def show_test_results(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        raise Exception("model, X_test and y_test must be given")

    # Show results
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    test_accuracy = 100 * np.sum(y_pred == y_test)/y_test.shape[0]
    print("Test Accuracy: {:.2f}%".format(
        test_accuracy))

    return test_accuracy


def get_word(word):

    synonym_map = {}
    with open(os.path.join('MS-ASL', "MSASL_synonym.json")) as f:
        data = json.load(f)
        for synonym in data:
            synonym_map[synonym[0]] = synonym
    word = word.lower()
    word = word.split("/")[0]
    word = word.replace("-", " ")
    word = word.replace("#", "")
    for key, value in synonym_map.items():
        if word in value:
            return key
    return word


if __name__ == '__main__':
    # if not os.path.exists(TRAIN_DATA_PATH):
    #     if os.path.exists(TEST_DATA_PATH):
    #         print("deleting old test data...")
    #         shutil.rmtree(TEST_DATA_PATH)
    #     if os.path.exists(VAL_DATA_PATH):
    #         print("deleting old val data...")
    #         shutil.rmtree(VAL_DATA_PATH)

    #     make_test_val_data(limit=1)
    #     # delete the old train data if exists
    #     if os.path.exists(TRAIN_DATA_PATH):
    #         print("deleting old train data...")
    #         shutil.rmtree(TRAIN_DATA_PATH)

    #     split_val_to_val_and_test()

    actions = []
    actions_count = {}
    limit = 0
    actions = os.listdir(TRAIN_DATA_PATH)
    actions = np.array(actions)
    print("we have {} actions".format(actions.shape[0]))

    label_map = {label: num for num, label in enumerate(actions)}

    # delete the old json file if exists
    if os.path.exists(LABEL_MAP_PATH):
        os.remove(LABEL_MAP_PATH)
    # save label map into json file reversed and
    with open(LABEL_MAP_PATH, 'w') as fp:
        json.dump({v: k for k, v in label_map.items()}, fp)

    # load x_test and y_test and x_val and y_val and x_train and y_train

    X_train, y_train = load_features(actions, label_map, data_type='train')
    X_test, y_test = load_features(actions, label_map, data_type='test')
    X_val, y_val = load_features(actions, label_map, data_type='val')
    print("loaded x_test, y_test, x_val, y_val, x_train, y_train")

    batch_sizes = [

        actions.shape[0]//4,
    ]

    # # array that start from 10 to 100
    # min_videos = []
    # min_vids_map = {}
    # for i in range(40, 41):
    #     if os.path.exists(TEST_DATA_PATH):
    #         print("deleting old test data...")
    #         shutil.rmtree(TEST_DATA_PATH)
    #     if os.path.exists(VAL_DATA_PATH):
    #         print("deleting old val data...")
    #         shutil.rmtree(VAL_DATA_PATH)

    #     make_test_val_data(min_videos_per_action=i)
    #     # delete the old train data if exists
    #     if os.path.exists(TRAIN_DATA_PATH):
    #         print("deleting old train data...")
    #         shutil.rmtree(TRAIN_DATA_PATH)

    #     make_train_data(use_val=True)
    #     actions = os.listdir(TRAIN_DATA_PATH)
    #     actions = np.array(actions)
    #     print("we have {} actions".format(actions.shape[0]))

    #     label_map = {label: num for num, label in enumerate(actions)}
    #     X_train, y_train = load_features(actions, label_map, data_type='train')
    #     X_test, y_test = load_features(actions, label_map, data_type='test')
    #     X_val, y_val = load_features(actions, label_map, data_type='val')
    #     print("loaded x_test, y_test, x_val, y_val, x_train, y_train")
    #     model = create_model(actions)
    #     history = train_model(model, X_train, y_train,
    #                           X_val, y_val, batch_size=batch_sizes[0])
    #     accuracy = show_test_results(model, X_test, y_test)
    #     min_vids_map["acc:"+str(i) + " num_of_actions:" +
    #                  str(actions.shape[0])] = accuracy
    # print(min_vids_map)

    # sort the batch sizes
    batch_sizes.sort()
    batch_sizes_accuracy = {}
    for batch_size in batch_sizes:
        model = create_model(actions)
        # model.load_weights("./action0.23-accuracy0.94.h5")
        history = train_model(model, X_train, y_train,
                              X_val, y_val, batch_size=batch_size)
        accuracy = show_test_results(model, X_test, y_test)
        batch_sizes_accuracy[batch_size] = accuracy
        model.save("./models.h5")

    print(batch_sizes_accuracy)
    print("top 5 batch sizes")
    print(sorted(batch_sizes_accuracy.items(),
                 key=lambda x: x[1], reverse=True)[:5])
