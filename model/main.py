import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mediapipeHelper import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic, mp_drawing
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

DATA_PATH = os.path.join('MP_Data') #mkan el folder elly 3mlna fe save ll features
MIN_VIDEOS = 40 #minimum number of videos for each action
FRAMES_PER_VIDEO = 15 #number of frames per video (wont be changed)

def save_features(actions,actions_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        pbar = tqdm(total=len(actions), desc="Processing actions", unit="action")
        for action in actions:
            
            # loop through every video in the folder of the name action
            videos_path = os.path.join(actions_path, action)
            videos = os.listdir(videos_path)
            for video in videos:
                # get the number of frames in the video
                cap = cv2.VideoCapture(os.path.join(videos_path, video))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Loop through frames
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

                    # Export to numpy array
                    npy_path = os.path.join(
                        DATA_PATH, action, str(start_folder), "{}.npy".format(frame_num))
                    # if the path doesn't exist, create it
                    if not os.path.exists(os.path.dirname(npy_path)):
                        os.makedirs(os.path.dirname(npy_path))

                    np.save(npy_path, keypoints)
                    # Show to screen
                    #cv2.imshow('OpenCV Feed', image)
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                start_folder += 1
            start_folder = 1
            pbar.update(1)

def load_features(actions, label_map):
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(FRAMES_PER_VIDEO):
                res = np.load(os.path.join(DATA_PATH, action, str(
                    sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return sequences, labels 

def create_model(actions):
    multi =1
    model = Sequential()
    model.add(Conv1D(128*multi, kernel_size=2, activation='relu', input_shape=(FRAMES_PER_VIDEO, 126)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(256*multi, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(512*multi, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512*multi, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256*multi , activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=500)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", 'test'))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './models/action{val_loss:.2f}-accuracy{val_accuracy:.2f}.h5', monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max', save_freq='epoch', save_weights_only=False
    )

    callbacks_list = [checkpoint, tensorboard,early_stopping]
    history = model.fit(X_train, y_train, epochs=1000000, batch_size=batch_size,
              validation_data=(X_val, y_val), verbose=1, callbacks=callbacks_list)
    return history


def show_results_and_graphs(history, X_test, y_test):
    if history is None or X_test is None or y_test is None:
        raise Exception("history, X_test and y_test must be given")

    # Show results
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("Test Accuracy: {:.2f}%".format(
        100 * np.sum(y_pred == y_test)/y_test.shape[0]))

    # Show graphs
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r',
               label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b',
               label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',
               label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    plt.show()


if __name__ == '__main__':
    actions = []
    for action in os.listdir(DATA_PATH):
        if len(os.listdir(os.path.join(DATA_PATH, action))) >= MIN_VIDEOS:
            actions.append(action)
    actions = np.array(actions)
    print("we have {} actions".format(actions.shape[0]))

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = load_features(actions, label_map)
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7 ,stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=7 ,stratify=y_test)
    #save_features(actions, DATA_PATH)

    model = create_model(actions)
    history = train_model(model, X_train, y_train, X_val, y_val)

    show_results_and_graphs(history, X_test, y_test)



    


    
    