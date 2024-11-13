import cv2 # Computer vision library allows you to work with your webcam
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import time # Time to get into position before capturing key points
import mediapipe as mp  # To extract key points from your body parts
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

print("Keras version:", tf.keras.__version__)

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


print("Starting..")


mp_holistic = mp.solutions.holistic  # Holistic model to make out detections of body parts
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities to draw landmarks on the parts

def mediapipe_detection(image, model):             # image is captured by webcam
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color Conversion from BGR to RGB
    image.flags.writeable = False                  # Image is no longer writeable here
    results = model.process(image)           # Make detections using our Holistic model
    image.flags.writeable = True                   # Image is now writeable here
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color Conversion from RGB to BGR
    return image, results               # Returning the image and also the prediction


def draw_landmarks(image, results):   # Using Drawing Utilities to draw landmarks on the detections returned by mediapipe
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):  # Using Drawing Utilities to draw landmarks on the detections returned by mediapipe
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


# cap = cv2.VideoCapture(0)  # cap is a variable representing the video capture device
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
#
#     while cap.isOpened():         # While webcam is running
#         ret, frame = cap.read()  # frame is the image from our webcam
#         image, results = mediapipe_detection(frame, model)  # Detection done on the frame
#
#         draw_styled_landmarks(image, results)   # Drawing landmarks on the different parts
#         cv2.imshow('OpenCV Feed', image)  # displaying the image on screen
#
#         # Breaking gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = []
        for res in results.pose_landmarks.landmark:
            temp = np.array([res.x, res.y, res.z, res.visibility])
            pose.append(temp)
        pose = np.array(pose).flatten()
    else:
        pose = np.zeros(33 * 4)

    # if results.pose_landmarks:
    #     pose = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    # else:
    #     pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    return np.concatenate([pose, lh, rh, face])


DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Actions we will try to detect
no_sequences = 30    # 30 videos worth of data
sequence_length = 30    # Each video will be 30 frames in length

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass
#
# cap = cv2.VideoCapture(0)  # cap is a variable representing the video capture device
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
#
#     for action in actions:    # Loop through each action
#         for sequence in range(no_sequences):    # Loop through sequences aka videos
#             for frame_num in range(sequence_length):    # Loop through frames of each video
#
#                 ret, frame = cap.read()  # frame is the image from our webcam
#                 image, results = mediapipe_detection(frame, model)  # Detection done on the frame
#                 draw_styled_landmarks(image, results)  # Drawing landmarks on the different parts
#
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # cv2.imshow('OpenCV Feed', image)  # displaying the image on screen
#                     cv2.waitKey(2000)  # At the first frame for a new video/sequence, we will do a break.
#                 else:
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # cv2.imshow('OpenCV Feed', image)   #  displaying the image on screen
#
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
#                 cv2.imshow('OpenCV Feed', image)  # displaying the image on screen
#
#
#                 # Breaking gracefully
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
#
#     cap.release()
#     cv2.destroyAllWindows()

label_map = {label:num for num, label in enumerate(actions)}   # Numbering each of the three actions
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()  # Instantiating the model
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

model.summary()
model.save('action.h5')

yhat = model.predict(X_test)
yhat = np.argmax(yhat, axis=1).tolist()
ytrue = np.argmax(y_test, axis=1).tolist()
accuracy_score(ytrue, yhat)

sequence = []
sentence = []
threshold = 0.7
predictions = []

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


cap = cv2.VideoCapture(0)  # cap is a variable representing the video capture device
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():         # While webcam is running
        ret, frame = cap.read()  # frame is the image from our webcam
        image, results = mediapipe_detection(frame, holistic)  # Detection done on the frame
        print(results)

        draw_styled_landmarks(image, results)   # Drawing landmarks on the different parts
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('OpenCV Feed', image)  # displaying the image on screen

        # Breaking gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

