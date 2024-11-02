import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# OpenCV setup for video capture
cap = cv2.VideoCapture(1)

# MediaPipe setup for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping integer labels to ASL signs and their meanings
asl_signs = {
    0: ('A', 'Letter A'),
    1: ('B', 'Letter B'),
    2: ('C', 'Letter C'),
    3: ('D', 'Letter D'),
    4: ('E', 'Letter E'),
    5: ('F', 'Letter F'), 
    6: ('G', 'Letter G'),
    7: ('H', 'Letter H'),
    8: ('I', 'Letter I'),
    9: ('J', 'Letter J'),
    10: ('K', 'Letter K'),
    11: ('L', 'Letter L'),
    12: ('M', 'Letter M'),
    13: ('N', 'Letter N'),
    14: ('O', 'Letter O'),
    15: ('P', 'Letter P'),
    16: ('Q', 'Letter Q'),
    17: ('R', 'Letter R'),
    18: ('S', 'Letter S'),
    19: ('T', 'Letter T'),
    20: ('U', 'Letter U'),
    21: ('V', 'Letter V'),
    22: ('W', 'Letter W'),
    23: ('X', 'Letter X'),
    24: ('Y', 'Letter Y'),
    25: ('Z', 'Letter Z'),
    26: ('0', 'Letter 0'),
    27: ('1', 'Letter 1'),
    28: ('2', 'Letter 2'),
    29: ('3', 'Letter 3'),
    30: ('4', 'Letter 4'),
    31: ('5', 'Letter 5'),
    32: ('6', 'Letter 6'),
    33: ('7', 'Letter 7'),
    34: ('8', 'Letter 8'),
    35: ('9', 'Letter 9'),
    
    # Add more ASL signs and their meanings here
}

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Main loop
while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read frame from the webcam
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the gesture using the pre-trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted character and its meaning from the ASL dictionary
        predicted_character, meaning = asl_signs[int(prediction[0])]

        # Convert the predicted character and its meaning to speech
        engine.say(f"The predicted sign is {meaning}")
        engine.runAndWait()

        # Annotate the frame with the predicted character and its meaning
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, f"{predicted_character} - {meaning}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 0), 3, cv2.LINE_AA)

    # Display the annotated frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
