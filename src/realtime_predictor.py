import pickle
import cv2
import mediapipe as mp
import numpy as np

class RealTimePredictor:
    def __init__(self, model_file='model/model.p'):
        self.model_file = model_file
        self.model = self._load_model()
        self.labels_dict = {i: chr(65 + i) for i in range(26)}
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, 
                                              min_detection_confidence=0.7)
        self.max_len = 42

    def _load_model(self):
        with open(self.model_file, 'rb') as f:
            return pickle.load(f)['model']

    def _pad_sequence(self, sequence):
        return sequence[:self.max_len] + [0] * (self.max_len - len(sequence))

    def predict(self):
        while True:
            data_aux, x_, y_ = [], [], []
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)
                    min_x, min_y = min(x_), min(y_)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min_x)
                        data_aux.append(hand_landmarks.landmark[i].y - min_y)
                data_aux_padded = self._pad_sequence(data_aux)
                prediction = self.model.predict([np.asarray(data_aux_padded)])
                predicted_char = self.labels_dict[int(prediction[0])]
                x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
                x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "Press E to exit", (frame.shape[1] // 2 - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break
        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    predictor = RealTimePredictor()
    predictor.predict()
