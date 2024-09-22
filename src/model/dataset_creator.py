import os
import pickle
import mediapipe as mp
import cv2

class DatasetCreator:
    def __init__(self, data_dir='./data', output_file='data.pickle'):
        self.data_dir = data_dir
        self.output_file = output_file
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.data = []
        self.labels = []

    def process_images(self):
        for class_dir in os.listdir(self.data_dir):
            full_dir_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(full_dir_path):
                continue
            for img_name in os.listdir(full_dir_path):
                self._process_image(full_dir_path, img_name, class_dir)
        self._save_data()

    def _process_image(self, dir_path, img_name, class_label):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            x_, y_, data_aux = [], [], []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                min_x, min_y = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min_x)
                    data_aux.append(hand_landmarks.landmark[i].y - min_y)
            self.data.append(data_aux)
            self.labels.append(int(class_label))
        else:
            print(f"No hand landmarks found in image {img_path}. Skipping.")

    def _save_data(self):
        with open(self.output_file, 'wb') as f:
            pickle.dump({'data': self.data, 'labels': self.labels}, f)
        print(f"Data saved to {self.output_file}. Total samples: {len(self.data)}")

if __name__ == '__main__':
    creator = DatasetCreator()
    creator.process_images()
