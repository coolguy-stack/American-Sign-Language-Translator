import os
import cv2

class DataCollector:
    def __init__(self, data_dir='./data', number_of_classes=26, dataset_size=100):
        self.data_dir = data_dir
        self.number_of_classes = number_of_classes
        self.dataset_size = dataset_size
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        self._prepare_directories()

    def _prepare_directories(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for i in range(self.number_of_classes):
            class_dir = os.path.join(self.data_dir, str(i))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    def collect_data(self):
        for class_id in range(self.number_of_classes):
            print(f'Collecting data for class {class_id}. Press S to start and E to exit.')
            if not self._wait_for_start_signal():
                break
            self._collect_images_for_class(class_id)
    
    def _wait_for_start_signal(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                return False
            cv2.putText(frame, 'Hit S to start. Hit E to exit.', (frame.shape[1]//2 - 185, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('s'):
                return True
            elif key == ord('e'):
                return False

    def _collect_images_for_class(self, class_id):
        counter = 0
        while counter < self.dataset_size:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('e'):
                return
            file_path = os.path.join(self.data_dir, str(class_id), f'{counter}.jpg')
            cv2.imwrite(file_path, frame)
            print(f"Image {counter} saved at {file_path}")
            counter += 1

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    collector.collect_data()
    collector.release()
