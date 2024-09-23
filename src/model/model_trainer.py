import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class ModelTrainer:
    def __init__(self, data_file='data.pickle', model_file='model.p'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None

    def load_data(self):
        with open(self.data_file, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.labels = np.array(data_dict['labels'])
        self.max_len = max(len(seq) for seq in self.data)
        self.data = self._pad_sequences(self.data, self.max_len)

    def _pad_sequences(self, data, max_len):
        padded_data = []
        for seq in data:
            padded_data.append(seq[:max_len] + [0] * (max_len - len(seq)))
        return np.array(padded_data)

    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, 
                                                            test_size=0.2, stratify=self.labels)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(26)]))
        self._save_model()

    def _save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump({'model': self.model}, f)
        print(f"Model saved to {self.model_file}")

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.load_data()
    trainer.train()
