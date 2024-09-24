# American Sign Language Translator

This project is a real-time American Sign Language (ASL) translator that uses a webcam to detect and classify hand gestures into corresponding letters of the alphabet. The system utilizes computer vision and machine learning to identify hand landmarks and make predictions based on a trained model.

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Real-Time Prediction](#real-time-prediction)
- [Contributing](#contributing)
- [License](#license)

## Demo

https://youtu.be/6hGrukDzb3c

## Features

- **Real-time Hand Gesture Recognition**: Uses a webcam to capture hand movements and predict corresponding ASL letters.
- **Machine Learning Model**: Utilizes a Random Forest classifier trained on hand landmarks data.
- **Simple and User-Friendly**: Easy to set up and use for educational or demonstration purposes.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/ASL-Translator.git
   cd ASL-Translator
   ```

2. **Install Dependencies**:

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Additional Requirements**:

   - Ensure you have a working webcam.
   - Python 3.10 or higher is recommended.

## Usage

1. **Run the Real-Time ASL Translator**:

   Navigate to the project directory and run the `realtime_predictor.py` script:

   ```bash
   python realtime_predictor.py
   ```

2. **Instructions**:

   - Make sure your webcam is connected.
   - The program will start capturing video. Show a hand gesture in front of the camera.
   - The corresponding ASL letter will be displayed on the screen.
   - Press `E` to exit the program.

## Project Structure

```
.
├── data/                          # Directory containing image data for training
├── model/                         # Directory for saving trained models
│   └── model.p                    # Pre-trained model file
├── src/
│   ├── dataset_creator.py         # Script for creating and saving the dataset
│   ├── model_trainer.py           # Script for training the model
│   └── realtime_predictor.py      # Script for real-time prediction
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Training the Model

To train the model on your own dataset:

1. **Organize Data**:

   Place your hand gesture images in the `data` directory, organized into subdirectories where each subdirectory name corresponds to the class label (e.g., `0`, `1`, `2`, ..., `25`).

2. **Generate Dataset**:

   Run the `dataset_creator.py` script to create the dataset:

   ```bash
   python src/dataset_creator.py
   ```

3. **Train the Model**:

   Run the `model_trainer.py` script to train the model:

   ```bash
   python src/model_trainer.py
   ```

4. The trained model will be saved in the `model` directory as `model.p`.

## Real-Time Prediction

To use the real-time prediction functionality:

1. Ensure your webcam is connected.
2. Run the `realtime_predictor.py` script:

   ```bash
   python src/realtime_predictor.py
   ```

3. The system will start capturing video, detecting hand gestures, and displaying predictions in real-time.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
