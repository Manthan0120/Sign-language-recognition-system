# Sign-language-recognition-system using Machine Learning

This project implements a real-time American Sign Language (ASL) recognition system using computer vision and machine learning. The system can recognize ASL alphabets (A-Z) and numbers (0-9) using a webcam feed and provides both visual and audio feedback of the detected signs.

## 🎯 Features

- Real-time hand gesture recognition
- Support for all ASL alphabets (A-Z) and numbers (0-9)
- Visual display of detected signs with bounding boxes
- Text-to-speech output for detected signs
- High-accuracy machine learning model using Random Forest Classifier
- Built with MediaPipe for robust hand landmark detection

## 🛠️ Technical Stack

- **Python 3.x**
- **OpenCV** - For webcam interface and image processing
- **MediaPipe** - For hand landmark detection
- **scikit-learn** - For machine learning (Random Forest Classifier)
- **pyttsx3** - For text-to-speech conversion
- **NumPy** - For numerical operations
- **pickle** - For model serialization

## 📁 Project Structure

```
.
├── collect_imgs.py          # Script to collect training images
├── create_dataset.py        # Script to process images and create training data
├── train_classifier.py      # Script to train the Random Forest model
├── inference_classifier.py  # Main script for real-time ASL recognition
├── data/                   # Directory containing training images
├── data.pickle            # Processed dataset
└── model.p                # Trained model file
```

## 🚀 Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/asl-recognition.git
cd asl-recognition
```

2. Install required packages:
```bash
pip install opencv-python mediapipe scikit-learn numpy pyttsx3
```

3. Create the dataset:
   - Run `collect_imgs.py` to capture training images
   - Follow the prompts to collect images for each sign
   - Press 'q' to start capturing images for each class

4. Process the dataset and train the model:
```bash
python create_dataset.py
python train_classifier.py
```

## 💻 Usage

1. To run the ASL recognition system:
```bash
python inference_classifier.py
```

2. Hold your hand in front of the webcam to show ASL signs
3. The system will:
   - Display the detected hand landmarks
   - Show the recognized sign with its meaning
   - Provide audio feedback for the detected sign
4. Press 'q' to quit the application

## 🎯 Model Training

The system uses a Random Forest Classifier trained on hand landmark features extracted using MediaPipe. The training process includes:

1. **Data Collection**: Capturing images for each ASL sign (36 classes - A-Z and 0-9)
2. **Feature Extraction**: Processing images to extract hand landmark coordinates
3. **Model Training**: Using Random Forest Classifier with default parameters
4. **Evaluation**: Testing on a 20% holdout set

## 📊 Dataset

- 36 classes (26 alphabets + 10 numbers)
- 100 images per class
- Total dataset size: 3,600 images
- Images are processed to extract 21 hand landmarks (x, y coordinates)

## ⚙️ System Requirements

- Python 3.x
- Webcam
- Sufficient lighting for hand detection
- Minimum 4GB RAM recommended
- CPU with decent processing power for real-time inference

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Add support for more ASL signs
2. Improve model accuracy
3. Optimize real-time performance
4. Add support for sentence formation
5. Improve the UI/UX



## 🔍 Future Improvements

- [ ] Support for dynamic gestures
- [ ] Multiple hand detection
- [ ] Sentence formation support
- [ ] Web interface
- [ ] Mobile app version
- [ ] Support for other sign languages
- [ ] Improved accuracy for various lighting conditions
- [ ] GPU acceleration support

## 👥 Acknowledgments

- MediaPipe team for the hand landmark detection system
- OpenCV community for computer vision tools
- scikit-learn team for machine learning implementations
