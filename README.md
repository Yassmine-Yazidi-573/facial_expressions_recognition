# Facial Expression Recognition using CNN on FER2013

This project implements a **Facial Expression Recognition (FER)** system using the **FER2013** dataset. The model uses a **Convolutional Neural Network (CNN)** to classify facial images into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## 🧠 Model Overview

The neural network is built using **TensorFlow/Keras** and trained on grayscale images of faces (48x48 pixels). It is designed to recognize facial emotions with a deep learning architecture that includes convolutional, pooling, normalization, and dropout layers.

---

## 📁 Dataset

- **Source:** `fer2013.csv`
- **Size:** 35,887 images
- **Features:**
  - `emotion`: Emotion label (0–6)
  - `pixels`: Flattened grayscale pixel values
  - `Usage`: One of `Training`, `PublicTest`, or `PrivateTest`

---

## ⚙️ Preprocessing

1. **Pixels** are converted into 48x48 grayscale image arrays.
2. Images are normalized (scaled between 0 and 1).
3. Emotion labels are one-hot encoded.
4. Dataset is split into:
   - `Training`: Model fitting
   - `PublicTest`: Validation
   - `PrivateTest`: Testing

---

## 🏗️ Model Architecture

```text
Input: 48x48x1 grayscale image

Conv2D (32 filters, 3x3) → BatchNorm → MaxPooling → Dropout
Conv2D (64 filters, 5x5) → BatchNorm → MaxPooling → Dropout
Flatten
Dense (128 units, ReLU) → Dropout
Dense (7 units, Softmax)
```

- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## 🏋️ Training

- **Epochs:** 30
- **Batch Size:** 128
- **Callbacks:**
  - `EarlyStopping`: Stops if no improvement in validation loss
  - `ModelCheckpoint`: Saves best model weights based on validation accuracy

---

## 📉 Evaluation

```python
test_loss, test_accuracy = model.evaluate(pic_test, label_test)
```

- **Test Accuracy:** ~15% (may vary, room for improvement through tuning or augmentation)

---

## 📊 Training History

Visualizations of accuracy and loss over epochs for both training and validation sets.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

---

## 🔍 Making Predictions

```python
import cv2
new_image = cv2.imread('sad_face.jpeg')
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_image = cv2.resize(new_image, (48, 48))
new_image = new_image / 255.0
new_image = new_image.reshape(1, 48, 48, 1)
prediction = model.predict(new_image)
```

- Predicted label is mapped back to:
  ```python
  emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  ```

---

## ✅ Requirements

```bash
tensorflow
numpy
pandas
matplotlib
opencv-python
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```

---

## 🚀 Future Improvements

- Data Augmentation to improve generalization
- Hyperparameter tuning
- More complex architectures (e.g., ResNet, EfficientNet)
- Real-time webcam emotion detection

---
