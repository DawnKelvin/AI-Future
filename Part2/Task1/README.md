# AI-Future

# Trash Classification Project

## Project Purpose
This project aims to build a lightweight convolutional neural network (CNN) to classify images of different types of trash and recyclable materials. The goal is to create a model that can be deployed on resource-constrained devices like the Raspberry Pi for real-time trash sorting applications.

## Dataset
The project utilizes the **TrashNet** dataset, an open-source collection of approximately 2,500 images categorized into six classes: glass, paper, cardboard, plastic, metal, and trash. The dataset was created by Stanford students and is commonly used for lightweight waste classification tasks. The dataset was accessed from its GitHub repository: https://github.com/garythung/trashnet.git.

## Model Architecture
A lightweight CNN model was designed and implemented using TensorFlow/Keras. The architecture consists of:
-   Several Convolutional layers (`Conv2D`) with ReLU activation for feature extraction.
-   MaxPooling layers (`MaxPooling2D`) to reduce spatial dimensions and computational complexity.
-   A Flatten layer to convert the 2D feature maps into a 1D vector.
-   Dense layers with ReLU activation for learning higher-level representations.
-   An output Dense layer with a softmax activation function for multi-class classification, producing probability distributions over the six classes.

The model was compiled with the Adam optimizer and categorical crossentropy loss.

## Results
The model was trained for 5 epochs on the training data and evaluated on the validation set.
-   **Validation Accuracy:** Approximately 47.92%
-   **Sample Test Result:** When tested on a sample plastic image, the model predicted the class as "paper" with a probability of approximately 0.464. This indicates that further training or model refinement is needed to improve accuracy.

The trained model was converted to the TensorFlow Lite format (`recyclable_classifier.tflite`) for potential deployment on edge devices.
"""

with open('README.md', 'w') as f:
    f.write(readme_content)

## Deployment Steps
Here’s a **complete breakdown of deployment steps** for lightweight image classification model ( recyclable item detector) to run on a **Raspberry Pi** after training and converting it on **Google Colab**:

---

## ✅ Step-by-Step Deployment Instructions

---

### 1. 🎓 **Train & Export the Model on Colab**

### 2. 💾 **Transfer `.tflite` Model & Labels to Raspberry Pi**

#### 📁 Files to transfer:

* `model.tflite`
* `labels.txt` – list of class names (e.g., `paper`, `plastic`, ...)

#### 🔄 Options to transfer files:

* **Via SCP** (from your PC/mac):

```bash
scp model.tflite pi@<raspberry_pi_ip>:/home/pi/
```

* **Google Drive → Download** on Raspberry Pi
* **USB Drive** or SD card

---

### 3. ⚙️ **Set Up TensorFlow Lite Runtime on Raspberry Pi**

> TensorFlow is too heavy for Pi; use `tflite-runtime`.

```bash
# For Raspberry Pi OS 32-bit (Python 3.7+)
pip install tflite-runtime
```

For other versions:
Use the correct wheel file from:
📦 [https://www.tensorflow.org/lite/guide/python#install\_tensorflow\_lite\_for\_python](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python)

---

### 4. 📷 **Capture or Load Image for Classification**

#### Option A: Load static image

```python
from PIL import Image
img = Image.open("test.jpg").resize((128, 128))
```

#### Option B: Capture image via Pi Camera

```bash
sudo apt install python3-picamera
```

```python
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.start_preview()
sleep(3)  # Let camera warm up
camera.capture('/home/pi/test.jpg')
camera.stop_preview()
```

---

### 5. 🧠 **Run TFLite Inference Code on Pi**

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Load and preprocess image
img = Image.open("test.jpg").resize((128, 128))
img = np.expand_dims(np.array(img).astype(np.float32) / 255.0, axis=0)

# Set input and run inference
input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_idx, img)
interpreter.invoke()

# Get and decode output
output = interpreter.get_tensor(output_idx)
pred = np.argmax(output)
with open("labels.txt") as f:
    labels = f.read().splitlines()
print(f"Prediction: {labels[pred]} ({output[0][pred]*100:.2f}%)")
```

---

### 6. 🎯 **(Optional) Enhance with GUI or Voice**

* Show result on a small touchscreen (e.g., PiTFT)
* Add `text-to-speech`:

```bash
sudo apt install espeak
espeak "This is plastic"
```

---

## 🧪 Optional Extras

* Batch classify multiple images
* Connect to web API (e.g., log classifications)
* Add confidence thresholding
* Add LEDs or buzzer for feedback

---
## Rasperry Pi .py file ready to run
A ready-to-run Raspberry Pi image classification project is packaged into a ZIP file recyclable_classifier_pi.zip attached to this project files.

The Contents of the zip file are:

1. classify_recyclable.py – main inference script

2. labels.txt – class names

3. model.tflite file
