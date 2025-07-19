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
