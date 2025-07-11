🐱🐶 Cat vs Dog Classifier with CNN
This project demonstrates a deep learning-based image classification model to distinguish between images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

📌 Project Overview
The goal is to build and train a CNN model that can accurately classify images into two categories: Cat or Dog. The model is trained on the popular Kaggle Cats vs Dogs dataset, and preprocessing steps such as image resizing, normalization, and data augmentation are applied to improve generalization.

🧠 Model Architecture
The CNN is composed of the following layers:

Convolutional layers with ReLU activation - 3 convolution layer

MaxPooling layers

Dropout layers to reduce overfitting

Flatten layer

Fully connected Dense layers

Final output layer with sigmoid activation (binary classification)

📂 Dataset
Source: Kaggle Dogs vs Cats Dataset

Preprocessing includes:

Image resizing (150x150 pixels)

Data augmentation (rotation, zoom, flip, etc.)

Train-validation split

📊 Training Details
Loss Function: binary_crossentropy

Optimizer: adam

Metrics: accuracy

Epochs: 10–20 (configurable)

Batch size: 32 (default)


Install dependencies:
Download the dataset from Kaggle and place it in the appropriate directory (instructions provided in the notebook).


🧪 Results
Achieved validation accuracy of ~85–98% depending on the number of epochs and augmentation.
Confusion matrix and accuracy/loss curves are visualized.

✅ Dependencies
1.Python 3.x
2.TensorFlow / Keras
3.NumPy
4.Matplotlib
5.Scikit-learn
6.Jupyter Notebook
7.Google Colab



📸 Sample Prediction
After training, the notebook includes code to test the model on unseen images and visualize predictions.
