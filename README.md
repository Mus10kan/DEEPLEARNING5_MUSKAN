Image Captioning Model (Model 5)

Overview
This project implements an image captioning model using deep learning. It combines image features extracted from a CNN with text sequences processed by an LSTM to generate captions.

Model Architecture
Image feature input (4096-dim vector)
Dense layer for feature processing
Text input processed using Embedding + LSTM
Fusion of image and text features
Dense + Softmax layer for word prediction

Requirements
TensorFlow / Keras , NumPy , Pandas , Training , Custom data generator used for efficient training , Sparse categorical crossentropy loss,
Adam optimizer , Right-padded sequences

Usage
Prepare dataset (image features + captions)
Tokenize captions
Train model using generator
Save trained model

Output
The model predicts the next word in a caption sequence given an image and partial caption.

Notes
LSTM is run with use_cudnn=False to avoid masking issues
Padding is applied using padding='post'
