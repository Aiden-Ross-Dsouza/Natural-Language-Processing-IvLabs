# Image Caption Generation

## Introduction to Image Captioning

Image captioning is the task of generating descriptive text for images. Early image captioning methods used template-based approaches, keyword mappings, and bag-of-words models, often resulting in rigid captions. Semantic models applied handcrafted rules to interpret object interactions, while probabilistic graphical models aimed to link visual features with text. Despite these innovations, captions frequently lacked depth and context before the advent of deep learning.

This repository focuses on these advanced neural approaches to image captioning.

All the models were trained on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset which contains 8,092 images that are each paired with five different captions.

Below is a table addressing some common data and optimization related parameters.

| Parameter      |       Value        |
| -------------- |:------------------:|
| Training Set   |    5665/8092     |
| Testing Set    |     809/8092     |
| Validation Set |     1618/8092     |
| Loss Function  | Cross Entropy Loss |
| Optimizer      |       AdamW        |

## Architectures

### 1. [Long Short-Term Memory Networks](https://github.com/Aiden-Ross-Dsouza/Natural-Language-Processing-IvLabs/blob/6857632075b374c98dec4e33e0c7a45e513f200d/image_caption_generation/notebooks/Image_Captioning_LSTM.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wpc36DjWBB_aHNZtCVU7BnloP-lCf962?usp=sharing)
LSTMs are used in image captioning to generate sequences of words that describe an image. The process starts by extracting features from the image using a CNN. These features are then fed into an LSTM, which predicts the next word in the caption sequence at each time step. The LSTM continues generating words until it outputs an end-of-sentence token, producing a complete caption that accurately reflects the image's content. LSTMs are effective in this task due to their ability to retain context over sequences, making them ideal for generating coherent and contextually relevant captions.

### 4. [Transformer Architecture](https://github.com/Aiden-Ross-Dsouza/Natural-Language-Processing-IvLabs/blob/6857632075b374c98dec4e33e0c7a45e513f200d/image_caption_generation/notebooks/Image_Captioning_using_transformers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e7z1hDSe0fYEZDwqeMjDoB1e9zZTDU_q?usp=sharing)
The encoder extracts visual features from the image using a convolutional neural network (CNN). These features are then input to a transformer decoder, which generates captions word by word. The transformer decoder leverages self-attention mechanisms to connect the current word to previously generated words and to incorporate the visual context from the CNN. During training, the model learns to predict the next word in the sequence by minimizing the difference between the predicted and actual captions. During inference, the decoder sequentially generates the full caption until it produces an end-of-sentence token. This approach effectively merges the visual feature extraction strengths of CNNs with the sequence modeling capabilities of transformer decoders, resulting in coherent and contextually accurate captions.

## Plots
Training Loss <br>
![Screenshot 2024-09-10 113826](https://github.com/user-attachments/assets/7e3afa9f-9c35-4c0c-8626-afc2b745d747)


### Reference(s):
* [PyTorch ImgCaptioning by Lalu Erfandi Maula Yusnu](https://github.com/nunenuh/imgcap.pytorch/blob/main/icap/data.py)
