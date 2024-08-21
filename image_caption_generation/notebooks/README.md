# Image Caption Generation

## Introduction to Image Captioning

Image captioning is the task of generating descriptive text for images. Initially, it was tackled using rule-based and retrieval methods, but these were limited and inefficient.

With the rise of deep learning, image captioning has seen a major shift. Modern methods use Convolutional Neural Networks (CNNs) to extract image features and Recurrent Neural Networks (RNNs) to generate text, leading to more accurate and flexible results.

This repository focuses on these advanced neural approaches to image captioning.

All the models were trained on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset which contains 8,092 images that are each paired with five different captions.

Below is a table addressing some common data and optimization related parameters.

| Parameter      |       Value        |
| -------------- |:------------------:|
| Training Set   |    29000/8092     |
| Testing Set    |     1000/8092     |
| Validation Set |     1014/8092     |
| Loss Function  | Cross Entropy Loss |
| Optimizer      |       AdamW        |

## Architectures

### 1. [Long Short-Term Memory Networks](https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/notebooks/Seq2Seq.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QaoSKUbLy4ViHnJsl3m3H6xEemDdowkL?usp=sharing)
LSTMs are used in image captioning to generate sequences of words that describe an image. The process starts by extracting features from the image using a CNN. These features are then fed into an LSTM, which predicts the next word in the caption sequence at each time step. The LSTM continues generating words until it outputs an end-of-sentence token, producing a complete caption that accurately reflects the image's content. LSTMs are effective in this task due to their ability to retain context over sequences, making them ideal for generating coherent and contextually relevant captions.

### 4. [Transformer Architecture](https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/notebooks/Attention_Is_All_You_Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RlDhclIlJWzcFC0iPqwbiFEuiCbYIJBG?usp=sharing)
The encoder extracts visual features from the image using a convolutional neural network (CNN). These features are then input to a transformer decoder, which generates captions word by word. The transformer decoder leverages self-attention mechanisms to connect the current word to previously generated words and to incorporate the visual context from the CNN. During training, the model learns to predict the next word in the sequence by minimizing the difference between the predicted and actual captions. During inference, the decoder sequentially generates the full caption until it produces an end-of-sentence token. This approach effectively merges the visual feature extraction strengths of CNNs with the sequence modeling capabilities of transformer decoders, resulting in coherent and contextually accurate captions.

## Summary
Below is a table, summarising the number of parameters and the BLEU scores achieved by each architecture.

| Architecture                        | No. of Trainable Parameters | BLEU Score |
| ----------------------------------- |:---------------------------:|:----------:|
| LSTM                                |         13,899,013          |   18.94    |
| Transformer                         |         20,518,917          |   31.24    |

<ins>**Note:**</ins>
1. The above BLEU scores may vary slightly upon training the models (even with fixed SEED).

## Plots
<p align="center">
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq_with_Attention.jpeg?raw=true"/> 
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Conv_Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Transformer.jpeg?raw=true"/>
</p>

### Reference(s):
* [PyTorch ImgCaptioning by Lalu Erfandi Maula Yusnu](https://github.com/nunenuh/imgcap.pytorch/blob/main/icap/data.py)
