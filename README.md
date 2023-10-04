# Human-Activity-Recognition-using-MutliHead-Convolutional-Attention

## Table of Contents

- [Introduction](#introduction)
  - [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Quantization](#quantization)
- [Results](#results)
- [References](#references)

## Introduction

This repository implements a multi-head convolutional attention-based model for human activity recognition. The model employs attention mechanisms to improve recognition accuracy by focusing on informative regions in the input data. Further, the obtained model is Quantized for supporting IoT-based devices.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Arshdeep-Singh-01/Human-Activity-Recogonition-using-MutliHead-Convolutional-Attention.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Human-Activity-Recogonition-using-MutliHead-Convolutional-Attention
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Dataset

This task use the WISDM dataset which contains the acceleration values (x,y,z) and the corresponding activity, along with the temporal components
Dataset is freely avaliable at [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
Train Data: 80%
Test Data: 20%

## Model Architecture

The overview of the model architecture is as follows:
- 3-Head Convolutional Neural Network
- Concatination followed by Maxpooling
- 30-Head Convolutional Attention
- Fully connected dense layers

![Model Architecture](https://github.com/Arshdeep-Singh-01/Human-Activity-Recognition-using-MutliHead-Convolutional-Attention/blob/main/img/Architecture.png)

## Training
The model was trained over 200 epochs (20+ hours of computation)
- Optimizer: SGD(lr = 0.001)
- Loss: Cross Entropy Loss

## Quantization
The trained model was then Quantized using Post Training Dynamic Quantization
| Model | Size (MB)|
|-------|-----------|
| Original Model | 373.094 | 
| Quantized Model | 0.009 |

# Results:
| Activity         | Accuracy (Original Model) | Accuracy (Quantized Model) |
|-------------------|------------------|---------------------------|
| Walking           | 98.06%           | 97.86%                    |
| Jogging           | 99.82%           | 98.96%                    |
| Upstairs          | 92.87%           | 92.57%                    |
| Downstairs        | 98.5%            | 98.11%                    |
| Sitting           | 95.05%           | 93.85%                    |
| Standing          | 99.65%           | 98.97%                    |
| **Overall**       | **97.33%**       | **96.72%**                |


## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [A Novel IoT-Perceptive Human Activity Recognition (HAR) Approach Using Multihead Convolutional Attention](https://ieeexplore.ieee.org/document/8883222)
3. [Post Training Dynamic Quantization](https://pytorch.org/docs/stable/quantization.html#:~:text=Quantization%20is%20primarily%20a%20technique,model%20is%20converted%20to%20INT8.)

