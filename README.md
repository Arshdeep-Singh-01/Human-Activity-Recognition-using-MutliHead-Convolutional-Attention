# Human-Activity-Recogonition-using-MutliHead-Convolutional-Attention

## Table of Contents

- [Introduction](#introduction)
  - [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [References](#references)
- [License](#license)

## Introduction

This repository implements a multi-head convolutional attention-based model for human activity recognition. The model employs attention mechanisms to improve recognition accuracy by focusing on informative regions in the input data.



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

## Training
The model was trained over 200 epochs (20+ hours of computation)
- Optimizer: SGD(lr = 0.001)
- Loss: Cross Entropy Loss

# Results:
Overall Accuracy: 98.33%

* Accuracy of Walking: 98.06%
* Accuracy of Jogging: 99.82%
* Accuracy of Upstairs: 92.87%
* Accuracy of Downstairs: 98.5%
* Accuracy of Sitting: 95.05%
* Accuracy of Standing: 99.65%

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [A Novel IoT-Perceptive Human Activity Recognition (HAR) Approach Using Multihead Convolutional Attention](https://ieeexplore.ieee.org/document/8883222)
3. [ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## License

State the license under which the code is distributed. This could be an open-source license like MIT, GPL, or any other license that fits your project.
```
