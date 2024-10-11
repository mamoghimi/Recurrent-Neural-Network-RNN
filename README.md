# Recurrent Neural Network (RNN) for Time Series Prediction with Sine and Complex Waveforms

This project demonstrates the use of a Recurrent Neural Network (RNN) for predicting time series data. The network is trained on two different types of waveforms to understand the RNN's ability to learn sequential patterns and make future predictions based on past data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
  
## Overview
The goal of this project is to explore how an RNN can be used for time series prediction. The RNN is tested on two different prediction tasks:
1. **Predicting a Cosine Wave from a Sine Wave:** The network is trained to learn the phase-shifted relationship between sine and cosine functions.
2. **Predicting a Complex Function from a Sine Wave:** The network learns to predict a more complex target function that combines multiple periodic and exponential components.

Both tasks demonstrate the RNN's ability to handle sequential data, with the results visualized to show the network's learning progress over time.

## Features
- 🌀 **Two types of time series predictions:** Sine-to-cosine mapping and sine-to-complex function mapping.
- 🎨 **Interactive visualization:** Real-time plotting of the RNN's predictions during training to observe how the model improves.
- ⚙️ **Configurable hyperparameters:** Adjustable learning rate, number of epochs, time steps, and more.
- 📈 **Evaluation with accuracy and loss metrics:** The project tracks both loss and accuracy during training, with accuracy measured based on a tolerance level for close predictions.

## Installation
To run this project, you need the following dependencies:
- `Python 3.x`
- `PyTorch`
- `matplotlib`
- `numpy`

Install the required libraries using:
```bash
pip install torch matplotlib numpy
```

# Usage
To run the project, follow these steps:

- Clone the repository:
```bash
git clone https://github.com/your-username/rnn-time-series-prediction.git
cd rnn-time-series-prediction
```
- Run the script for sine to cosine prediction:
```bash
python sine_to_cosine_prediction.py
```
- Run the script for sine to complex function prediction:
```bash
python sine_to_complex_prediction.py
```
The scripts will open an interactive plot that shows the RNN's prediction results during training.

# Examples
- Sine to Cosine Prediction


https://github.com/user-attachments/assets/fdcc4af8-5f0e-4b89-b7fa-0a6c39cb1785



- Sine to Complex Function Prediction



https://github.com/user-attachments/assets/22ff011e-30dd-4289-b6b8-e8e7c038de78




