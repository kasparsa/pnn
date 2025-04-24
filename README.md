# PNN
C/C++ Artificial Neural Network from Scratch

This code implements a Feedforward Neural Network (FNN) with backpropagation for training, using various optimization algorithms and activation functions. Here's a summary of its key components and characteristics:

## 1. Architecture
- A multi-layer perceptron (MLP) with multiple layers, where each layer consists of neurons connected to the previous layer.
- Processes input data through a feedforward mechanism, computing weighted sums and applying activation functions.

## 2. Activation Functions
- Supports multiple activation functions: Sigmoid, Tanh, Gaussian, and Sine, applied to the weighted sum of inputs for each neuron.
- Derivatives of these functions are computed for backpropagation.

## 3. Training Process
- **Feedforward Phase**: Input data is passed through the layers, computing weighted sums and applying activation functions to produce outputs.
- **Error Calculation**: Computes the Mean Squared Error (MSE) by comparing the network's output to the target output for each sample.
- **Backpropagation**: Calculates gradients (deltas) for each neuron, starting from the output layer and moving backward to update weights.
- **Weight Updates**: Adjusts weights using gradients and various optimization algorithms.

## 4. Optimization Algorithms
- Supports multiple optimization methods for weight updates:
  - **RProp+**: Resilient Propagation with sign-based updates.
  - **RMSprop**: Adaptive learning rate based on moving average of squared gradients.
  - **Adam**: Combines momentum and RMSprop with bias correction.
  - **Adagrad with Momentum**: Adapts learning rate based on accumulated gradients.
  - **iRProp+**: Improved RProp with momentum.
- Includes L2 regularization (weight decay) to prevent overfitting.

## 5. Parallelization
- Uses OpenMP for parallel processing (e.g., across samples, neurons, or weights) to improve performance, with 16 threads specified.

## 6. Error Handling
- Throws errors for unrecognized activation functions or derivatives.

## Summary
This is a fully connected feedforward neural network trained using backpropagation with support for multiple activation functions, optimization algorithms, and parallel processing. It is designed for supervised learning tasks, calculating MSE as the loss function and optimizing weights to minimize it.
# Feedforward Neural Network with Backpropagation

This repository contains an implementation of a **Feedforward Neural Network (FNN)** with backpropagation, a versatile machine learning model designed for supervised learning tasks. Its flexibility in activation functions, optimization algorithms, and parallel processing makes it suitable for a wide range of applications.

## Primary Use Cases

### 1. Classification Tasks
- **Pattern Recognition**: Classify input data into categories or labels, such as:
  - Image classification (e.g., handwritten digit recognition like MNIST).
  - Email filtering (spam vs. non-spam).
  - Medical diagnosis (disease vs. healthy based on patient data).
- **Sentiment Analysis**: Classify text data, such as determining whether a product review is positive, negative, or neutral.
- **Fraud Detection**: Identify fraudulent transactions by classifying patterns in financial data.

### 2. Regression Tasks
- **Predictive Modeling**: Predict continuous values, such as:
  - House prices based on features like size and location.
  - Stock prices or energy consumption.
- **Time Series Forecasting**: Estimate future values in time-dependent data, such as weather predictions or sales trends.

### 3. Function Approximation
- **Non-linear Mapping**: Learn complex relationships between inputs and outputs, such as modeling physical systems (e.g., control systems in robotics) or simulating mathematical functions.
- **Data Fitting**: Fit curves or surfaces to noisy data in scientific or engineering applications.

### 4. Pattern Recognition and Feature Learning
- **Signal Processing**: Analyze and classify signals, such as speech recognition or audio classification.
- **Image Processing**: Perform tasks like edge detection, object recognition, or pixel-wise classification in simpler image processing pipelines.

### 5. Control Systems
- **Robotics and Automation**: Learn control policies for robotic movements or autonomous systems, such as adjusting motor outputs based on sensor inputs.
- **Game AI**: Implement decision-making systems for game agents, such as predicting optimal moves in strategy games.

### 6. Anomaly Detection
- **Network Security**: Detect unusual patterns in network traffic to identify potential cyber threats or intrusions.
- **Industrial Monitoring**: Identify defective components or machinery failures by learning normal operational patterns and flagging deviations.

### 7. Scientific Research and Simulations
- **Physics and Chemistry**: Model complex systems, such as predicting molecular interactions or simulating physical phenomena.
- **Bioinformatics**: Analyze biological data, such as predicting protein structures or classifying gene expressions.

### 8. Education and Prototyping
- **Teaching Tool**: Demonstrate neural network concepts like backpropagation, activation functions, and optimization algorithms in academic settings.
- **Research Prototyping**: Test new activation functions, optimization techniques, or regularization methods in a controlled environment.

### 9. Embedded Systems and Optimization
- **Resource-Constrained Environments**: With parallel processing (e.g., OpenMP), the code can be optimized for performance in embedded systems or IoT devices for tasks like real-time classification or regression.
- **Custom Hardware**: Adapt the network for specific hardware (e.g., GPUs or FPGAs) to perform tasks like sensor data processing in autonomous vehicles.

## Why This Implementation?

- **Multiple Activation Functions**: Experiment with Sigmoid, Tanh, Gaussian, and Sine to suit tasks requiring specific non-linearities.
- **Optimization Flexibility**: Supports RProp+, RMSprop, Adam, Adagrad, and iRProp+ for tailoring to different datasets and convergence needs.
- **Parallel Processing**: Leverages OpenMP for efficiency with large datasets or real-time applications.
- **Regularization**: Includes L2 weight decay to prevent overfitting, improving generalization for real-world noisy data.

## Limitations and Considerations

- **Simpler Architecture**: FNNs are less suited for sequential data (e.g., time series or natural language) compared to recurrent neural networks (RNNs) or transformers.
- **Scalability**: For very large datasets or complex tasks (e.g., deep learning for image recognition), advanced architectures like convolutional neural networks (CNNs) or frameworks like TensorFlow/PyTorch may be preferred.
- **Manual Tuning**: Requires manual configuration of hyperparameters (e.g., learning rate, regularization strength), which may limit use in fully automated systems.

## Summary

This FNN implementation is ideal for **small-to-medium-scale supervised learning tasks**, **prototyping**, **educational purposes**, and applications requiring **custom optimization** or **parallel processing**. It excels in domains like classification, regression, and pattern recognition where fully connected networks are sufficient.

---

*Contributions and feedback are welcome!*
