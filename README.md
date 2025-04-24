# pnn
C/C++ artifical neural network from scratch
This code implements a Feedforward Neural Network (FNN) with backpropagation for training, using various optimization algorithms and activation functions. Here's a summary of its key components and characteristics:
    1. Architecture: 
        ◦ A multi-layer perceptron (MLP) with multiple layers, where each layer consists of neurons connected to the previous layer.
        ◦ Processes input data through a feedforward mechanism, computing weighted sums and applying activation functions.
    2. Activation Functions:
        ◦ Supports multiple activation functions: Sigmoid, Tanh, Gaussian, and Sine, applied to the weighted sum of inputs for each neuron.
        ◦ Derivatives of these functions are computed for backpropagation.
    3. Training Process:
        ◦ Feedforward Phase: Input data is passed through the layers, computing weighted sums and applying activation functions to produce outputs.
        ◦ Error Calculation: Computes the Mean Squared Error (MSE) by comparing the network's output to the target output for each sample.
        ◦ Backpropagation: Calculates gradients (deltas) for each neuron, starting from the output layer and moving backward to update weights.
        ◦ Weight Updates: Adjusts weights using gradients and various optimization algorithms.
    4. Optimization Algorithms:
        ◦ Supports multiple optimization methods for weight updates:
            ▪ RProp+: Resilient Propagation with sign-based updates.
            ▪ RMSprop: Adaptive learning rate based on moving average of squared gradients.
            ▪ Adam: Combines momentum and RMSprop with bias correction.
            ▪ Adagrad with Momentum: Adapts learning rate based on accumulated gradients.
            ▪ iRProp+: Improved RProp with momentum.
        ◦ Includes L2 regularization (weight decay) to prevent overfitting.
    5. Parallelization:
        ◦ Uses OpenMP for parallel processing (e.g., across samples, neurons, or weights) to improve performance, with 16 threads specified.
    6. Error Handling:
        ◦ Throws errors for unrecognized activation functions or derivatives.
In summary, this is a fully connected feedforward neural network trained using backpropagation with support for multiple activation functions, optimization algorithms, and parallel processing. It is designed for supervised learning tasks, calculating MSE as the loss function and optimizing weights to minimize it.
