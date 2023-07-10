# MNIST with a Twist

This repository presents a unique approach towards classifying the classic MNIST dataset using a custom neural network implemented in PyTorch. This network incorporates custom weight initialization, custom layers, custom activation function, and custom loss function to improve upon the vanilla implementation.

## Features

1. **Custom Weight Initialization** : A method that samples random values from Beta Distribution and uses them as initial weights.
   
   The weight initialization function, `beta_init`, will follow the equation:

   ![equation](https://latex.codecogs.com/svg.latex?\color{white}w_i%20%5Csim%20Beta(%5Calpha%2C%20%5Cbeta)%20%5Cforall%20i%20%5Cin%20%5B1%2C%20n%5D)

   where `w_i` are the weights, `n` is the total number of weights, and `Beta` represents the Beta Distribution with shape parameters `\alpha` and `\beta`.

2. **Custom Neural Network Layer** : A layer performing a unique operation as shown in the class `CustomLayer`.
   The operation this layer performs can be represented as:

   ![equation](https://latex.codecogs.com/svg.latex?\color{white}O_i%20=%20max(w_%7Bj,i%7Dx_j%20+%20b_i)%20%5Cforall%20j%20%5Cin%20%5B1%2C%20n%5D)

   where `O_i` is the output of the ith neuron, `w_{j,i}` represents the weight from the jth neuron of the previous layer to the ith neuron of the current layer, `x_j` is the jth input, `b_i` is the bias for the ith neuron, and `n` is the total number of inputs.

3. **Custom Activation Function** : A unique activation function `CustomActivation` combining three common activation functions (`tanh`, `sigmoid`, `relu`) controlled by trainable parameters.
   The operation this function performs can be represented as:

   ![equation](https://latex.codecogs.com/svg.latex?\color{white}y%20=%20%5Calpha_1%20%5Ccdot%20tanh(x)%20+%20%5Calpha_2%20%5Ccdot%20sigmoid(x)%20+%20%5Calpha_3%20%5Ccdot%20relu(x))

   where `y` is the output, `x` is the input, `\alpha_1`, `\alpha_2`, and `\alpha_3` are trainable parameters, and `tanh`, `sigmoid`, and `relu` are the respective activation functions.
   
4. **Custom Loss Function** : A custom loss function `CustomLoss` designed to work well with the unique aspects of the neural network.
   The loss function can be represented as:

   ![equation](https://latex.codecogs.com/svg.latex?\color{white}L%20=%20%5Csum_%7Bi=1%7D%5E%7Bn%7D%20(y_i%20%5Ccdot%20log(%5Cfrac%7Byhat_i%7D%7By_i%7D)))

   where `L` is the loss, `yhat_i` is the true probability and `y_i` is the predicted probability for the ith instance, and `n` is the total number of instances.

5. **Dense Neural Network** : The neural network is built using the custom objects defined above in conjunction with dense layers. It is trained on the MNIST dataset.

## Results

After training the custom neural network for `num_epochs = 10`, `batch_size = 100`, and `learning_rate = 0.001`, the model achieved an accuracy of `96.91%`.

## Usage

Please refer to the [Jupyter notebook](./MNIST_with_a_Twist.ipynb) for more information on how to use these custom classes and the specific implementation details.
