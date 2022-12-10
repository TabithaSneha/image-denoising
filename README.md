# Image Denoising

***

In this project, the architecture of a Denoising Autoencoder has been studied and implemented using the CIFAR-10 image dataset.

## Introduction:-

### Autoencoders:

Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation.

It consists of two networks:

* *Encoder network*: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size.
* *Decoder network*: The decoder network recovers the data from the code, likely with larger and larger output layers.

<img src="https://i.imgur.com/E3iZ1a2.png" height="350" width="600" >
Image Source: Lilian Weng Blog

The difference between the two vectors- the input *x* and the reconstructed input *x'* is calculated using the Mean Square Error loss function.

<img src="https://i.imgur.com/WRuQCOH.png" height="200" width="500" >

### Denoising Autoencoders:

Since the autoencoder learns the identity function, we are facing the risk of overfitting when there are more network parameters than the number of data points.

To avoid overfitting and improve the robustness, Denoising Autoencoder was proposed as a modification to the basic autoencoder. The input is partially corrupted by adding noises (i.e. masking noise, Gaussian noise, salt-and-pepper noise, etc.) to the input vector in a stochastic manner. Then the model is trained to recover the original input from the noisy input.

<img src="https://i.imgur.com/TKk2EJB.png" height="400" width="650" >
Image Source: Lilian Weng Blog

<img src="https://i.imgur.com/ZbqTDun.png" height="200" width="500" >

## Methodology:-

### Libraries used:

* PyTorch 3.11
* Matplotlib
* Numpy
* Pandas

### Dataset used:

The CIFAR-10 dataset has been used which is most commonly seen in training machine learning and computer vision algorithms. It  contains 60,000 32x32 RGB color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

### Structure of Code:

#### Obtaining Dataset and Displaying Images:-
* Importing the required libraries given above.
* Defining the hyperparameters and transforms for the dataset. Resized the input image dimensions from 32x32 to 28x28.
* Downloading the CIFAR-10 dataset from PyTorch and loading the train and test data into dataloaders.
* Creating a function to display one batch of training images as 'Original Images' and adding Gaussian noise to it to obtain 'Noisy Images'.

#### Defining the Denoising Autoencoder Architecture:-
* Defining functions *in_dim* and *out_dim* to determine the output dimensions from Convolution and Transposed Convolution.
* Creating classes for *Encoder* and *Decoder* which uses the nn.Sequential function for adding the various layers of the network.
* Defining the *MSELoss* loss function and initializing the *Adam* optimizer.
* Creating a function for returning noisy signals by the addition of stochastic noise sampled from a normal Gaussian distribution to the image dataset.

#### Training the Model and Generating the Reconstructed Image:-
* Defining the training function for both the Encoder and the Decoder with forward pass and backpropagation.
* Defining the testing function with the Encoder and Decoder data and evaluating the loss using pairwise distance function.
* Iterating over the training and testing functions and printing the validation loss.
* Plotting the 'Original Image', 'Noisy Image' and 'Reconstructed Image' and the 'Loss curve'.
* Finally saving and loading the model.

### Denoising Autoencoder Architecture:

The Encoder model consists of 4 Convolutional Layers that takes an input image dimension of 28x28 and returns an output of dimension 7x7.
The Decoder model consists of 4 Deconvolutional Layers that takes the input dimension of 7x7 and outputs 28x28 again.

![Pic](https://i.imgur.com/HQIALcN.png)

### Parameters used:

| Parameters| Values| 
| -------- | -------- | 
| Optimizer     | Adam     |
| Batch Size     | 64     |
| Epochs     | 250     |
| Learning Rate     | 0.001     |
| Weight Decay     | 1e-5     |

## Results:-

### Output Images:-

![Pic](https://i.imgur.com/sMTtzYs.png)

![Pic](https://i.imgur.com/KbuN6zp.png)

### Plot of Loss vs Epochs:-

![Pic](https://i.imgur.com/IYi3yOM.png)

Train Loss: 0.140

| Dataset | Score |
| -------- | -------- |
| Training     | 1.154085|
| Validation     | 1.757395|

## References:-

* [From Autoencoder to Beta-VAE: Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)
* [Denoising Autoencoder in Pytorch: Eugenia Anello](https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e)
