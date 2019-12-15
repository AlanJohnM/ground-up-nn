# Ground-Up Neural Network

## Why did I want to build this?

Machine Learning is often considered a gateway to the future. With the sorts of promises it offers, people rush to build the latest and greatest machine learning models through powerful resources like TensorFlow or PyTorch. Unfortunately, this is often without taking the time to fully understand what is going on behind the scenes. I personally aim to always understand what I'm building, so I figured the best way to do that is build it from scratch, with the only external library being for linear algebra. The choice of C++ breaks away from the usual use of Python for machine learning projects, but this choice forces me to implement every little detail of the project, and gives clear performance benefits with large datasets.

## What exactly is it?

This is a digit recognition neural network using the MNIST database of handwritten digits. It is the classic example of the foundations of machine learning. I wrote a custom parser for the files containing the MNIST data, which is then fed through the neural network for training. The network uses Stochastic Gradient Descent with a variable learning rate, the  Sigmoid function for activation, and Mean Squared Error (MES) for a loss function. These functions, along with the architecture of the network are easily adjustable.

## What's next?

I will use this as a boilerplate for my continued research about Machine Learning. I currently hope to re-implement this network as a Convolutional Neural Network (CNN) that specializes in computer vision. This will require more computing power to train, so I will also need to modify it to allow for parallel computing through GPUs or large scale multi-threading. 
