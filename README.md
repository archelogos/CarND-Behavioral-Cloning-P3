# Udacity SDCN - Project 3 - Behavioral Cloning

## Summary

Thanks to the Udacity Simulator we have the opportunity to generate a large amount of data
that clones a real self-driving car behavior. Taking advantage of this we are driving the car
along the track to generate not only a lot of data but also a good-quality one.

As we know, there are several ML techniques that can be applied to the same problem. In this
case we are going to go straightaway and work in a Regression Problem building a ConvNet.
However, other methods could be suggested and probably fit better (i.e. Q-Learning).

In general, the key point when facing a ML problem, is to take advantage of all of these techniques and not just be focused just on the first happy idea.

Said that, the goal of this project is related to analyze the data and it's nature and
design a stable NN model that allows the car to drive itself at least 1 lap successfully.

## Data

The generated data of the Simulator are mainly images from the 3 cameras placed in front of the car and
some parameters related to the car driving. In this project, the model only knows about
the steering angle and it doesn't care about the others variables. Once the model
is trained and it properly predicts an output from a random input, that output is used to drive the car.

The throttle is always a constant in the driving simulation and the steering angle comes from the model.

To train the model it's necessary to use the images from the cameras as inputs, so basically our model makes predictions of angles based on images.

As it was mentioned before, it could be a good idea to not only use a ConvNet model to create a "driving agent".
Furthermore we can assure that is not even a good idea to just use resized images to create it.

Instead of that, it could be better to think in detect the edges from the images and combine that with the learning
from the real images.

The images have dimensions of 320x160 images (in color). Getting some pieces of advice from the forum and following the instructions from the paper, there are some important concepts to take into account before start the adventure.

- It's a good idea to resize the images to train our model faster.
- Our model is going to be overfitted in any cases (almost always). We should apply Dropout to make our model generalizes better.
- It's a good recommendation and perfectly approachable to augment some of our images in the training phase.
- The NVIDIA ConvNet works perfectly and its a very good starting point.
- Feed our model from a Python Generator is highly recommended.

Curve examples

![Curve1](imgs/curve1.png?raw=true)] ![Curve2](imgs/curve2.png?raw=true)]

Recovery examples

![Recovery1](imgs/recovery1.png?raw=true)] ![Recover2](imgs/recovery2.png?raw=true)]

## Preprocessing

Keeping the previous points in mind it was built a very easy preprocessing stage where
the only thing that the script did is to open the images, resize them and flip some of them.

There is no grayscaling processing, cropping or YUV transformation.

Once the data has been generated, the mod_csv script changes the name of the
images in the csv. And then the model.py script can be executed.

Getting a qualify data is one of the most important points of this project. It seems pretty obvious that
if the car is not well driven the model won't be able to drive properly by its own.
It's necessary to be careful when driving the car and ensure that sudden movements aren't done. It is possible to extract these movements from the dataset applying an Outliers Detection phase but is not
quite easy to do that.

## Model

In the first attempts, the model was based on a 2CNNs + 3FCCNs Layer. With that, the model converged
fast, but the car went out of the track in the first curve. The model wasn't good enough and it didn't fit this problem
in a good way so it was decided to based the model in the mentioned NVIDIA one.

![NVIDIA model](imgs/model.png?raw=true)

- 5 CNNs (24, 35, 48, 64, 64)
- 5x5 kernel in the three first ones and 3x3 in the others
- x2 subsampling in the three first one and no subsampling in the others
- 3 FCCNs (100, 50, 10) and final output neuron
- All the activation functions are RELUs
- There is also a dropout (40% rate) before the FCCNs layers to help to decrease the overfitting
- It was also set a Regularization Phase at the beginning of the model

![Summary](imgs/summary.png?raw=true)

## Training

As it happened in the previous project the optimizer choice can drastically
change the way that the model learns and converges.

It's surprising how the Adam Optimizer performs perfectly in these Images-Classification problems and
it's a good idea to go deeper and analyze why this is so important in these cases.

The SGD algorithm was tried in the first attempts just to check if it could handle the training as Adam does but
it was rapidly discarded.

The settings of the training phase were.

- 5 epochs (not needed to set more, it really didn't help)
- 100 batch_size

Once the model was able to drive the car along the track at least once, the model was saved and
retrained with new and fresh data. Sometimes if the data wasn't good enough the model performed worse, as it was mentioned before.

Finally, and in order to reach the goal of this project, the model was trained in 3 phases.

- First one: 8000 samples of normal driving
- Second one: 2000 samples of complicated curves and bridge footage
- Third one: 3000 samples of "recovery" driving

With these three phases the model is perfectly able to drive the car several times without touching
the ledges or roll over any surface.

![Recovery Training](imgs/from_training.png?raw=true)

## Bonus Clip

[Video](https://youtu.be/tULhqVPfABw)
