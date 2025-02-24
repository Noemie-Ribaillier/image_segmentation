# Semantic image segmentation

## Description of the project
In this repository we work on an semantic image segmentation problem/model.
Semantic image segmentation is a computer vision task where the goal is to label each pixel of an image with a class from a predefined set of categories or classes. 
It's very useful for autonomous vehicules use case for example. From an image, we can identify and segment automatically different objects in a scene (like roads, pedestrians, vehicles) to help self-driving cars understand their environment. The goal is to carry on a region-specific labeling. It is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples' lives in danger. 

## Dataset
We use the CARLA self-driving car dataset.
We have 2 folders with 1060 files each:
* the original image
* the masks (labelled) of the images

## Model
We use the U-Net model for this project. It's called like that for the U-shape. U-Net is a type of Convolutional Neural Network (CNN) designed for quick and precise image segmentation (but we replace dense layers with a transposed convolution layer)

it upsamples the feature map back to the size of the original input image (while preserving spatial information).
Indeed, the dense layers destroy spatial information (the "where" of the image).
When using transpose convolutions, the input size no longer needs to be fixed (contrary to when dense layers are used).

It also adds skip connections, to retain information that would otherwise become lost during encoding. 
Skip connections send information to every upsampling layer in the decoder 
from the corresponding downsampling layer in the encoder, 
capturing finer information while also keeping computation low. 
These help prevent information loss, as well as model overfitting. 

Details of the model:
* downsampling step: images are first fed through several convolutional layers (which reduce height and width, 
while growing the number of channels). It follows a regular CNN architecture, with convolutional layers, 
their activations, and pooling layers to downsample the image and extract its features. 
In detail, it consists of the repeated application of two 3 x 3 same padding convolutions, 
each followed by a rectified linear unit (ReLU) and a 2 x 2 max pooling operation with stride 2 for downsampling. 
At each downsampling step, the number of feature channels is doubled.
* crop function step: this step crops the image from the downsampling path and concatenates it to the 
current image on the expanding path to create a skip connection. Skip connections are used to prevent border pixel information loss and overfitting in U-Net
* expanding path (upsampling steps): it performs the opposite operation of the downsampling path, growing the image back to its original size, while shrinking the channels gradually. In detail, each step in the expanding path upsamples the feature map, followed by a 2 x 2 convolution (the transposed convolution). This transposed convolution halves the number of feature channels, while growing the height and width of the image.
* concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU. You need to perform cropping to handle the loss of border pixels in every convolution.
* in the final layer, a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. The channel dimensions from the previous layer correspond to the number of filters used, so when you use 1x1 convolutions, you can transform that dimension by choosing an appropriate number of 1x1 filters. When this idea is applied to the last layer, you can reduce the channel dimensions to have one layer per class. 

The U-Net network has 23 convolutional layers in total:
* 18 conv 3x3, ReLU (8 during the downsampling part, 2 between the downsampling and upsamping and 8 during the upsampling part )
* 4 up-conv 2x2 (during the upsampling part)
* 1 conv 1x1 (final step)
U-Net uses an equal number of convolutional blocks and transposed convolutions for downsampling and upsampling.

## Loss function
For semantic image segmentation we use sparse categorical crossentropy instead of categorical crossentropy.
* sparse categorical crossentropy: contains the index of the class
* categorical crossentropy: only contains 0s and 1s.
We use sparse categorical crossentropy to perform pixel-wise multiclass prediction (each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes-1). The correct class is the layer with the higher probability.

## Insights from the project
We tried with:
* 15 epochs: pretty fast to train
* 40 epochs: longer to train and we found great results
* more epochs but we didn't get per se better results and very longer training time