# Semantic image segmentation

## Project description
In this repository, we work on a semantic image segmentation problem/model. Semantic image segmentation is a computer vision task where the goal is to label each pixel of an image with a class from a predefined set of categories or classes. 
It's very useful for autonomous vehicules use case for example. From an image, we can identify and segment automatically different objects in a scene (like roads, pedestrians, vehicles) to help self-driving cars understand their environment. The goal is to carry on a region-specific labeling. It is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples' lives in danger. 


## Dataset
We use the CARLA self-driving car dataset.
We have 2 folders with 1060 files each:
* the original images
* the masks (labelled) of each image


### Data preprocessing
We carry on a few steps to make the data ready:
* images:
    * read the image path and load the image file
    * decode the png picture into RGB format (because colored images so 3 channels)
    * normalize the images
    * resize the images to ensure all input images have the same size (assigning each pixel in the output image to the value of the nearest pixel from the input image)
* masks:
    * read the mask path and load the mask file
    * decode the png mask into RGB format
    * collapse the channels by using the maximum value across the RGB channels, so each pixel represents the class with the highest label
    * resize the mask similarly to the images, ensuring it has the same size as the input image (assigning each pixel in the output mask to the value of the nearest pixel from the input mask)


## Model
We use the U-Net model for this project. It's called like that for its U-shaped architecture (with a contracting/downsampling path and an expanding/upsampling path). U-Net is a type of Convolutional Neural Network (CNN) designed for quick and precise image segmentation (but we replace dense layers with transposed convolution layers).
Transposed convolution layers aims at upsampling the feature map back to the size of the original input image (while preserving spatial information). Indeed, the dense layers destroy spatial information (the "where" of the image).

The U-Net model also adds skip connections, to retain information that would otherwise become lost during the contracting path. Skip connections send information to every upsampling layer in the expanding path from the corresponding downsampling layer in the contracting path, capturing finer information while also keeping computation low. This helps prevent information loss, as well as model overfitting. 

Details of the model:
* contracting path (downsampling steps): images are first fed through several convolutional layers (which reduce height and width, while growing the number of channels). It follows a regular CNN architecture, with convolutional layers, their activations, and maxpooling layers to downsample the image and extract its features. 
In details, it consists of the repeated application of two 3x3 same padding convolutions, each followed by a rectified linear unit (ReLU) activation function and a 2x2 max pooling operation with stride 2 for downsampling. Max pooling reduces the height and width by half at each step, progressively capturing more abstract features. At each downsampling step, the number of feature channels is doubled, allowing the network to capture increasingly complex features. For some layers, the dropout layer is added to regularize the model and avoid overfitting. Dropout randomly sets some activations to zero during training.
* expanding path (upsampling steps): it performs the opposite operation of the downsampling path, growing the image back to its original size, while shrinking the channels gradually. 
In details, each step in the expanding path upsamples the previous output with a 3x3 transposed convolution. This transposed convolution halves the number of feature channels, while growing the height and width of the image. After each upsampling step, the feature map is concatenated with the corresponding feature map from the contracting path (this is known as a skip connection). The skip connections help preserve fine-grained information that might otherwise be lost during downsampling, particularly around the edges of objects. Two additional 3x3 convolutions are applied, each followed by a ReLU activation function. The skip connections are used to prevent border pixel information loss and overfitting in U-Net.
* final layer: a 1x1 convolution is applied to map each feature vector to the desired number of classes. We use a 1x1 convolution here with 23 channels (since we aim at classifying among 23 classes in this project). 

The U-Net network has 23 convolutional layers in total:
* 18 3x3 convolutions, with ReLU activation function (8 during the downsampling part, 2 between the downsampling and upsampling parts and 8 during the upsampling part)
* 4 3x3 up-convolutions (during the upsampling part)
* 1 1x1 convolution (final step)
U-Net uses an equal number of convolutional blocks for downsampling path and upsampling path.


## Loss function
For semantic image segmentation we use sparse categorical crossentropy instead of categorical crossentropy.
* sparse categorical crossentropy: contains the index of the class
* categorical crossentropy: only contains 0s and 1s
We use sparse categorical crossentropy to perform pixel-wise multiclass prediction (each pixel in every mask is assigned a single probability that it belongs to a certain class, from 0 to num_classes-1). The correct class is the channel/layer with the highest probability. Sparse categorical crossentropy compares the predicted probability distribution (output of the network) to the true label (which is an integer corresponding to the correct class).


## Insights from the project
We tried with:
* 15 epochs: pretty fast to train
* 40 epochs: longer to train and we found great results
* more epochs but we didn't get per se better results and very longer training time


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.
