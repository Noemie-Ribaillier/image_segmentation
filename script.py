##################################################################################################
#####                                                                                        #####
#####                               SEMANTIC IMAGE SEGMENTATION                              #####
#####                                 Created on: 2025-02-18                                 #####
#####                                  Updated on 2025-03-18                                 #####
#####                                                                                        #####
##################################################################################################

##################################################################################################
#####                                        PACKAGES                                        #####
##################################################################################################

# To clear the environment
globals().clear()

# Load the libraries
import numpy as np
import setuptools.dist
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import pandas as pd
import imageio
import matplotlib.pyplot as plt

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/image_segmentation')


##################################################################################################
#####                                     IMPORT THE DATA                                    #####
##################################################################################################

# Define the paths (on one side, the images and on the other side the masks [semantic image segmentation])
image_path = 'data/CameraRGB/'
mask_path = 'data/CameraMask/'

# List all the files (here, images name) in a specified directory (because images and masks have same names, splitted per folder)
image_list_orig = os.listdir(image_path)

# List of paths of all images/masks
image_list = [image_path+i for i in image_list_orig]
mask_list = [mask_path+i for i in image_list_orig]


##################################################################################################
#####                              GET A FIRST LOOK AT THE DATA                              #####
##################################################################################################

# Randomly get an image and its mask
N = np.random.randint(len(image_list))
img = imageio.imread(image_list[N])
mask = imageio.imread(mask_list[N])

# Show them (to have a first look)
fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img)
arr[0].set_title('Image')
arr[1].imshow(mask[:, :, 0])
arr[1].set_title('Segmentation')
plt.show()


##################################################################################################
#####                          SPLIT THE DATA INTO IMAGES AND MASKED                         #####
##################################################################################################

# Transform the image and mask lists into datasets (without shuffling to keep the same order between images and masks)
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

# Check for 3 examples, the pairs between image_list_ds and mask_list_ds
# .take() method limits the number of elements to take into account
# zip() function pairs elements from both datasets, element-wise. It combines the element from image_list_ds and mask_list_ds
for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)

# Convert image and mask lists into immutable (don't change throughout the computation) constant tensor (that can be used in TF operations)
# The paths won't change so we transform them as immutable (it will help with data consistency as well)
image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

# Create a TF Dataset from tensors (takes the two lists and constructs a dataset where each element is a pair of (image, mask) files)
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

# Get the first example of dataset
for image, mask in dataset.take(1):
    print(image)
    print(mask)


##################################################################################################
#####                                  PREPROCESS THE DATA                                   #####
##################################################################################################

# Create the process_path function to get the images/masks from the path
def process_path(image_path, mask_path):
    '''
    Get the images from the path, it processes the data with 3 steps:
    * read the path
    * decode the png image considering there are 3 channels because we have colors pictures
    * convert the image to float32 type (normalizing between 0 and 1 the values of images, avoiding to divide by 255)

    Get the masks from the path, it processes the data with 3 steps:
    * read the path
    * decode the png image considering there are 3 channels because we have colors masks
    * collapse the channels into a single channel that contains the maximum class label for each pixel

    Inputs:
    image_path -- path of the images (string)
    mask_path -- path of the masks (string)

    Returns:
    img -- image tensor with shape (height, width, 3) and dtype float32 where pixel values are in the range [0.0, 1.0]
    mask -- mask tensor with shape (height, width, 1), where each pixel contains the maximum value from the original channels of the mask
    '''
    # Read the path
    img = tf.io.read_file(image_path)
    # Decode the png image considering there are 3 channels because we have colors pictures
    img = tf.image.decode_png(img, channels=3)
    # Convert the image to float32 type (normalizing between 0 and 1 the values of images, avoiding to divide by 255)
    img = tf.image.convert_image_dtype(img, tf.float32) 
    
    # Read the path and decode the png image (considering 3 channels because RGB images)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    # Collapse the channels (masks often have multiple channels, one per class) into a single channel that contains the maximum class label for each pixel
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    
    return img, mask


# Create a function to resize all the images and masks (with identic channels as in the original image)
def preprocess(image, mask):
    '''
    Reshape the images and masks to have the same size
    nearest method: assign each pixel in the output image to the value of the nearest pixel from the input image

    Inputs:
    image -- the tensor representing an image
    mask -- the tensor representing a mask

    Returns:
    input_image -- the resized image tensor, with shape [96, 128, channels], where channels is the same as the input image's number of channels
    input_mask -- the resized mask tensor, with shape [96, 128, channels], where channels is the same as the input mask's number of channels
    '''
    # Resize the images/masks to have same shape
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask


# Apply the process_path function to dataset (which gathers images and masks)
image_mask_ds = dataset.map(process_path)

# Apply the preprocess function to image_mask_ds (so each image is pixelised now)
processed_image_mask_ds = image_mask_ds.map(preprocess)


##################################################################################################
#####                          CONTRACTING PATH (DOWNSAMPLING BLOCK)                         #####
##################################################################################################

# Create the function to create the conv block (used for the contracting part of the model)
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block: Conv2D (3x3) ReLU x2
    then ready for the skip connection or MaxPool2D to further the downsampling step
    
    Inputs:
    inputs -- input tensor
    n_filters -- number of filters for the convolutional layers
    dropout_prob -- dropout probability
    max_pooling -- whether to use MaxPooling2D to reduce the spatial dimensions of the output volume
    
    Returns: 
    next_layer -- next layer output (that will go into the next block)
    skip_connection --  skip connection output (that will go into the corresponding decoding block)
    """
    # 1st Conv2D (taking input as input)
    conv = Conv2D(n_filters,
                  # Kernel size
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    # 2nd Conv2D (taking the previous output as input)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    
    # Apply the Dropout layer if the dropout probability is not 0 (apply it on the previous output, that's a sequential model)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    
    # Apply the MaxPooling2D layer if True 
    # If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer 
    # but the skip_connection will be the output of the previously applied layer (Conv2D or Dropout, depending on the case)
    if max_pooling:
        next_layer = MaxPooling2D((2,2))(conv)
    # Else, both results (next_layer & skip_connection) will be identical
    else:
        next_layer = conv
    
    # Keep the element for skip connection as it is before the maxpooling
    skip_connection = conv
    
    return next_layer, skip_connection


##################################################################################################
#####                           EXPANDING PATH (UPSAMPLING BLOCK)                            #####
##################################################################################################

# Create the function to create the upsampling block (used for the expanding part of the model)
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block, it upsamples the features back to the original image size
    Conv2DTranspose (3,3) + contractive_input (from the downsampling step)
    then Conv2D (3,3) ReLU x2
    
    Inputs:
    expansive_input -- input tensor from previous layer
    contractive_input -- input tensor from previous skip layer
    n_filters -- number of filters for the convolutional layers
    
    Returns: 
    conv -- tensor output
    """
    # Transpose convolution of the input
    up = Conv2DTranspose(
                 n_filters,
                 # Kernel size
                 3,         
                 strides = 2,
                 padding = 'same')(expansive_input)
    
    # Concatenate the Conv2DTranspose layer output to the contractive input
    merge = concatenate([up, contractive_input], axis=3)

    # Use 2 Conv2D layers with same parameters that we used for contracting path
    conv = Conv2D(n_filters,
                 3,
                 activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(merge)
    conv = Conv2D(n_filters,
                 3,
                 activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(conv)
    
    return conv


##################################################################################################
#####                                     BUILD THE MODEL                                    #####
##################################################################################################

# Create the function to implement the U-Net model
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Implement the U-Net model
    
    Inputs:
    input_size -- input shape 
    n_filters -- number of filters for the convolutional layers
    n_classes -- number of output classes (here we take 23 because there are 23 possible labels for each pixel in this dataset)
    
    Returns: 
    model -- tf.keras.Model (U-Net model)
    """
    # Create the input layer
    inputs = Input(input_size)
    # Contracting path
    # 1. Use a conv block that takes the inputs of the model and the number of filters
    cblock1 = conv_block(inputs, n_filters)
    # 2. For the next steps, we chain the first output element of each block to the input of the next conv block (doubling the number of filters at each step)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    # 3. For the 4th conv_block, we add dropout_prob of 0.3
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)
    # 4. For the 5th conv_block, we add dropout_prob of 0.3 and we turn off max_pooling
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Expanding path
    # 5. Bottleneck layer: we use cblock5 as expansive_input and cblock4 as contractive_input, with n_filters * 8
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)
    # 6. For the next steps, we chain the output of the previous block as expansive_input and the corresponding skip_connection (we use half the number of filters of the previous block)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    # 7. conv9 is a Conv2D layer with ReLU activation, He normal initializer and same padding (useful for refining the features from the upsampling process and improving the final segmentation map)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    # 8. conv10 is a Conv2D that takes the number of classes as the filter, a kernel size of 1 and same padding
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    # The output of conv10 is the output of the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


##################################################################################################
#####                     SET IMAGE DIMENSIONS AND CREATE THE U-NET MODEL                     #####
##################################################################################################

# Set image dimensions
img_height = 96
img_width = 128
num_channels = 3

# Create the model with the right dimensions (input_size)
unet = unet_model((img_height, img_width, num_channels))

# Check out the model summary
unet.summary()


##################################################################################################
#####                                      LOSS FUNCTION                                     #####
##################################################################################################

# Use Adam optimizer and sparse categorical crossentropy as loss
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
# from_logits=True indicates to the loss function that it should apply the softmax activation internally before computing the categorical crossentropy loss
# The model's output is raw logits (ie unnormalized scores) and not probabilities


##################################################################################################
#####                                    DATASET HANDLING                                    #####
##################################################################################################

# Create a function that shows until an input image, its true mask and its predicted mask
def display(display_list):
    """
    Plot some elements among an input image, its true mask and its predicted mask
    
    Inputs:
    display_list -- list of elements we want to plot [input image, true mask, predicted mask]
    """
    # Set up the plot window
    plt.figure(figsize=(15, 15))
    # Determine the title of each plot
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    # Iterate on the elements we want to display
    for i in range(len(display_list)):
        # Split the row into possible several plots
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # Plot the image
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Take 1 example and display the input image and its true mask
for image, mask in image_mask_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])

# Take 1 example and display the input (processed) image and its (processed) true mask (we see each pixels)
for image, mask in processed_image_mask_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])


##################################################################################################
#####                                     TRAIN THE MODEL                                    #####
##################################################################################################

# Variables to use to train the model
EPOCHS = 15 #40
BUFFER_SIZE = 500
BATCH_SIZE = 32

# Fix the seed to get stable results (it removes the random factor)
tf.keras.utils.set_random_seed(1)

# Use enable_op_determinism to get the same outputs even after running several times (with the same inputs on the same hardware)
tf.config.experimental.enable_op_determinism()

# cache(): it caches the data in memory after it has been processed for the first time
# shuffle(): it shuffles the data to prevent model from learning the order of data (and prevent overfitting)
# BUFFER_SIZE argument controls the number of elements that are randomly shuffled at a time
# batch(): it groups the dataset into batches of size BATCH_SIZE to process multiple samples at once (to have more efficient computation and memory usage)
train_dataset = processed_image_mask_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Train the model, we use the train dataset and run it EPOCHS times
model_history = unet.fit(train_dataset, epochs=EPOCHS)


##################################################################################################
#####                                   PLOT MODEL ACCURACY                                  #####
##################################################################################################

# Plot the model accuracy (per epoch)
plt.plot(model_history.history["accuracy"])
plt.xticks()
plt.xlabel('Number of epoches')
plt.ylabel('Accuracy')
plt.title ('Evolution of the accuracy with respect to the number of epoches')
plt.show()


##################################################################################################
#####                                    SHOW PREDICTIONS                                    #####
##################################################################################################

# Create a function to determine the class based on the max probability
def create_mask(pred_mask):
    """
    Determine/predict the class of each pixel in a mask
    
    Inputs:
    pred_mask -- predicted mask (output of model.predict())
    """
    # Find the index of the class with the highest score 
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # Create a new axis for data consistency
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask[0]


# Create a function to check our predicted masks against the true mask and the original input image
def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    # If dataset parameter is filled
    if dataset:
        # Select certain images/masks from the dataset
        for image, mask in dataset.take(num):
            # Predict the mask of the input image
            pred_mask = unet.predict(image)
            # Create the mask
            pred_mask2 = create_mask(pred_mask)
            # Display the input image, its true mask and the predict mask
            display([image[0], mask[0], pred_mask2])


# Check the predicted mask against the true mask and the original input image for 6 images
show_predictions(train_dataset, 6)
