##################################################################################################
#####                                                                                        #####
#####                               SEMANTIC IMAGE SEGMENTATION                              #####
#####                                 Created on: 2025-02-18                                 #####
#####                                  Updated on 2025-02-23                                 #####
#####                                                                                        #####
##################################################################################################

##################################################################################################
#####                                        PACKAGES                                        #####
##################################################################################################

# To clear the environment
globals().clear()

# Load libraries
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

# Set up the paths (on one side, the raw pictures and on the other side the masks [semantic image segmentation])
image_path = 'data/CameraRGB/'
mask_path = 'data/CameraMask/'

# List all the entries (here, names) in a specified directory (because images and masks have same names, splitted per folder)
image_list_orig = os.listdir(image_path)

# List of paths of all images/masks
image_list = [image_path+i for i in image_list_orig]
mask_list = [mask_path+i for i in image_list_orig]


##################################################################################################
#####                                    CHECK SOME DATA                                     #####
##################################################################################################

# Randomly get an image and its mask
N = 21
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

# Transform the images and masks lists into datasets
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

# Check for 3 examples, the pairs between image_list_ds and mask_list_ds
# .take() method limits the number of elements in a dataset
# zip() function pairs elements from both datasets, element-wise. It combines the element from image_list_ds and mask_list_ds
for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)

# tf.constant() is essential for creating tensors that don't change throughout the computation
image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

# tf.data.Dataset.from_tensor_slices() function takes these two lists and constructs a dataset where each element is a pair of (image, mask) files
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

# Get the first example of dataset
for image, mask in dataset.take(1):
    print(image)
    print(mask)


##################################################################################################
#####                                  PREPROCESS THE DATA                                   #####
##################################################################################################

# We create a process_path function, with 3 steps:
# * it reads the images/masks paths
# * it decodes the png image considering there are 3 channels because we have colors pictures
# * it converts the image to float32 type (normalizing between 0 and 1 the values of images, avoiding to divide by 255)
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) 
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

# We create a function to resize all the images and masks (where channels is the same as in the original image)
# nearest method refers to assigning each pixel in the output image the value of the nearest pixel from the input image
def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    return input_image, input_mask

# Apply the process_path function to dataset (which gathers images and masks)
image_ds = dataset.map(process_path)

# Apply the preprocess function to image_ds (so each image is like pixelised now)
processed_image_ds = image_ds.map(preprocess)


##################################################################################################
#####                                       U-NET MODEL                                      #####
##################################################################################################

##################################################################################################
#####                              ENCODER (DOWNSAMPLING BLOCK)                              #####
##################################################################################################

# The encoder is a stack of various conv_blocks, so we first create the conv_block function. 
# Each conv_block is composed of 2 Conv2D layers with ReLU activations. 
# We apply Dropout and MaxPooling2D to the last two blocks of the downsampling. 
# The function will return two tensors: 
# * next_layer: that will go into the next block
# * skip_connection: that will go into the corresponding decoding block

# Note: if max_pooling=True, the next_layer will be the output of the MaxPooling2D layer 
# but the skip_connection will be the output of the previously applied layer (Conv2D or Dropout, depending on the case). 
# Else, both results will be identical.

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block: Conv2D (3x3) ReLU -> Conv2D (3x3) ReLU
    then ready for the skip connection or MaxPool2D to further the downsampling step
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    conv = Conv2D(n_filters,    # Number of filters
                  3,            # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
         
    if max_pooling:
        next_layer = MaxPooling2D((2,2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


##################################################################################################
#####                               DECODER (UPSAMPLING BLOCK)                               #####
##################################################################################################

# The upsampling block, upsamples the features back to the original image size. 
# At each upsampling level, we'll take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.
# We create the function upsampling_block.
# It takes the arguments expansive_input (which is the input tensor from the previous layer) and contractive_input (the input tensor from the previous skip layer)
# The number of filters here is the same as in the downsampling block we just created
# Our Conv2DTranspose layer will take n_filters with shape (3,3) and a stride of (2,2), with padding set to same.
# It's applied to expansive_input, or the input tensor from the previous layer. 
# Then we concatenate our Conv2DTranspose layer output to the contractive input, with an axis of 3.
# Finally, we use 2 Conv2D layers with same parameters that we used for encoder path (ReLU activation, He normal initializer, same padding). 

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block: Conv2DTranspose (3,3) + contractive_input (from the downsampling step)
    then Conv2D (3,3) ReLU -> Conv2D (3,3) ReLU     
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    up = Conv2DTranspose(
                 n_filters,     # Number of filters
                 (3,3),         # Kernel size
                 strides=(2,2),
                 padding='same')(expansive_input)
    
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                 (3,3),
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                 (3,3),
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)
    expansive_input, contractive_input, n_filters

    return conv


##################################################################################################
#####                                     BUILD THE MODEL                                    #####
##################################################################################################

# We need to specify the number of output channels (in that case 23 because there are 23 possible labels for each pixel in this self-driving car dataset). 

# Model steps:
# 1. Use a conv block that takes the inputs of the model and the number of filters
# 2. For the next steps, we chain the first output element of each block to the input of the next convolutional block,
# we double the number of filters at each step.
# 3. For the 4th conv_block, we add dropout_prob of 0.3
# 4. For the 5th conv_block, we add dropout_prob of 0.3 and we turn off max_pooling
# 5. Bottleneck layer: we use cblock5 as expansive_input and cblock4 as contractive_input, with n_filters * 8
# 6. For the next steps, we chain the output of the previous block as expansive_input and the corresponding skip_connection.
# We use half the number of filters of the previous block
# 7. conv9 is a Conv2D layer with ReLU activation, He normal initializer and same padding
# 8. conv10 is a Conv2D that takes the number of classes as the filter, a kernel size of 1 and same padding. 
# The output of conv10 is the output of your model. 

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Encoder path
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    # Decoder path
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


##################################################################################################
#####                     SET IMAGE DIMENSIONS AND CERATE THE UNET MODEL                     #####
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

# We use Adam optimizer and sparse categorical crossentropy as loss
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


##################################################################################################
#####                                    DATASET HANDLING                                    #####
##################################################################################################

# We create a function that shows until an input image, its true mask and its predicted mask
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# We take 1 example and display the input image and its true mask
for image, mask in image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])

# We take 1 example and display the input (processed) image and its (processed) true mask (we see each pixels)
for image, mask in processed_image_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
display([sample_image, sample_mask])


##################################################################################################
#####                                     TRAIN THE MODEL                                    #####
##################################################################################################

# Variables to use to train the model
EPOCHS = 15 #40
VAL_SUBSPLITS = 5
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
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Create a function to check our predicted masks against the true mask and the original input image
def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(unet.predict(sample_image[tf.newaxis, ...]))])

# Check our predicted mask against the true mask and the original input image for 6 images
show_predictions(train_dataset, 6)
