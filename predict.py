import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from model import U_Net

# Load the U-Net model
model = U_Net(n_classes=18)

model.load_weights('big_last_model_weights.h5')

# Path to the input image
image_path = 'path/to/input_image.jpg'

# Read and preprocess the input image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = tf.expand_dims(img, axis=0)

# Get the segmentation mask from the model
mask = model.predict(img)[0]
mask = tf.argmax(mask, axis=-1)

# Display the input image and segmentation mask
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img[0])
axs[0].set_title('Input Image')
axs[0].axis('off')

axs[1].imshow(mask, cmap='jet')
axs[1].set_title('Segmentation Mask')
axs[1].axis('off')

plt.show()
