#!/usr/bin/env python
# coding: utf-8

# # Helper Functions

# In[4]:


# imports

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
import numpy as np
import random


# In[5]:


# Set the seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ## Image Visualization Function

# In[6]:


import random
import numpy as np
import matplotlib.pyplot as plt

def view_random_images(X, y, classes_names, seed=42):
    """
    Displays a grid of 1 sample image per class from the dataset, along with their corresponding labels.

    Args:
    
        X: array-like
            The image dataset, typically a 4D array with shape (num_samples, height, width, channels).
        y: array-like
            The labels corresponding to each image in `X`, typically a 1D array with shape (num_samples,).
        classes_names: list of str
            A list of class names where each index corresponds to a class label in `y`.

    Returns:
        None
            Displays images in a grid with their respective labels beneath each image.

    Notes:
        - This function assumes that `y` contains integer class labels and that each class has at least one image.
        - Images are displayed in a grid format, with each label shown below its corresponding image.
    """
    
    labels = np.unique(y)
    num_classes = len(labels)
    num_cols = 5  # Set a fixed number of columns
    num_rows = (num_classes + num_cols - 1) // num_cols  # Calculate rows to fit all classes

    plt.figure(figsize=(16, 10))
    for i, label in enumerate(labels):
        label_indices = [index for index, yi in enumerate(y) if yi == label]
        rand_ind = random.choice(label_indices)  # Select a random index for the current label
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(X[rand_ind])
        plt.title(classes_names[label])
        plt.axis('off')  # Hide axes for clarity

    plt.tight_layout()
    plt.show()


# ## Plot loss and accuracy curves function

# In[7]:


def plot_training_curves(model_history):
    """
    This function generates two plots:
    1. Training Loss vs. Validation Loss.
    2. Training Accuracy vs. Validation Accuracy.

    Args:
        model_history (History): A Keras History object containing the training and validation metrics over epochs.

    Returns:
        None: The function plots the curves and does not return any value.
    """
    # Losses
    training_loss = model_history.history["loss"]  # training loss across epochs
    val_loss = model_history.history["val_loss"]  # validation loss across epochs

    # Accuracies
    training_accuracy = model_history.history["accuracy"]  # training accuracy across epochs
    val_accuracy = model_history.history["val_accuracy"]  # validation accuracy across epochs

    number_of_epochs = len(training_loss)
    epochs = range(number_of_epochs)

    # Plot losses
    plt.plot(epochs, training_loss, label='training_loss')
    plt.plot(epochs, val_loss, label='validation_loss')
    plt.xlabel('Epochs')
    plt.title('Training loss vs. Validation loss')
    plt.legend()

    # Plot accuracies
    plt.figure()  # a new figure for a separate plot
    plt.plot(epochs, training_accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='validation_accuracy')
    plt.xlabel('Epochs')
    plt.title('Training accuracy vs. validation accuracy')
    plt.legend()


# ## Model Evaluation results (for classification tasks)

# In[8]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def model_evaluation(y_true, y_pred):
    """
    Performs model evaluation by calculating model accuracy, precision, recall and f1 score for a multi-class classification model.

    Args:
        y_true (1D array): true labels (ground truth).
        y_pred (1D array): predicted labels.

    Returns:
        A dictionary of accuracy, precision, recall, and f1-score.
    """

    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate precision, recall and f1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    # reuslts in a dictionary
    model_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
    }
    
    return model_results
    


# ## LearningRateScheduler function

# In[9]:


def lr_schedule(epoch_number, initial_lr=1e-3, drop_rate=0.5, epochs_drop=10):
    """
    Adjust the learning rate based on the current epoch.
    Note: This function is needed for the 'LearningRateScheduler' tf.keras callback

    Parameters:
        epoch_number (int): Current epoch number
        initial_lr (float): Initial learning rate (default 1e-3)
        drop_rate (float): Factor to drop the learning rate (default 0.5)
        epochs_drop (int): Number of epochs before dropping the learning rate (default 10)

    Returns:
        lr (float): Updated learning rate
    """
    # validate inputs
    if not (0 < drop_rate <= 1):
        raise ValueError("drop_rate must be between 0 and 1")
    if epochs_drop <= 0:
        raise ValueError("epochs_drop must be a positive integer")

    # compute learning rate
    lr = initial_lr * (drop_rate ** (epoch_number // epochs_drop))
    
    return lr


# ## check_image_shapes_and_ranges function

# In[1]:


import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import numpy as np

def check_image_shapes_and_ranges(data_dir):
    shapes = []
    pixel_ranges = []  # To store the min and max pixel values for each image

    for class_ in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path)
                shapes.append(image.size)

                # Convert the image to a numpy array to find pixel range
                image_array = np.array(image)
                pixel_min = image_array.min()
                pixel_max = image_array.max()
                pixel_ranges.append((pixel_min, pixel_max))
    
    shapes_counter = Counter(shapes)
    return shapes, shapes_counter, pixel_ranges


# ## Image visualization function for images in a directory

# In[20]:


# Plot 1 random sample image per class with a fixed grid of 5 rows and 2 columns
def view_random_images_data_dir(data_dir, classes):
    """
    Display a grid of random sample images, one from each class in the dataset.

    Args:
        data_dir (str): Path to the dataset directory containing class subdirectories.
        classes (list): List of class names (subdirectory names) to display.

    Returns:
        None: Displays a 5x2 grid of images with class labels.
    """
    plt.figure(figsize=(12, 20))  # Adjust figure size for a 5x2 grid
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        image_files = os.listdir(class_dir)
        # Randomly select one image
        img_file = random.choice(image_files)
        img_path = os.path.join(class_dir, img_file)
        img = Image.open(img_path).resize((240, 240))  # Load and resize image using PIL
        plt.subplot(5, 2, i + 1)  # 5 rows, 2 columns
        plt.imshow(img)
        plt.axis('off')
        plt.title(class_name, fontsize=16, pad=8)  # Add class name as the title
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust spacing
    plt.suptitle("Random Sample Images from FoodVision-10 Dataset", fontsize=24, y=0.92)  # Add a title
    plt.show()


# In[ ]:





# In[2]:


# import nbformat
# from nbconvert import PythonExporter

# # Load the notebook
# input_file = 'helper_functions.ipynb'
# output_file = 'helper_functions.py'

# with open(input_file, 'r', encoding='utf-8') as f:
#     notebook = nbformat.read(f, as_version=4)

# # Convert notebook to script
# python_exporter = PythonExporter()
# script, _ = python_exporter.from_notebook_node(notebook)

# # Save the script
# with open(output_file, 'w', encoding='utf-8') as f:
#     f.write(script)

# print(f"Notebook converted to script: {output_file}")


# In[10]:


# !jupyter nbconvert --to script helper_functions.ipynb


# In[ ]:





# In[ ]:




