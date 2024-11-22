#!/usr/bin/env python
# coding: utf-8

# # Helper Functions

# In[28]:


# imports

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
import numpy as np
import random


# In[29]:


# Set the seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ## Image Visualization Function

# In[30]:


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

# In[31]:


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

# In[32]:


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

# In[1]:


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


# In[ ]:





# In[ ]:





# In[2]:


# !jupyter nbconvert --to script helper_functions.ipynb


# In[ ]:





# In[ ]:




