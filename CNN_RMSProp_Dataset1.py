#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from PIL import Image, ImageChops, ImageEnhance
from tqdm.notebook import tqdm

# Display plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def convert_to_ela_image(path, quality):
    try:
        original_image = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    try:
        resaved_file_name = 'resaved_image.jpg'
        original_image.save(resaved_file_name, 'JPEG', quality=quality)
        resaved_image = Image.open(resaved_file_name)
    except Exception as e:
        print(f"Error saving or opening resaved image: {e}")
        return None

    try:
        ela_image = ImageChops.difference(original_image, resaved_image)

        extrema = ela_image.getextrema()
        max_difference = max([pix[1] for pix in extrema])
        if max_difference == 0:
            max_difference = 1
        scale = 350.0 / max_difference

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    except Exception as e:
        print(f"Error processing ELA image: {e}")
        return None

    return ela_image

def prepare_image(image_path):
    image_size = (128, 128)
    ela_image = convert_to_ela_image(image_path, 90)
    if ela_image is None:
        return None
    resized_ela_image = ela_image.resize(image_size)
    return np.array(resized_ela_image).flatten() / 255.0


    
# Load dataset and prepare images
X = []  # ELA converted images
Y = []  # 0 for fake, 1 for real
authentic_path = "D:/CASIA2/Au"  # Folder path of the authentic images in the dataset
forged_path = "D:/CASIA2/Tp"     # Folder path of the forged images in the dataset

for path, label in [(authentic_path, 1), (forged_path, 0)]:
    for filename in tqdm(os.listdir(path), desc="Processing Images: "):
        if filename.endswith(('jpg', 'png')):
            full_path = os.path.join(path, filename)
            X.append(prepare_image(full_path))
            Y.append(label)

X = np.array(X)
Y = np.array(Y)
X = X.reshape(-1, 128, 128, 3)

# Split the dataset into train, validation, and test sets
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.05, random_state=5)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=5)

print(f'Training images: {len(X_train)}, Training labels: {len(Y_train)}')
print(f'Validation images: {len(X_val)}, Validation labels: {len(Y_val)}')
print(f'Test images: {len(X_test)}, Test labels: {len(Y_test)}')


# In[8]:


def build_model():
    model = Sequential()  # Sequential Model
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation = 'sigmoid'))
    return model

model = build_model()
model.summary()


# In[10]:


epochs = 15
batch_size = 32

#Optimizer
from tensorflow.keras.optimizers import RMSprop
init_lr = 1e-4   #learning rate for the optimizer
# Compile the model with RMSprop optimizer
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[11]:


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy',
                               min_delta=0,
                               patience=10,
                               verbose=0,
                               mode='auto')

# Train the model
hist = model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_val, Y_val),
                 callbacks=[early_stopping])

# Save the model
model.save('model.h5')


# In[29]:


# Save training history
history_dict = hist.history
with open('history.json', 'w') as json_file:
    json.dump(history_dict, json_file)
    
# Plot training and validation curves
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
ax[0].plot(history_dict['loss'], color='b', label="Training loss")
ax[0].plot(history_dict['val_loss'], color='r', label="Validation loss", axes=ax[0])
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Loss', fontsize=16)
ax[0].legend(loc='best', shadow=True)

# Plot accuracy
ax[1].plot(history_dict['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history_dict['val_accuracy'], color='r', label="Validation accuracy")
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Accuracy', fontsize=16)
ax[1].legend(loc='best', shadow=True)

fig.suptitle('Metrics', fontsize=20)


# In[33]:


def plot_confusion_matrix(cf_matrix):
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]  # Number of images in each classification block
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]  # Percentage value of images in each block w.r.t total images

    axes_labels = ['Forged', 'Authentic']
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="flare", xticklabels=axes_labels, yticklabels=axes_labels)
    plt.xlabel('Predicted labels', fontsize=13)
    plt.ylabel('True labels', fontsize=13)
    plt.title('Confusion Matrix', fontsize=10, fontweight='bold')
    plt.show()

Y_pred = model.predict(X_val)               # Predict the values from the validation dataset
Y_pred_classes = np.round(Y_pred)           # Round off the sigmoid value
Y_true = Y_val

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)  # Compute the confusion matrix
plot_confusion_matrix(confusion_mtx)


# In[34]:


# Print classification report
print(classification_report(Y_true, Y_pred_classes))

# Calculate testing accuracy
class_names = ['Forged', 'Authentic']
correct_test = 0  # Correctly predicted test images
total_test = 0    # Total test images

for index, image in enumerate(tqdm(X_test, desc="Processing Images: ")):
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.round(y_pred)
    total_test += 1
    if y_pred_class == Y_test[index]:  # If prediction is correct
        correct_test += 1
    
print(f'Total test images: {total_test}\nCorrectly predicted images: {correct_test}\nAccuracy: {correct_test / total_test * 100.0} %')


# In[35]:


# Test an image
test_image_path = "D:/CASIA1/Au/Au_ani_0097.jpg"  # Test image path
test_image = prepare_image(test_image_path)
test_image = test_image.reshape(-1, 128, 128, 3)

y_pred = model.predict(test_image)
y_pred_class = round(y_pred[0][0])

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Display original image
original_image = plt.imread(test_image_path) 
ax[0].axis('off')
ax[0].imshow(original_image)
ax[0].set_title('Original Image')

# Display ELA applied image
ax[1].axis('off')
ax[1].imshow(convert_to_ela_image(test_image_path, 90)) 
ax[1].set_title('ELA Image')

print(f'Prediction: {class_names[y_pred_class]}')
if y_pred <= 0.5:
    print(f'Confidence: {(1 - (y_pred[0][0])) * 100:0.2f}%')
else:
    print(f'Confidence: {(y_pred[0][0]) * 100:0.2f}%')
print('--------------------------------------------------------------------------------------------------------------')


# In[36]:


# Test on the entire dataset
test_folder_path = "D:/CASIA1/Sp"  # Dataset path
authentic, forged, total = 0, 0, 0

for filename in tqdm(os.listdir(test_folder_path), desc="Processing Images: "):
    if filename.endswith(('jpg', 'png')):
        test_image_path = os.path.join(test_folder_path, filename)  # Corrected variable name
        test_image = prepare_image(test_image_path)
        test_image = test_image.reshape(-1, 128, 128, 3)  # Reshape image if needed
        y_pred = model.predict(test_image)
        y_pred_class = np.round(y_pred)
        total += 1
        if y_pred_class == 0:
            forged += 1
        else:
            authentic += 1

print(f'Total images: {total}\nAuthentic Images: {authentic}\nForged Images: {forged}')


# In[37]:


# Test on the entire dataset
test_folder_path = "D:/CASIA1/Au"  # Dataset path
authentic, forged, total = 0, 0, 0

for filename in tqdm(os.listdir(test_folder_path), desc="Processing Images: "):
    if filename.endswith(('jpg', 'png')):
        test_image_path = os.path.join(test_folder_path, filename)  # Corrected variable name
        test_image = prepare_image(test_image_path)
        test_image = test_image.reshape(-1, 128, 128, 3)  # Reshape image if needed
        y_pred = model.predict(test_image)
        y_pred_class = np.round(y_pred)
        total += 1
        if y_pred_class == 0:
            forged += 1
        else:
            authentic += 1

print(f'Total images: {total}\nAuthentic Images: {authentic}\nForged Images: {forged}')

