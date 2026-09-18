import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset location
DATA_PATH = "ct/data"

# Image size
IMG_SIZE = 224

images = []
labels = []

# COVID = 1
# non-COVID = 0

folders = {
    "COVID": 1,
    "non-COVID": 0
}

for folder, label in folders.items():

    folder_path = os.path.join(DATA_PATH, folder)

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        try:
            image = Image.open(file_path).convert("RGB")

            image = image.resize((IMG_SIZE, IMG_SIZE))

            image = np.array(image) / 255.0

            images.append(image)
            labels.append(label)

        except:
            print("Could not read:", file_path)


# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

print("Total images:", len(images))
print("Image shape:", images.shape)
print("Labels shape:", labels.shape)


# Split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))