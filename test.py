from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
import random
from sklearn.metrics import confusion_matrix

"""
    Projekt: Rozpoznawanie zwierząt na podstawie zdjęć ich głów
    Autor: Michał Wojtasik

    Opis:
    1. Wczytanie zdjęć z folderu i przeskalowanie ich do rozmiaru 100x100 
        oraz przekształcenie do skali szarości
    2. Wyodrębnienie unikalnych etykiet
    3. Ekstrakcja cech
    4. Podzielenie danych na zbiór treningowy i testowy
    4. Stworzenie modelu
    5. Trenowanie modelu
    6. Testowanie modelu
    7. Wyświetlenie wyników
"""

def image_resize_to_grayscale(img_path: str, new_width: int) -> np.ndarray:
    img = Image.open(img_path).convert('L')
    new_img = img.resize((new_width, new_width))
    #img = sk.io.imread(img_path)
    #new_img = sk.transform.resize(img, (new_width, new_width))
    return np.array(new_img)


def load_images_to_dict(dir_path: str, resize_size=100) -> dict:
    images = {}
    images['data'] = []
    images['name'] = []
    images['label'] = []
    for root,  folders, _ in os.walk(f"{dir_path}"):
        for folder in folders:
            file_number = 0
            for file in os.listdir(f"{root}/{folder}"):                
                path = f"{root}/{folder}/{file}"
                file_name = file.split(".")[0]
                resized_image = image_resize_to_grayscale(path, resize_size)
                images['data'].append(resized_image)
                images['name'].append(file_name)
                images['label'].append(folder)

                # extension = file.split(".")[-1].lower()
                # file_number += 1
                # Renaming the files to the format: folder_number.extension
                # os.rename(path, f"{root}/{folder}/{folder}_{file_number}.{extension}")

    return images


def divide_image_to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    blocks = np.zeros((img.shape[0]//block_size, img.shape[1]//block_size))
    blocks_black = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if i % block_size == 0 and j % block_size == 0:
                blocks[i//block_size, j//block_size] = img[i, j]
                blocks_black[i, j] = img[i, j]
            #img[i, j] = 0
            
    #return blocks
    return blocks_black


#"""

# 1. Load the images from the folder and resize them to 100x100 grayscale
data = load_images_to_dict("images", 100)
images = np.array(data['data'])

print("Step 1: Images loaded..")

# 2. Get the unique labels and organize the data
labels = np.unique(data['label'])
y_data = np.array(data['label'])
x_img = np.zeros(len(images))
print("Step 2: Labels extracted..")

# 3. Feature extraction
for iter, image in enumerate(images):
    val, image = hog(image, cells_per_block=(3, 3), pixels_per_cell=(8, 8), visualize=True)
    if iter == 0:
        x_img = np.zeros((len(images), val.shape[0]))

    x_img[iter] = val

#block_size = 5
#x_data = np.array([divide_image_to_blocks(img, block_size) for img in x_img])

print("Step 3: Features extracted..")

# 4. Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_img, y_data, test_size=0.2, stratify=y_data)
#random_state=42
print("Step 4: Data split..")

# 5. Train the model
sgd = SGDClassifier(max_iter=100_000, tol=1e-1)
#random_state=42
sgd.fit(x_train, y_train)

print("Step 5: Model trained..")

# 6. Test the model
print(f"    Accuracy sgd: {sgd.score(x_test, y_test)}")
print("Step 6: Model tested..")

# 7: plot the results
y_pred = sgd.predict(x_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        title="Confusion matrix",
        ylabel='True label',
        xlabel='Predicted label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

fmt = '.2f'
thresh = cm.max() / 2.

plt.show()

#"""