from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pickle
from typing import Tuple
from typing import Union

def image_resize_to_grayscale(img_path: str, new_width: int) -> np.ndarray:
    img = Image.open(img_path).convert('L')
    new_img = img.resize((new_width, new_width))
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
                if file == ".DS_Store":
                    continue            
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

def save_model(model: object, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model(filename: str) -> object:
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

def extract_features_from_images(image_dict: dict) -> dict:
    for iter, image in enumerate(image_dict):
        val, image = hog(
            image, 
            cells_per_block=(3, 3), 
            pixels_per_cell=(8, 8), 
            visualize=True
            )

        if iter == 0:
            img_reduced = np.zeros((len(image_dict), val.shape[0]))
        img_reduced[iter] = val
    
    return img_reduced


def extract_features_from_image(image: Image, visualization=False):
    if visualization:
        image_reduced, visualization = hog(
            image, 
            cells_per_block=(3, 3), 
            pixels_per_cell=(8, 8), 
            visualize=True
            )
        return image_reduced, visualization
    else:
        image_reduced = hog(
            image, 
            cells_per_block=(3, 3), 
            pixels_per_cell=(8, 8), 
            )
        return image_reduced


def test_single_image(filename: str, model: object, image_width=100, visualization=False):
    image = image_resize_to_grayscale(filename, image_width)
    if visualization:
        image_reduced, image_original = extract_features_from_image(image, visualization=True)
        label = model.predict(image_reduced.reshape(1, -1))
        plt.imshow(image)
        plt.title(label)
        plt.show()
        return label
    else:
        image_reduced = extract_features_from_image(image, visualization=False)
        label = model.predict(image_reduced.reshape(1, -1))
        return label

#4 rzeczy
#jakaś definicja
#jakiś algorytm
#jakiś dowówd