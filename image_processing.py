import os
import tqdm
import cv2
import numpy as np


def preprocess_pet_image_data(cat_path, dog_path, img_size=128):
    cats = cat_path
    dogs = dog_path
    labels = {cats: 0, dogs: 1}

    processed_img_data = []
    cat_count = len(os.listdir(cats))
    dog_count = len(os.listdir(dogs))

    for label in labels:
        for file in tqdm(os.listdir(label)):
            path = os.path.join(label, file)

            # converting image to grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # resizing image to 128x128
            img = cv2.resize(img, (img_size, img_size))

            # converting image to numpy array and building dataset
            processed_img_data.append([np.array(img), np.eye(2)[labels[label]]])

    # shuffling dataset
    np.random.shuffle(processed_img_data)
    np.save('processed_img_data.npy', processed_img_data)
    print('Cat datapoints: ', cat_count)
    print('Dog datapoints: ', dog_count)
    return processed_img_data

