import os
from tqdm import tqdm
import cv2
import numpy as np


def preprocess_pet_image_data(cat_path, dog_path, img_size=128):
    cat = cat_path
    dog = dog_path
    labels = {cat: 0, dog: 1}

    processed_img_data = []
    cat_count = 0
    dog_count = 0

    for label in labels:
        for file in tqdm(os.listdir(label)):
            try:
                path = os.path.join(label, file)

                # converting image to grayscale
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # resizing image to 128x128
                img = cv2.resize(img, (img_size, img_size))

                # converting image to numpy array and building dataset
                # img_array = np.array(img, dtype=np.float32)
                # label_array = np.array([labels[label]], dtype=np.int64).flatten()

                # print("Image shape:", img_array.shape)
                # print("Label shape:", label_array.shape)

                processed_img_data.append([np.array(img), np.eye(2)[labels[label]]])

                if label == cat:
                    cat_count += 1
                elif label == dog:
                    dog_count += 1

            except Exception as e:
                print(str(e))
                pass

    # shuffling dataset
    for i in range(10):  # Print the first 10 data points for inspection
        print(processed_img_data[i])
    processed_img_data = np.array(processed_img_data, dtype=object)
    print('Shape: ', processed_img_data.shape)
    np.random.shuffle(processed_img_data)
    print('Shape: ', processed_img_data.shape)
    np.save('processed_img_data.npy', processed_img_data)
    print('Processed image data saved to processed_img_data.npy')
    print('Cat datapoints: ', cat_count)
    print('Dog datapoints: ', dog_count)
    return processed_img_data

