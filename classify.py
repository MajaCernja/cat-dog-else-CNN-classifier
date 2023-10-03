from helpers.image_processing import preprocess_pet_image_data

cat_path = 'PetImages/Cat'
dog_path = 'PetImages/Dog'

if __name__ == '__main__':
    preprocess_pet_image_data(cat_path, dog_path)
