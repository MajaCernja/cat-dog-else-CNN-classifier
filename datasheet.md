# Datasheet

## Motivation

- For what purpose was the dataset created? 
  - Research and learning.
- Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset? 
  - The dataset was created by Microsoft Research and provided by Kaggle.

 
## Composition

- What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? 
  - The dataset contains images of cats and dogs.
- How many instances of each type are there? 
  - 12500 images of cats and 12500 images of dogs.
- Is there any missing data?
  - Some images were not successfully processed with the OpenCV library and were removed from the dataset.
- Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by    doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? 
  - No to author's knowledge.
- Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
  - Unlikely, but possible. The dataset contains images of cats and dogs, which may cause anxiety to some people.

## Collection process

- How was the data acquired? 
  - Unknown to datasheet author.
- If the data is a sample of a larger subset, what was the sampling strategy? 
    - The model was trained on a subset of 1280 randomly selected images from the Kaggle Cats and Dogs Dataset. It is unknown to author if Kaggle dataset itself is a subset.
- Over what time frame was the data collected?
  - Unknown to datasheet author.

## Preprocessing/cleaning/labelling

- Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section. 
  - Image data was processed with the OpenCV library to resize images to 128x128 pixels and convert them to grayscale.
- Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? 
  - Raw images can be found in the Kaggle Cats and Dogs Dataset.
 
## Uses

- What other tasks could the dataset be used for? 
  - The dataset could be used to train other image classifiers.
- Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms? 
  - Not to author's knowledge.
- Are there tasks for which the dataset should not be used? If so, please provide a description.
  - The dataset should not be used to train a model to classify images of anything other than cats and dogs.

## Distribution

- How has the dataset already been distributed? 
  - The dataset was distributed by Microsoft Research and Kaggle.It is available for download on Kaggle https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data.
- Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?  
 - Yes, the dataset is subject to the Microsoft Research Data License Agreement for Kaggle Cats and Dogs Dataset.

## Maintenance

- Who maintains the dataset? 
  - The dataset is static and does not require maintenance.