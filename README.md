# Covid19-CT-ImageSegmentation

The model has created to solve the COVID19 CT scan (Human Lungs) Image segmentation task. Here I have used the Padded Convnet UNET (Input: 3x3x3, output: 3x3xno_classes) architecture for training.

The kaggle dataset: https://www.kaggle.com/c/covid-segmentation

# Loss and Accuracy Polts 

Used 200+ Images for training and 60+ for testing 
I used the data augmentation with albumentation to increase the number of training set for better model performance

![Screenshot (75)](https://github.com/user-attachments/assets/457e18cd-a5e2-4ba7-99cf-ca9c8834adb1)

# The Augmented dataset 

![Screenshot (94)](https://github.com/user-attachments/assets/ef9f2f2a-bd13-4b75-93f1-8d706da1b68e)

# Results

The Result I got after the training are.

Sections are: {original, labeles, prediction}


![Screenshot (95)](https://github.com/user-attachments/assets/860fd275-19c8-4031-8d7f-838dcc48b6ef)
