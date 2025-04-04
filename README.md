# Covid19-CT-ImageSegmentation

The model has created to solve the COVID19 CT scan (Human Lungs) Image segmentation task. Here I have used the Padded Convnet UNET (Input: 3x3x3, output: 3x3xno_classes) architecture for training.

The kaggle dataset: https://www.kaggle.com/c/covid-segmentation

The Result I got after the training are.

Sections are: {original, labeles, prediction}

# Loss and Accuracy Polts 

Used 200+ Images for training and 60+ for testing 
I used the data augmentation with albumentation to increase the number of training set for better model performance

![Screenshot (75)](https://github.com/user-attachments/assets/457e18cd-a5e2-4ba7-99cf-ca9c8834adb1)

# The Augmented dataset 

![Screenshot (94)](https://github.com/user-attachments/assets/ef9f2f2a-bd13-4b75-93f1-8d706da1b68e)

# Results
111...............

![Original](https://user-images.githubusercontent.com/75822824/149130504-bb95baca-843c-49e4-b0e0-d6a36dd302cc.png)
![labeled](https://user-images.githubusercontent.com/75822824/149130506-66ef808f-3666-40f8-a5df-7061a6c199ed.png)
![prediction](https://user-images.githubusercontent.com/75822824/149130509-64c2430b-26aa-4003-b0bd-4b5d40c9d9e4.png)

222...............

![4](https://user-images.githubusercontent.com/75822824/149133795-5b1ae6a2-61f3-48a5-b34b-2543d2413ea4.png)
![5](https://user-images.githubusercontent.com/75822824/149133807-bc6fba3b-fc07-4537-afb3-9c645bda2bde.png)
![6](https://user-images.githubusercontent.com/75822824/149133817-b3d5cbad-b313-44ac-82ba-d8b7a55f372b.png)
