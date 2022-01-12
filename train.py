import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import cv2
from data import COVIData
from model_v2 import Unet
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyper-parameters
learning_rate = 0.001
epochs = 200
batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = '/kaggle/input/covid-segmentation/images_medseg.npy'
mask_path = '/kaggle/input/covid-segmentation/masks_medseg.npy'
out_channels = 4

# model
model = Unet(1, out_channels).to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Albumentations transformation
transforms = A.Compose(
    [
        A.Resize(width=100, height=100),
        #A.Normalize(mean=[0], std=[1], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

# for testing
test = image_test[1]
mask = mask_test[1]
augmentation = transforms(image=test, mask=mask)
test_image = augmentation['image'].unsqueeze(0)
test_mask = augmentation['mask'].permute(2, 0, 1).unsqueeze(0)

# load the dataset
dataset = COVIData(image_path, mask_path, transforms=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# training baby
for epoch in range(epochs):
    print(f'Epochs [{epoch}/{epochs}]')
    losses = []
    
    # Testing
    
    model.eval()
    
    pred = model(test_image.to(device))
    pred = pred[0].permute(1, 2, 0).to('cpu').detach().numpy()
    
    image = test_image[0].permute(1, 2, 0)
    plt.imshow(image)
    print(image.shape)
    plt.show()
    
    mask = test_mask[0].clone().permute(1, 2 ,0)
    mask[..., 0] = mask[..., 0]*255
    mask[..., 1] = mask[..., 1]*255
    mask[..., 2] = mask[..., 2]*255
    mask[..., 3] = mask[..., 3]*255

    plt.imshow(mask[..., 1:4])
    plt.show()
    
    mask = np.expand_dims(np.argmax(pred, axis=2), axis=2) * 85
    print(mask.shape)
    #mask[..., 0] = mask[..., 0]*255
    #mask[..., 1] = mask[..., 1]*255
    #mask[..., 2] = mask[..., 2]*255
    #mask[..., 3] = mask[..., 3]*255

    plt.imshow(mask)
    plt.show()
    
    model.train()
    
    for batch_idx, (train, label) in enumerate(loader):
        train = train.to(device)
        label = label.to(device).permute(0, 3, 1, 2)

        with torch.cuda.amp.autocast():
            score = model(train)
            
            # reshaping for cross entropy loss
            score = score.reshape(score.shape[0], out_channels, -1)
            label = label.argmax(dim=1).reshape(score.shape[0], -1)

            # loss value
            loss = loss_f(score, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss)

    print(f'Loss {epoch} = {sum(losses)/len(losses)}')
