import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from data import COVIData
import cv2
from model_v2 import Unet
from albumentations.pytorch import ToTensorV2
import torch.optim as optim

# Hyper-parameters
learning_rate = 0.001
epochs = 10
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = ''
mask_path = ''
out_channels = 4

# model
model = Unet(out_channels, device).to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Albumentations transformation
transform = A.Compose(
    [
        A.Resize(width=400, height=400),
        A.Rotate(limit=20, p=0.8, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=20, p=0.6),

        A.OneOf([
            A.Blur(blur_limit=6, p=0.5),
            A.ColorJitter(p=0.5)
        ], p=0.5),

        A.Normalize(mean=[0], std=[1], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

dataset = COVIData(image_path, mask_path, transforms=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# training baby
for epoch in range(epochs):
    print(f'Epochs [{epoch}/{epochs}]')
    losses = []
    for batch_idx, (train, label) in enumerate(loader):
        train = train.to(device)
        label = label.to(device)

        with torch.cuda.amp.autocast():
            score = model(train)
            loss = loss_f(score, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss)

    print(f'Loss {epoch} = {sum(losses)/len(losses)}')



