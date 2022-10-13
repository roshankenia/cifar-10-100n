import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import optim, nn
from torchvision import models, transforms

import os
import sys
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


model = models.vgg16(pretrained=True)


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
new_model = new_model.cuda()


# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()
])

# Will contain the feature
features = []

# Iterate each image
for i in tqdm(sample_submission.ImageID):
    # Set the image path
    path = os.path.join('data', 'test', str(i) + '.jpg')
    # Read the file
    img = cv2.imread(path)
    # Transform the image
    img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = new_model(img)
        # Convert to NumPy Array, Reshape it, and save it to features variable
    features.append(feature.cpu().detach().numpy().reshape(-1))

# Convert to NumPy Array
features = np.array(features)


# Initialize the model
model = KMeans(n_clusters=5, random_state=42)

# Fit the data into the model
model.fit(features)

# Extract the labels
labels = model.labels_

print(labels)  # [4 3 3 ... 0 0 0]


sample_submission = pd.read_csv('sample_submission.csv')
new_submission = sample_submission
new_submission['label'] = labels
new_submission.to_csv('submission_1.csv', index=False)
