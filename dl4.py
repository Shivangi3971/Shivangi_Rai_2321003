import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch

# Transformations for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for models like VGG and ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pre-trained models
])

# Loading the SVHN dataset
train_dataset = SVHN(root='./data', split='train', transform=transform, download=True)
test_dataset = SVHN(root='./data', split='test', transform=transform, download=True)

# Data loaders for the training and test set
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)



def modify_model(model):
    # Modify the last layer based on the model type
    if isinstance(model, models.ResNet):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for SVHN
    elif isinstance(model, models.AlexNet) or isinstance(model, models.VGG):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    return model

# Load and modify pretrained models
alexnet = modify_model(models.alexnet(pretrained=True))
vgg16 = modify_model(models.vgg16(pretrained=True))
resnet18 = modify_model(models.resnet18(pretrained=True))
resnet50 = modify_model(models.resnet50(pretrained=True))
resnet101 = modify_model(models.resnet101(pretrained=True))




def train_model(model, train_loader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')



models = {'AlexNet': alexnet, 'VGG16': vgg16, 'ResNet18': resnet18, 'ResNet50': resnet50, 'ResNet101': resnet101}

for name, model in models.items():
    print(f'Training {name}...')
    train_model(model, train_loader)
    print(f'Evaluating {name}...')
    evaluate_model(model, test_loader)
