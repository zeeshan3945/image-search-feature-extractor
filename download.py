import torch
from torchvision import transforms, models

def download_model():
    # Load the pretrained VGG16 model
    print("downloading the model...")
    model = models.vgg16(pretrained=True)

    # Extract the features, pooling, flatten, and classifier layers
    #features = list(model.features)
    features = torch.nn.ModuleList(list(model.features))
    pooling = model.avgpool
    flatten = torch.nn.Flatten()
    fc = model.classifier[0]

    # Create a sequential model with the extracted layers
    model = torch.nn.Sequential(*features, pooling, flatten, fc)
    print("Done...")

if __name__ == "__main__":
    download_model()