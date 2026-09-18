import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create ResNet18
model = models.resnet18(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)


# Load our trained CT model
model.load_state_dict(
    torch.load(
        "ct/models/ct_resnet18.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# CT prediction function
def predict_ct(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    classes = [
        "COVID",
        "non-COVID"
    ]

    label = classes[predicted.item()]

    return {
        "label": label,
        "confidence": round(
            confidence.item() * 100,
            2
        )
    }
