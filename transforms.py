from PIL.Image import Image
from PIL import Image
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_and_preprocess_image(image_path: str) -> Image:
    if image_path is None:
        return ValueError(image_path)
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")
