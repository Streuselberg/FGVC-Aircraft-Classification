from os import path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transforms import train_transform, val_transform

# TODO:
# you're going to need to adjust to wherever you have the FGVC dataset

data_base_dir = '../Engine/Data/FGVC'
images_dir = path.join(data_base_dir, 'images')

BATCH_SIZE = 16


class FGVAircraftDataset(Dataset):
    def __init__(self, txt_file, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        self.image_data = [(line.split()[0], ' '.join(line.split()[1:])) for line in lines]
        self.image_paths = [data[0] + '.jpg' for data in self.image_data]
        self.labels = [data[1] for data in self.image_data]
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_training_datasets():
    """create a mapping of expert names->training data set
    :returns mapping
    """
    return {
        'family': FGVAircraftDataset(txt_file=path.join(data_base_dir, 'images_family_train.txt'),
                                     image_dir=images_dir,
                                     transform=train_transform),
        'manufacturer': FGVAircraftDataset(txt_file=path.join(data_base_dir, 'images_manufacturer_train.txt'),
                                           image_dir=images_dir, transform=train_transform),
        'variant': FGVAircraftDataset(txt_file='../Engine/Data/FGVC/images_variant_train.txt', image_dir=images_dir,
                                      transform=train_transform)
    }


def create_label_mappings(dataset):
    return {
        'family': dataset['family'].label_to_idx,
        'manufacturer': dataset['manufacturer'].label_to_idx,
        'variant': dataset['variant'].label_to_idx
    }


def create_validation_datasets():
    return {
        'family': FGVAircraftDataset(txt_file='../Engine/Data/FGVC/images_family_val.txt', image_dir=images_dir,
                                     transform=val_transform),
        'manufacturer': FGVAircraftDataset(txt_file='../Engine/Data/FGVC/images_manufacturer_val.txt',
                                           image_dir=images_dir, transform=val_transform),
        'variant': FGVAircraftDataset(txt_file='../Engine/Data/FGVC/images_variant_val.txt', image_dir=images_dir,
                                      transform=val_transform)
    }
