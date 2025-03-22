from os import path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from fgvc_config import ExpertName, Config
from transforms import train_transform, val_transform


class FGVAircraftDataset(Dataset):
    def __init__(self, txt_file, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(txt_file, 'r', encoding='utf-8') as f:
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


def create_training_datasets(config: Config):
    """create a mapping of expert names->training data set
    this
    :param config:
    :returns mapping
    """
    return {
        expert.name: FGVAircraftDataset(txt_file=path.join(config.DATA_DIR, f"images_{expert.name}_train.txt"),
                                        image_dir=config.IMAGES_DIR,
                                        transform=train_transform)
        for expert in ExpertName
    }


def create_validation_datasets(config: Config) -> dict[str, FGVAircraftDataset]:
    """
    creates a mapping of expert names->validation data set.
    NOTE: this assumes that you have a train and validation set in the dataset's directory
    with the following name format: "images_{expert_name}_val.txt`
    :param config: Config
    :return: mapping
    """
    return {
        expert.name: FGVAircraftDataset(txt_file=path.join(config.DATA_DIR, f"images_{expert.name}_val.txt"),
                                        image_dir=config.IMAGES_DIR,
                                        transform=val_transform)
        for expert in ExpertName
    }


def create_label_mappings(dataset: FGVAircraftDataset) -> dict[str, dict[str, int]]:
    """
    create a mapping of expert names -> {label: class_id}
    :param dataset: FGVAircraftDataset
    :return: label mappings by expert
    """
    return {
        name.name: dataset[name.name].label_to_idx for name in ExpertName
    }
