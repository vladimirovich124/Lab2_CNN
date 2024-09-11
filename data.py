import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_class_labels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def create_directory_structure(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def combine_batches_physically(batch_dir, selected_batches):
    combined_dir = f"combined_batches_{'-'.join(map(str, selected_batches))}"
    if os.path.exists(combined_dir):
        logger.info(f"Using an existing directory: {combined_dir}")
        return combined_dir

    create_directory_structure(combined_dir, ["images", "test/images"])
    test_labels_file = os.path.join(combined_dir, "test", "labels.txt")
    open(test_labels_file, 'w').close()

    logger.info(f"Combining batches {', '.join(map(str, selected_batches))} в {combined_dir}")
    for batch_idx in selected_batches:
        copy_batch_data(batch_dir, combined_dir, batch_idx)

    return combined_dir

def copy_batch_data(batch_dir, combined_dir, batch_idx):
    batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
    for img_file in os.listdir(os.path.join(batch_path, "images")):
        shutil.copy(os.path.join(batch_path, "images", img_file),
                    os.path.join(combined_dir, "images", img_file))

    append_labels(os.path.join(batch_path, "labels.txt"),
                  os.path.join(combined_dir, "labels.txt"))

def append_labels(src_file, dst_file):
    with open(dst_file, "a") as dst, open(src_file, "r") as src:
        dst.writelines(src.readlines())

def split_combined_dataset(combined_dir, train_val_split):
    dataset = CustomDataset(combined_dir)
    train_size = int(train_val_split * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    test_dataset = CustomDataset(os.path.join(combined_dir, "test"))
    return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.labels, self.classes = self.load_labels()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_labels(self):
        labels, classes = {}, []
        with open(os.path.join(self.data_dir, "labels.txt"), "r") as f:
            for line in f:
                idx, label = line.strip().split(",")
                labels[idx] = label
                if label not in classes:
                    classes.append(label)
        return labels, classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "images", self.image_files[idx])
        image = self.transform(Image.open(img_path))
        label = self.labels[self.image_files[idx].split("_")[1].split(".")[0]]
        return image, label

def load_cifar10(root):
    return datasets.CIFAR10(root=root, train=True, download=True, transform=default_transforms())

def default_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def split_dataset_into_batches(dataset, num_batches, batch_dir):
    if os.path.exists(batch_dir):
        logger.info(f"Using an existing batch directory: {batch_dir}")
        return

    os.makedirs(batch_dir, exist_ok=True)
    logger.info(f"Dividing the data set into {num_batches} batches in {batch_dir}")

    batch_size = len(dataset) // num_batches
    for i, (img, label) in enumerate(dataset):
        batch_idx = i // batch_size + 1
        save_image_and_label(batch_dir, img, label, i, batch_idx)

def save_image_and_label(batch_dir, img, label, idx, batch_idx):
    batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
    create_directory_structure(batch_path, ["images"])

    img_path = os.path.join(batch_path, "images", f"img_{idx}.png")
    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(img_path)

    with open(os.path.join(batch_path, "labels.txt"), "a") as f:
        f.write(f"{idx},{label}\n")

def get_data_loaders(config, root):
    cifar10 = load_cifar10(root)
    batch_dir = "batches"

    split_dataset_into_batches(cifar10, config.num_batches, batch_dir)
    combined_dir = combine_batches_physically(batch_dir, config.selected_batches)

    original_dataset = CustomDataset(combined_dir)
    train_dataset, val_dataset, test_dataset = split_combined_dataset(combined_dir, config.train_val_split)

    return create_data_loaders(train_dataset, val_dataset, test_dataset, original_dataset, config.batch_size)

def create_data_loaders(train_dataset, val_dataset, test_dataset, original_dataset, batch_size):
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, original_dataset)),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, original_dataset)),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, original_dataset))
    )

def collate_fn(batch, dataset):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor([dataset.classes.index(label) for label in labels])