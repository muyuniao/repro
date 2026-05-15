import os
import glob
import json
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch

class HCIDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, seed=42, splits_file='data/splits.json'):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.classes = ['1930s', '1940s', '1950s', '1960s', '1970s']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Gather all images
        all_imgs = []
        all_labels = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = glob.glob(os.path.join(cls_dir, '*.jpg'))
            all_imgs.extend(imgs)
            all_labels.extend([self.class_to_idx[cls]] * len(imgs))
            
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        else:
            # Create splits
            X_train, X_temp, y_train, y_temp = train_test_split(all_imgs, all_labels, test_size=0.3, stratify=all_labels, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
            splits = {
                'train': X_train,
                'val': X_val,
                'test': X_test
            }
            os.makedirs(os.path.dirname(splits_file), exist_ok=True)
            with open(splits_file, 'w') as f:
                json.dump(splits, f)
                
        self.samples = splits[split]
        # map paths to labels by their folder name
        self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(p))] for p in self.samples]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class BagDataset(Dataset):
    def __init__(self, hci_dataset):
        self.hci_dataset = hci_dataset
        self.classes = hci_dataset.classes
        
        # Group indices by class
        self.class_indices = {i: [] for i in range(len(self.classes))}
        for idx, label in enumerate(hci_dataset.labels):
            self.class_indices[label].append(idx)
            
        # Determine number of bags as the max samples in any class
        self.num_bags = max(len(indices) for indices in self.class_indices.values())
        
    def __len__(self):
        return self.num_bags
        
    def __getitem__(self, idx):
        bag_imgs = []
        bag_labels = []
        for c in range(len(self.classes)):
            sample_idx = random.choice(self.class_indices[c])
            img, label = self.hci_dataset[sample_idx]
            bag_imgs.append(img)
            bag_labels.append(label)
            
        # stack into a tensor
        bag_imgs = torch.stack(bag_imgs)
        bag_labels = torch.tensor(bag_labels)
        
        return bag_imgs, bag_labels

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HCIDataset(data_dir, split='train', transform=train_transform)
    val_dataset = HCIDataset(data_dir, split='val', transform=val_transform)
    test_dataset = HCIDataset(data_dir, split='test', transform=val_transform)
    
    train_bag_dataset = BagDataset(train_dataset)
    
    # normal dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # bag dataloader
    train_bag_loader = DataLoader(train_bag_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_bag_loader
