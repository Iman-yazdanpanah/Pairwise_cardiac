# custom_dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import medical_transforms as tr
    

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.data1_files = self.data['image_id'].tolist()  # Assuming the CSV file has a column 'image' with paths
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0] + '.png')
        mask_name = os.path.join(self.mask_dir, self.data.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        sample = {'image': image, 'mask': mask }

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['mask'], idx
    def get_image_paths(self):
        return self.data1_files

        
crop_size=512
# Example usage
transform  = transforms.Compose([
            tr.FixScaleCrop(crop_size=crop_size),
            tr.Normalize(),
            tr.ToTensor()])

'''         transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor()
])'''
dataset = dataset = CustomDataset(csv_file='./train/train.csv', image_dir='./train/images', mask_dir='./train/masks', transform=transform)


# Create a DataLoader
#dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
