from dataloaders.custom_transforms import Normalize, ToTensor, RandomHorizontalFlip, RandomRotate
from torch.utils.data import DataLoader
import custom_dataset
import os,torch
from torchvision import transforms
from dataloaders.datasets.custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    indices = [item[2] for item in batch]
    return images, masks, indices

def make_data_loader(args, **kwargs):
    if args.dataset == 'lits':
        train_set = lits.LitsDataloader(args, data_phase='train', margin=5)
        val_set = lits.LitsDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'pairwise_lits':
        train_set = pairwise_lits.LitsDataloader(args, data_phase='train', margin=5)
        val_set = pairwise_lits.LitsDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'chaos':
        train_set = chaos.ChaosDataloader(args, data_phase='train', margin=5)
        val_set = chaos.ChaosDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'pairwise_chaos':
        train_set = pairwise_chaos.ChaosDataloader(args, data_phase='train', margin=5)
        val_set = pairwise_chaos.ChaosDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif  args.dataset == 'custom1':
        transform = transforms.Compose([
            #RandomHorizontalFlip(),
            #RandomRotate(),
            Normalize(),
            ToTensor()
        ])
        #image_dir = os.path.join(args.data_root, 'images')
       # mask_dir = os.path.join(args.data_root, 'masks')
       # csv_file = os.path.join(args.data_root, 'data.csv')
        '''csv_file='./train/train.csv'
        image_dir='./train/images'
        mask_dir='./train/masks'
        
        dataset = CustomDataset(csv_file=csv_file, image_dir=image_dir, mask_dir=mask_dir, transform=transform)

        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=args.seed)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, test_idx)

        num_class = 2  # Update this based on your specific number of classes

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, 
                                collate_fn=custom_collate, **kwargs)
        test_loader = None
        
        return train_loader, val_loader, test_loader, num_class
'''     
        
        
        # image_dir = os.path.join(args.data_root, 'images')
        # mask_dir = os.path.join(args.data_root, 'masks')
        # csv_file = os.path.join(args.data_root, 'data.csv')
        csv_file = './train/train.csv'
        image_dir = './train/images'
        mask_dir = './train/masks'

        dataset = CustomDataset(csv_file=csv_file, image_dir=image_dir, mask_dir=mask_dir, transform=transform)

        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=args.seed)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, test_idx)

        num_class = 2  # Update this based on your specific number of classes

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, 
                                collate_fn=custom_collate, **kwargs)
        
        # Create an all_loader for all data
        all_loader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True, 
                                collate_fn=custom_collate, **kwargs)
        
        test_loader = None

        return train_loader, val_loader, test_loader, num_class, all_loader

    else:
        raise NotImplementedError
