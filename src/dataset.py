import lightning.pytorch as pl
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import medmnist
import torch
import torchio as tio
import math
import numpy as np
import joblib
import json
import tqdm
import os
from collections import defaultdict, Counter
import pickle
from einops import reduce
from torch.utils.data import WeightedRandomSampler
import warnings
from src.vectorizer import Vectorizer

dirname = os.path.dirname(__file__)
global_seed = json.load(open(os.path.join(dirname, '..', 'global_seed.json')))['global_seed']
root_dir = os.path.join(dirname, '../data')
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

torch.serialization.add_safe_globals([defaultdict, list])

class PathMnist(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for PathMnist dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, use_data_augmentation=False, batch_size=256, num_workers=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.statistics = {'mean': torch.mean, 'std': torch.std}
        self.global_stats_path = os.path.join(root_dir, f"pathmnist_stats_seed_{global_seed}.pt")
        self.global_stats = torch.load(self.global_stats_path, weights_only=True) if os.path.exists(self.global_stats_path) else defaultdict(list) 
        self.common_transforms = None

        self.prepare_data_transforms()
    
    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''

        self.test_transform = []

        if self.use_data_augmentation:
            self.train_transform = [
                v2.RandomHorizontalFlip(p=0.375),
                v2.RandomVerticalFlip(p=0.375),
            ]
        else:
            self.train_transform = []
    
    def prepare_data(self):
        train_data = medmnist.PathMNIST(root=root_dir, split='train', download=True)
        medmnist.PathMNIST(root=root_dir, split='val', download=True)
        medmnist.PathMNIST(root=root_dir, split='test', download=True)

        if not self.global_stats:
            print(f'Calculating {list(self.statistics.keys())} from train data')
            for img, _ in train_data:
                img = torchvision.transforms.ToTensor()(img)
                for stat_name, stat_func in self.statistics.items():
                    self.global_stats[stat_name].append(torch.Tensor([*reduce(img, 'c w h -> c () ()', stat_func)]))

            for stat_name, stat_func in self.statistics.items():
                self.global_stats[stat_name] = torch.stack(self.global_stats[stat_name]).mean(dim=0).squeeze().tolist()

            print(f'Saving {list(self.statistics.keys())} from train data')
            torch.save(self.global_stats, self.global_stats_path)

            del train_data
        
    def get_transform(self, split='train'):
        if split not in ['train', 'test']:
            raise NotImplementedError(f'{split} is not a valid split name')

        if split == 'train':
            split_transform = self.train_transform
        elif split == 'test':
            split_transform = self.test_transform
        
        if not self.global_stats:
            self.global_stats = torch.load(self.global_stats_path, weights_only=True)

        self.common_transform = [
            torchvision.transforms.ToTensor(),
            v2.Normalize(mean=self.global_stats['mean'], std=self.global_stats['std'])
            ]

        return v2.Compose([*split_transform, *self.common_transform])  

    def setup(self, stage=None):
        print('Transforming data')
        self.train = medmnist.PathMNIST(root=root_dir, split='train', download=True, transform=self.get_transform('train'))
        self.val = medmnist.PathMNIST(root=root_dir, split='val', download=True, transform=self.get_transform('test'))
        self.test = medmnist.PathMNIST(root=root_dir, split='test', download=True, transform=self.get_transform('test'))
        print('Data transformed.')
     
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)      

VOXEL_SPACING = (0.703125, 0.703125, 2.5)
CACHE_IMG_SIZE = [256, 256]
torch_io_message = (
                'Using TorchIO images without a torchio.SubjectsLoader in PyTorch >='
                ' 2.3 might have unexpected consequences, e.g., the collated batches'
                ' will be instances of torchio.Subject with 5D images. Replace'
                ' your PyTorch DataLoader with a torchio.SubjectsLoader so that'
                ' the collated batch becomes a dictionary, as expected. See'
                ' https://github.com/fepegar/torchio/issues/1179 for more'
                ' context about this issue.'
            )
warnings.filterwarnings("ignore", message=torch_io_message)
    
class NLST(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for NLST dataset. This will load the dataset, as used in https://ascopubs.org/doi/full/10.1200/JCO.22.01345.

        The dataset has been preprocessed for you fit on each CPH-App nodes NVMe SSD drives for faster experiments.
    """

    ## Voxel spacing is space between pixels in orig 512x512xN volumes
    ## "pixel_spacing" stored in sample dicts is also in orig 512x512xN volumes

    def __init__(
            self,
            num_channels=3,
            use_data_augmentation=False,
            batch_size=1,
            num_workers=0,
            nlst_metadata_path="/scratch/project2/nlst-metadata/full_nlst_google.json",
            valid_exam_path="/scratch/project2/nlst-metadata/valid_exams.p",
            nlst_dir="/scratch/project2/compressed",
            lungrads_path="/scratch/project2/nlst-metadata/nlst_acc2lungrads.p",
            num_images=200,
            max_followup=6,
            img_size = [256, 256],
            class_balance=False,
            group_keys=['race', 'educat', 'gender', 'age', 'ethnic'],
            clinical_features=[],
            feature_config=sorted(json.load(open("feature_config.json", "r"))['features']),
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_followup = max_followup

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        self.nlst_metadata_path = nlst_metadata_path
        self.nlst_dir = nlst_dir
        self.num_images = num_images
        self.img_size = img_size
        self.valid_exam_path = valid_exam_path
        self.class_balance = class_balance
        self.lungrads_path = lungrads_path
        self.group_keys = group_keys
        self.clinical_features = clinical_features
        self.criteria = 'lung_rads'
        self.feature_config = [x for x in feature_config if x in [*self.group_keys, *self.clinical_features]]
        self.vectorizer = Vectorizer(feature_config=self.feature_config, num_bins=5)
        self.name = 'NLST'

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''

        resample = tio.Resample(target=VOXEL_SPACING)
        padding = tio.CropOrPad(
            target_shape=tuple(CACHE_IMG_SIZE + [self.num_images]), padding_mode=0
        )

        # Base transforms applied to both train and test datasets
        base_transforms = [resample, padding]

        # Data augmentation transforms for training
        if self.use_data_augmentation:
            augmentation_transforms = [
                tio.RandomFlip(axes=('LR', 'AP', 'SI'), flip_probability=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=(-10, 10),
                    translation=(-5, 5),
                    isotropic=False,
                    image_interpolation='linear',
                    p=0.5
                ),
                # tio.RandomElasticDeformation(
                #     num_control_points=(7, 7, 7),
                #     max_displacement=(5.0, 5.0, 5.0),
                #     locked_borders=2,
                #     p=0.5
                # ),
                tio.RandomNoise(mean=0.0, std=(0, 0.1), p=0.25),
                # tio.RandomBiasField(coefficients=0.5, p=0.3),
            ]
            self.train_transform = tio.Compose(
                base_transforms + augmentation_transforms
            )
        else:
            self.train_transform = tio.Compose(base_transforms)

        # Test transforms (no augmentation)
        self.test_transform = tio.Compose(base_transforms)

        self.normalize = torchvision.transforms.Normalize(mean=[128.1722], std=[87.1849])

    def setup(self, stage=None):
        self.metadata = json.load(open(self.nlst_metadata_path, "r"))
        self.acc2lungrads = pickle.load(open(self.lungrads_path, "rb"))
        self.valid_exams = set(torch.load(self.valid_exam_path, weights_only=True))
        self.train, self.val, self.test = [], [], []

        for mrn_row in tqdm.tqdm(self.metadata, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            dataset = {"train": self.train, "dev": self.val, "test": self.test}[split]

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():

                    exam_str = "{}_{}".format(exam_dict["exam"], series_id)

                    if exam_str not in self.valid_exams:
                        continue


                    exam_int = int(
                        "{}{}{}".format(int(pid), int(exam_dict["screen_timepoint"]), int(series_id.split(".")[-1][-3:]))
                    )

                    y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, exam_dict["screen_timepoint"])

                    # add group info
                    group_info = {group_key: pt_metadata[group_key][0] for group_key in self.group_keys} if self.group_keys else {}

                    # add clinical features
                    clinical_features = {k: pt_metadata[k][0] for k in self.clinical_features} if self.clinical_features else {}

                    sample_kwargs = {**group_info, **clinical_features}

                    sample = {
                        "pid": pid,
                        "exam_str": exam_str,
                        "exam_int": exam_int,
                        "path": os.path.join(self.nlst_dir, exam_str + ".pt"),
                        "y": y,
                        "y_seq": y_seq, # size: (self.maxfollowup, )
                        "y_mask": y_mask, # size: (self.maxfollowup, )
                        "time_at_event": time_at_event, # int
                        # lung_rads 0 indicates LungRads 1 and 2 (negative), 1 indicates LungRads 3 and 4 (positive)
                        # Follows "Pinsky PF, Gierada DS, Black W, et al: Performance of lung-RADS in the National Lung Screening Trial: A retrospective assessment. Ann Intern Med 162: 485-491, 2015"
                        "lung_rads": self.acc2lungrads[exam_int],
                        **sample_kwargs
                    }

                    dataset.append(sample)


        self.fit_vectorizer(self.train)

        NLST_kwargs = dict(normalize=self.normalize, 
                           img_size=self.img_size, 
                           num_images=self.num_images, 
                           num_channels=self.num_channels, 
                           group_keys=self.group_keys,
                           clinical_features=self.clinical_features)
        
        if self.class_balance:
            # calculate class sample count for each split
            if stage == 'fit':
                self.train_sampler = WeightedRandomSampler(self.get_samples_weight(self.train), num_samples=300, replacement=True)
                # self.val_sampler = WeightedRandomSampler(self.get_samples_weight(self.val), num_samples=300, replacement=False)
            
            # if stage == 'validate':
            #     self.val_sampler = WeightedRandomSampler(self.get_samples_weight(self.val), num_samples=len(self.val), replacement=False)

            # if stage in ['test', 'predict']:
            #     self.test_sampler = WeightedRandomSampler(self.get_samples_weight(self.test), num_samples=len(self.test), replacement=False)

        if stage == 'fit':
            self.train = NLST_Dataset(self.train, self.train_transform, **NLST_kwargs)
            self.val = NLST_Dataset(self.val, self.test_transform, **NLST_kwargs)
        if stage == 'validate':
            self.val = NLST_Dataset(self.val, self.test_transform, **NLST_kwargs)
        if stage in ['test', 'predict']:
            self.test = NLST_Dataset(self.test, self.test_transform, **NLST_kwargs)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.max_followup
        y_seq = np.zeros(self.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (self.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.val_sampler)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.test_sampler)

    @staticmethod
    def get_samples_weight(data):
        target = np.array([int(sample['y']) for sample in data])
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        return samples_weight

    def fit_vectorizer(self, data):
        self.vectorizer.fit(data)

class NLST_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for NLST dataset. Loads preprocesses data from disk and applies data augmentation. Generates masks from bounding boxes stored in metadata..
    """

    def __init__(self, dataset, transforms, normalize, img_size=[128, 128], num_images=200, num_channels=1, group_keys=[], clinical_features=[]):
        self.dataset = dataset
        self.transform = transforms
        self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images
        self.num_channels = num_channels
        self.group_keys = group_keys
        self.clinical_features = clinical_features

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_path = self.dataset[idx]['path']
        sample = joblib.load(sample_path + ".z")
        orig_pixel_spacing = torch.diag(torch.tensor(sample['pixel_spacing'] + [1]))
        num_slices = sample['x'].size(0)  # D

        # Determine cancer laterality (keep for future localization)
        right_side_cancer = sample['cancer_laterality'][0] == 1 and sample['cancer_laterality'][1] == 0
        left_side_cancer = sample['cancer_laterality'][1] == 1 and sample['cancer_laterality'][0] == 0

        # TODO: You can modify the data loading of the bounding boxes to suit your localization method.
        # Hint: You may want to use the "cancer_laterality" field to localize the cancer coarsely.

        # Handle bounding boxes and masks
        if not sample['has_localization']:
            sample['bounding_boxes'] = None

        mask = self.get_scaled_annotation_mask(sample['bounding_boxes'], CACHE_IMG_SIZE + [num_slices])

        # Create TorchIO subject
        subject = tio.Subject({
            'x': tio.ScalarImage(tensor=sample['x'].unsqueeze(0).to(torch.double), affine=orig_pixel_spacing),
            'mask': tio.LabelMap(tensor=mask.to(torch.double), affine=orig_pixel_spacing)
        })

        '''
            TorchIO will consistently apply the data augmentations to the image and mask, so that they are aligned. Note, the 'bounding_boxes' item will be wrong after after random transforms (e.g. rotations) in this implementation. 
        '''
        try:
            subject = self.transform(subject)
        except:
            raise Exception("Error with subject {}".format(sample_path))

        # Extract data      
        x = subject['x']['data'].to(torch.float)  # Shape: [C, W, H, D]
        mask = subject['mask']['data'].to(torch.float)

        # Permute x and mask to [C, D, H, W]
        x = x.permute(0, 3, 2, 1)  # From [C, W, H, D] to [C, D, H, W]
        mask = mask.permute(0, 3, 2, 1)

        # Normalize volume
        x = self.normalize(x)  # Custom normalization (mean=0, std=1)
        # print(f"Original sample['x'] shape: {x.shape}")
        # Add batch dimension
        x = x.unsqueeze(0)  # Shape: [1, C, D, H, W]
        mask = mask.unsqueeze(0)
        # print(f"Shape after adding batch dimension: {x.shape}")
        # Resize x and mask to fixed size (e.g., D=32, H=224, W=224)
        D_size, H_size, W_size = 32, 224, 224
        # D_size, H_size, W_size = 16, 112, 112
        x = F.interpolate(x, size=(D_size, H_size, W_size), mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=(D_size, H_size, W_size), mode='nearest')  # Use 'nearest' for masks

        # print(f"Shape after interpolation: {x.shape}")
        # Remove batch dimension
        x = x.squeeze(0)  # Shape: [C, D_size, H_size, W_size]
        mask = mask.squeeze(0)
        # print(f"Shape after squeezing batch dimension: {x.shape}")
        # Expand channels if needed
        if self.num_channels == 3:
            x = x.repeat(3, 1, 1, 1)  # From (1, D, H, W) to (3, D, H, W)
            # print(f"Shape after channel repeat: {x.shape}")

        # get group info
        group_info = {k:torch.tensor([v]) for k,v in self.dataset[idx].items() if k in self.group_keys} if self.group_keys else {}

        # get clinical features
        clinical_features = {k:torch.tensor([v]) for k,v in self.dataset[idx].items() if k in self.clinical_features} if self.clinical_features else {}
        
        sample_dict_kwargs = {**group_info, **clinical_features}

        # Prepare the sample
        sample_dict = {
            'x': x,  # Shape: (C, D, H, W)
            'mask': mask,  # Shape: (1, D, H, W)
            'y': torch.tensor(sample['y'], dtype=torch.long),
            'y_seq': torch.tensor(self.dataset[idx]['y_seq'], dtype=torch.float),
            'y_mask': torch.tensor(self.dataset[idx]['y_mask'], dtype=torch.float),
            'lung_rads': torch.tensor([self.dataset[idx]['lung_rads']], dtype=torch.int),
            'time_at_event': torch.tensor([self.dataset[idx]['time_at_event']], dtype=torch.int),
            'pid': torch.tensor([int(self.dataset[idx]['pid'])], dtype=torch.int),
            **sample_dict_kwargs
        }
        # print(f"y_seq shape: {sample_dict['y_seq'].shape}")

        # Remove unnecessary items
        del sample['bounding_boxes']

        return sample_dict

    def get_scaled_annotation_mask(self, bounding_boxes, img_size=[128,128, 200]):
        """
        Construct bounding box masks for annotations.

        Args:
            - bounding_boxes: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
            - img_size per slice
        Returns:
            - mask of same size as input image, filled in where bounding box was drawn. If bounding_boxes = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
        """
        H, W, Z = img_size
        if bounding_boxes is None:
            return torch.zeros((1, Z, H, W))

        masks = []
        for slice in bounding_boxes:
            slice_annotations = slice["image_annotations"]
            slice_mask = np.zeros((H, W))

            if slice_annotations is None:
                masks.append(slice_mask)
                continue

            for annotation in slice_annotations:
                single_mask = np.zeros((H, W))
                x_left, y_top = annotation["x"] * W, annotation["y"] * H
                x_right, y_bottom = (
                    min( x_left + annotation["width"] * W, W-1),
                    min( y_top + annotation["height"] * H, H-1),
                )

                # pixels completely inside bounding box
                x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
                x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

                # excess area along edges
                dx_left = x_quant_left - x_left
                dx_right = x_right - x_quant_right
                dy_top = y_quant_top - y_top
                dy_bottom = y_bottom - y_quant_bottom

                # fill in corners first in case they are over-written later by greater true intersection
                # corners
                single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
                single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
                single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
                single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

                # edges
                single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
                single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
                single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
                single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

                # completely inside
                single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

                # in case there are multiple boxes, add masks and divide by total later
                slice_mask += single_mask
                    
            masks.append(slice_mask)

        return torch.Tensor(np.array(masks)).unsqueeze(0)

    def get_summary_statement(self):
        num_patients = len(set([d['pid'] for d in self.dataset]))
        num_cancer = sum([d['y'] for d in self.dataset])
        num_cancer_year_1 = sum([d['y_seq'][0] for d in self.dataset])
        return "NLST Dataset. {} exams ({} with cancer in one year, {} cancer ever) from {} patients".format(len(self.dataset), num_cancer_year_1, num_cancer, num_patients)