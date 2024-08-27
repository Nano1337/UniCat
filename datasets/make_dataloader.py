import torch
import copy
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.distributed as dist
from timm.data.random_erasing import RandomErasing

from .preprocessing import RandomGrayscalePatchReplacement
from .bases import ImageDataset, MergedImageDataset
from .sampler import RandomIdentitySampler, RandomIdentityBatchSampler
from .sampler_ddp import RandomIdentitySampler_DDP, RandomIdentityBatchSampler_DDP
from .corrupt_res import ResCorrupter
from .rgbn300 import RGBN300
from .rgbnt100 import RGBNT100
from .rgbnt201 import RGBNT201
from .flare import Flare
from .merge_datasets import merge_datasets

# Dataset factory
__factory = {
    'rgbn300': RGBN300,
    'rgbnt100': RGBNT100,
    'rgbnt201': RGBNT201,
    'flare': Flare
    }

def train_collate_fn(batch):
    """
    Collate function for training data loader.
    
    Args:
        batch: A list of tuples (img, pid, camid, img_path)
    
    Returns:
        A tuple of tensors for (imgs, pids, camids, camid_tensor, img_paths)
    """
    imgs, pids, camids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camid_tensor = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camid_tensor, img_paths

def val_collate_fn(batch):
    """
    Collate function for validation data loader.
    
    Args:
        batch: A list of tuples (img, pid, camid, img_path)
    
    Returns:
        A tuple of tensors for (imgs, pids, camids, camid_tensor, img_paths)
    """
    imgs, pids, camids, img_paths = zip(*batch)
    camid_tensor = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camid_tensor, img_paths

def make_dataloader(cfg):
    """
    Create data loaders for training and validation.
    
    Args:
        cfg: Configuration object containing dataset and data loader parameters
    
    Returns:
        train_loader: DataLoader for training
        train_loader_normal: DataLoader for training set with validation transforms
        val_loader: DataLoader for validation
        len(all_query): Number of query samples
        num_mode: Number of modes in the dataset
        num_classes: Total number of identity classes
        cam_num: Total number of cameras
    """
    if cfg.INPUT.RE_MODE == 'timm':
        re = RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
    else:
        re = T.RandomErasing(p=cfg.INPUT.RE_PROB, value='random')
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            RandomGrayscalePatchReplacement(probability=cfg.INPUT.GS_PROB),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            re
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    datasets = []
    all_train_transforms = []
    all_val_transforms = []
    num_classes = cam_num = 0
    num_mode = None
    for n, root, res_mode in zip(cfg.DATASETS.NAMES, cfg.DATASETS.ROOT_DIR, cfg.INPUT.RES_MODE):
        info_ = n.split('__')
        if len(info_) == 1:
            name = info_[0]
            mode = None
        else:
            name, mode = info_

        datasets.append(__factory[name](root=root, mode=mode, create_cam=cfg.TEST.CREATE_CAM_EVAL))
        num_classes += datasets[-1].num_train_pids
        cam_num += datasets[-1].num_train_cams
        assert num_mode is None or num_mode == datasets[-1].num_mode
        num_mode = datasets[-1].num_mode
        n_train_transforms = copy.deepcopy(train_transforms)
        n_val_transforms = copy.deepcopy(val_transforms)
        if res_mode != 'false':
            n_train_transforms.transforms.append(ResCorrupter(mode=res_mode))
            n_val_transforms.transforms.append(ResCorrupter(mode=res_mode))

        all_train_transforms.append(n_train_transforms)
        all_val_transforms.append(n_val_transforms)
    
    # handle multi-dataset training
    if len(datasets) > 1:
        print('MERGING')
        assert not cfg.MODEL.SIE_CAMERA and not cfg.MODEL.SIE_VIEW  # if we're merging datasets then camera/view info cannot be used during training
        
        print('Total merged train classes:', num_classes)
        all_train, all_query, all_gallery, train_ind_to_dataset, query_ind_to_dataset, gal_ind_to_dataset = merge_datasets(datasets)
        print(len(all_train), len(all_query), len(all_gallery))

        val_ind_to_dataset = query_ind_to_dataset
        offset = len(val_ind_to_dataset)
        for ind, d in gal_ind_to_dataset.items():
            val_ind_to_dataset[ind + offset] = d

        train_set = MergedImageDataset(all_train, all_train_transforms, train_ind_to_dataset)
        train_set_normal = MergedImageDataset(all_train, all_val_transforms, train_ind_to_dataset)
        val_set = MergedImageDataset(all_query + all_gallery, all_val_transforms, val_ind_to_dataset)
    else:
        all_train, all_query, all_gallery = datasets[0].train, datasets[0].query, datasets[0].gallery
        train_set = ImageDataset(datasets[0].train, all_train_transforms[0])
        train_set_normal = ImageDataset(datasets[0].train, all_val_transforms[0])
        val_set = ImageDataset(datasets[0].query + datasets[0].gallery, all_val_transforms[0])

    # handle various ways of creating contrastive triplets for training, different samplers involved
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            if cfg.MODEL.ID_HARD_MINING:
                data_sampler = RandomIdentityBatchSampler_DDP(all_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, num_classes)
            else:
                data_sampler = RandomIdentitySampler_DDP(all_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, random_order=cfg.DATALOADER.TRAIN_RANDOM_ORDER)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if cfg.MODEL.ID_HARD_MINING:
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentityBatchSampler(all_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, num_classes),
                    num_workers=num_workers, collate_fn=train_collate_fn
                )
            else:
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(all_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, random_order=cfg.DATALOADER.TRAIN_RANDOM_ORDER),
                    num_workers=1, collate_fn=train_collate_fn
                )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(all_query), num_mode, num_classes, cam_num
