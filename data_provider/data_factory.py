import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from utils.timefeatures import time_features

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'BTCPrice': Dataset_Custom,
    'm4': Dataset_M4,
}

def custom_collate_fn(batch):
    """Custom collate function to handle numpy arrays and datetime objects"""
    transposed = list(zip(*batch))
    
    # Convert numpy arrays to tensors and stack them
    tensor_batch = [torch.stack([torch.from_numpy(np.array(s)) for s in samples]) 
                   for samples in transposed[:4]]
    
    # Convert datetime64 to string to avoid collate issues
    timestamp_batch = np.array([str(t) for t in transposed[4]])
    
    return tensor_batch[0], tensor_batch[1], tensor_batch[2], tensor_batch[3], timestamp_batch

def data_provider(args, flag):
    """
    Args:
        args: Configuration
        flag: Train/test/val flag
    """
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.data == 'm4':
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
        percent=args.percent
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    return data_set, data_loader
