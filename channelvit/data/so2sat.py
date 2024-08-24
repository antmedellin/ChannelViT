from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset
import h5py


class So2Sat(S3Dataset):
    """So2Sat"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        path: str,
        split: -1,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        channels: Union[List[int], None],
        channel_mask: bool = False,
        scale: float = 1,
        dataset_keys: List[str] = ['sen1', 'sen2']  # Default to ['sen1', 'sen2']
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        # read the cyto mask df
        # self.df = pd.read_parquet(path)
        if path.endswith('.parquet'):
            self.df = pd.read_parquet(path)
        elif path.endswith('.h5'):
            # import h5py
            # with h5py.File(path, 'r') as f:
            #     def print_attrs(name, obj):
            #         print(name)
            #         for key, val in obj.attrs.items():
            #             print(f"    {key}: {val}")

            #     f.visititems(print_attrs)
            
            # self.df = pd.read_hdf(path)
            # self.df = {}
            # with h5py.File(path, 'r') as f:
            #     for key in dataset_keys:
            #         self.df[key] = pd.DataFrame(f[key][:])
                    
            self.df = {}
            with h5py.File(path, 'r') as f:
                for key in dataset_keys:
                    self.df[key] = f[key][:]
                self.df['label'] = f['label'][:]
                
        else:
            raise ValueError(f"Unsupported file format: {path}")

        self.channels = torch.tensor([c for c in channels])
        self.scale = scale  # scale the input to compensate for input channel masking

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=transform_cfg.normalization.mean,
            normalization_std=transform_cfg.normalization.std,
        )

        self.channel_mask = channel_mask
        


    def __getitem__(self, index):
        # row = self.df.iloc[index]
        # img_chw = self.get_image(row["path"]).astype("float32")
        # if img_chw is None:
        #     return None
        
        
        
        # sen1:	N*32*32*8	
        # sen2:	N*32*32*10
        # label:	N*17 (one-hot coding)

        img_sen1 = self.df['sen1'][index].astype("float32")
        img_sen2 = self.df['sen2'][index].astype("float32")

        if img_sen1 is None or img_sen2 is None:
            return None

        # Concatenate sen1 and sen2 along the channel dimension
        img_chw = np.concatenate((img_sen1, img_sen2), axis=-1)
        
        # Ensure the image is in the correct shape (C, H, W)
        if img_chw.shape[-1] == 18:
            img_chw = np.transpose(img_chw, (2, 0, 1))  # Convert (H, W, C) to (C, H, W)

        img_chw = self.transform(img_chw)

        channels = self.channels.numpy()

        if self.scale != 1:
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [img * self.scale for img in img_chw]
            else:
                # single view for linear probing
                img_chw *= self.scale

        # mask out channels
        if type(img_chw) is list:
            if self.channel_mask:
                unselected = [c for c in range(len(img_chw[0])) if c not in channels]
                for i in range(len(img_chw)):
                    img_chw[i][unselected] = 0
            else:
                img_chw = [img[channels] for img in img_chw]
        else:
            if self.channel_mask:
                unselected = [c for c in range(len(img_chw)) if c not in channels]
                img_chw[unselected] = 0
            else:
                img_chw = img_chw[channels]

        # label = row.label.astype(int)
        # Access the label directly from self.df using the index
        label = self.df['label'][index].astype(int)
        if sum(label) > 1:
            raise ValueError("More than one positive")

        for i, y in enumerate(label):
            if y == 1:
                label = i
                break

        return (
            torch.tensor(img_chw.copy()).float(),
            {"channels": channels, "label": label},
        )

    def __len__(self) -> int:
        return len(self.df)
