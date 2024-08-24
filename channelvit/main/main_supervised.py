import logging

import boto3
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

import channelvit.data as data
from channelvit.meta_arch import Supervised
from torch.utils.data import DataLoader, Dataset, default_collate


class LazyDataset(Dataset):
    def __init__(self, data_cfg, transform_cfg, subset_size=100):
        self.data_cfg = data_cfg
        self.transform_cfg = transform_cfg
        self.data = None  # Data will be loaded on demand
        self.subset_size = subset_size  # Limit the number of samples loaded

    def __len__(self):
        file_paths = self.data_cfg.args.get('path', [])
        return min(len(file_paths), self.subset_size)

    def __getitem__(self, idx):
        try:
            if self.data is None:
                self.data = getattr(data, self.data_cfg.name)(
                    is_train=True,
                    transform_cfg=self.transform_cfg,
                    **self.data_cfg.args,
                )
            return self.data[idx]
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise
    
    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)
    
def get_train_loader(cfg: DictConfig):
    # Define the training data loader.
    if len(cfg.train_data) == 1:

        print("There is only one training data")
        train_data_cfg = next(iter(cfg.train_data.values()))
        
        with open_dict(cfg):
            cfg.train_data = train_data_cfg
        
        print("Initializing train_data")
        try:
            # import psutil
            # # Check memory usage before loading the dataset
            # process = psutil.Process()
            # mem_before = process.memory_info().rss / (1024 ** 2)
            # print(f"Memory usage before loading dataset: {mem_before:.2f} MB")
            
            # # Print dataset size
            # print(f"Number of samples in dataset: {len(train_data)}")
            # sample = train_data[0]
            # sample_size = sum([s.nbytes for s in sample if isinstance(s, np.ndarray)])
            # print(f"Size of one sample: {sample_size / (1024 ** 2):.2f} MB")
            
            # train_data = getattr(data, train_data_cfg.name)(
            #     is_train=True,
            #     transform_cfg=cfg.train_transformations,
            #     **train_data_cfg.args,
            # )
            
            # Use LazyDataset to load data in smaller chunks
            train_data = LazyDataset(train_data_cfg, cfg.train_transformations)
            
            # Check memory usage after loading the dataset
            # mem_after = process.memory_info().rss / (1024 ** 2)
            # print(f"Memory usage after loading dataset: {mem_after:.2f} MB")
            # print(f"Memory increase: {mem_after - mem_before:.2f} MB")
            
            
        except Exception as e:
            print(f"Error initializing train_data: {e}")
            raise
        
        print("Train data initialized successfully")
        
        # Reduce batch size if necessary
        train_data_cfg.loader.batch_size = min(train_data_cfg.loader.batch_size, 1)
        
        print("Creating DataLoader")
        
        # train_loader = DataLoader(
        #     train_data, **train_data_cfg.loader,  collate_fn=train_data.collate_fn
        # )
        
        train_loader = DataLoader(
            train_data, 
            batch_size=1,  # Reduce batch size
            num_workers=0,  # Reduce number of workers
            pin_memory=False,  # Set pin_memory to False
            # persistent_workers=True,  # Use persistent workers
            collate_fn=LazyDataset.collate_fn
        )
        
        
        print("DataLoader created successfully")
        # We also need to pre-compute the number of batches for each epoch.
        # We will use this inforamtion for the learning rate schedule.
        with open_dict(cfg):
            # get number of batches per epoch (many optimizers use this information to schedule
            # the learning rate)
            cfg.train_data.loader.num_batches = (
                len(train_loader) // cfg.trainer.devices + 1
            )

        return train_loader

    else:
        print("There're more than one training data")
        train_loaders = {}
        len_loader = None
        batch_size = 0

        for name, train_data_cfg in cfg.train_data.items():
            print(f"Loading {train_data_cfg.name}")
            train_data = getattr(data, train_data_cfg.name)(
                is_train=True,
                transform_cfg=cfg.train_transformations,
                **train_data_cfg.args,
            )
            train_loader = DataLoader(
                train_data, **train_data_cfg.loader, collate_fn=train_data.collate_fn
            )
            train_loaders[name] = train_loader

            print(f"Dataset {name} has length {len(train_loader)}")

            if len_loader is None:
                len_loader = len(train_loader)
            else:
                len_loader = max(len_loader, len(train_loader))

            # batch_size += train_data_cfg.loader.batch_size
            batch_size = train_data_cfg.loader.batch_size

        with open_dict(cfg):
            cfg.train_data.loader = {}
            cfg.train_data.loader.num_batches = len_loader // cfg.trainer.devices + 1
            cfg.train_data.loader.batch_size = batch_size

        return train_loaders


@hydra.main(version_base=None, config_path="../config", config_name="main_supervised")
def train(cfg: DictConfig) -> None:
    if cfg.checkpoint != None:
        # load checkpont and perform inference
        return predict(cfg)

    # get the train data loader
    train_loader = get_train_loader(cfg)

    # Load the Supervised Meta arch
    model = Supervised(cfg)

    # Define the trainer.
    # We will use the configurations under cfg.trainer.
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=cfg.trainer.default_root_dir, save_top_k=-1),
    ]
    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true", callbacks=callbacks, **cfg.trainer
    )

    # Fit the model. if cfg.checkpoint is specified, we will start from the saved
    # checkpoint.
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=cfg.checkpoint)


def predict(cfg: DictConfig) -> None:
    print(f"loading model from ckpt {cfg.checkpoint}")
    model = Supervised.load_from_checkpoint(cfg.checkpoint)

    # load val data
    assert len(cfg.val_data_dict) == 1
    val_data_cfg = next(iter(cfg.val_data_dict.values()))
    val_data = getattr(data, val_data_cfg.name)(
        is_train=False, transform_cfg=cfg.val_transformations, **val_data_cfg.args
    )
    val_loader = DataLoader(
        val_data, **val_data_cfg.loader, collate_fn=val_data.collate_fn
    )

    trainer = pl.Trainer(strategy="ddp", **cfg.trainer)

    trainer.validate(model=model, dataloaders=val_loader)


if __name__ == "__main__":
    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
    torch.set_float32_matmul_precision("high")
    train()
