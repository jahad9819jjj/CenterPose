import hydra
import logging
import pathlib
import torch 
import json
from dataset import ObjectronDataset 
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    logger = logging.getLogger(__name__)
    datapath = pathlib.Path(cfg.dataset_root)
    dataset = ObjectronDataset(cfg, datapath)
    dataloader = torch.utils.data.DataLoader(dataset, 
                            batch_size=cfg.batch_size,
                            shuffle=True, 
                            num_workers=cfg.num_workers,
                            pin_memory=True
                            )
    
    for epoch in range(cfg.max_epoch):
        logger.info(f"Epoch: {epoch}")
        with tqdm(dataloader) as t:
            for batch in t:
                logger.info(f"============== Batch Key: {batch.keys()}")
                logger.info(f"============== Batch shape: {batch['points'].shape}")
    
    
    logger.info(f"Category: {cfg.category.name}")

if __name__ == "__main__":
    main()