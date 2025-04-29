import os
from logging import Logger
import torch.nn.functional as F

from yolo import ROOT_DIR
from yolo.model import Yolo
from yolo.config import Config
from yolo.guru import Guru
from yolo.data import create_voc_dataloader


def train_model(args, logger: Logger):
    config_path = os.path.join(ROOT_DIR, "config", "default.toml")
    logger.info(f"Loading config from {config_path}")
    config = Config(config_path)
    model = Yolo(config["model"], logger)
    train_dataloader = create_voc_dataloader(
        config["train"]["batch_size"], train=True, num_workers=4
    )
    logger.info(f"Creating dataloader with {len(train_dataloader)} batches")
    guru = Guru(logger, model, train_dataloader)

    for epoch in range(config["train"]["epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{config['train']['epochs']}")
        avg_loss = guru.train_one_epoch()
        logger.info(f"Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}")

    logger.info("Training completed")


def test_model(args, logger: Logger):
    config_path = os.path.join(ROOT_DIR, "config", "default.toml")
    logger.info(f"Loading config from {config_path}")
    config = Config(config_path)
    model = Yolo(config["model"], logger)

    eval_dataloader = create_voc_dataloader(64, train=False, num_workers=4)
    for batch_index, (images, targets) in enumerate(eval_dataloader):
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        logger.info(f"Batch {batch_index}, Loss: {loss.item()}")

    logger.info("Evaluation completed")
