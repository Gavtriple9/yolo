import os
from logging import Logger

from yolo import ROOT_DIR
from yolo.model import Yolo
from yolo.config import Config
from yolo.guru import Guru


def train_model(args, logger: Logger):
    config_path = os.path.join(ROOT_DIR, "config", "default.toml")
    logger.info(f"Loading config from {config_path}")
    config = Config(config_path)
    model = Yolo(config["model"], logger)


def test_model(args, logger: Logger):
    config_path = os.path.join(ROOT_DIR, "config", "default.toml")
    logger.info(f"Loading config from {config_path}")
    config = Config(config_path)
    model = Yolo(config["model"], logger)
