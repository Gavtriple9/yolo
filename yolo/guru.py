from logging import Logger
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from alive_progress import alive_bar

BATCH_LOG_PERIOD = 10


class Guru:
    """
    A Guru or instructor that trains models
    """

    def __init__(self, logger: Logger, model: nn.Module, train_loader: DataLoader):
        """
        Initializes the Guru class.
        """
        self.logger: Logger = logger
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.train_dset: DataLoader = train_loader
        self.model = model
        self.epoch_index = 0

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        self.logger.info("Training for one epoch")
        avg_loss = 0.0
        with alive_bar(len(self.train_dset)) as bar:
            for batch_index, (images, targets) in enumerate(self.train_dset):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if batch_index % BATCH_LOG_PERIOD == 0:
                    self.logger.info(f"Batch {batch_index}, Loss: {loss.item()}")
                avg_loss += loss.item()
                bar()
        avg_loss /= len(self.train_dset)
        self.epoch_index += 1
        return avg_loss
