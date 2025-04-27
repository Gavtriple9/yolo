import torch
import torch.nn as nn
import torch.optim as optim


class Guru:
    """
    A Guru or instructor that trains models
    """

    def __init__(self, model: nn.Module, training_loader):
        """
        Initializes the Guru class.
        """
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.training_loader = training_loader
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss
