import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import Logger

from yolo.config import Config

DEFAULT_IN_CHANNELS = 3


def get_layers(config: Config):
    layers: list[dict] = config["layers"]
    prev_layer = None
    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        if layer_type == "conv":
            in_channels = (
                DEFAULT_IN_CHANNELS if prev_layer == None else prev_layer.out_channels
            )
            out_channels = layer["filters"]
            kernel_size = layer["kernel_size"]
            stride = layer["stride"]
            activation = layer.get("activation", "relu")
            if activation == "relu":
                activation_fn = nn.ReLU()
            elif activation == "leaky_relu":
                activation_fn = nn.LeakyReLU()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            prev_layer = conv_layer
            yield conv_layer
            yield activation_fn

        elif layer_type == "maxpool":
            kernel_size = layer["kernel_size"]
            stride = layer["stride"]
            pool_layer = nn.MaxPool2d(kernel_size, stride)
            yield pool_layer

        elif layer_type == "fc":
            in_features = (
                DEFAULT_IN_CHANNELS if prev_layer == None else prev_layer.out_channels
            )
            out_features = layer["filters"]
            activation = layer.get("activation", "relu")
            if activation == "relu":
                activation_fn = nn.ReLU()
            elif activation == "leaky_relu":
                activation_fn = nn.LeakyReLU()
            elif activation == "linear":
                activation_fn = nn.Identity()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            fc_layer = nn.Linear(in_features, out_features)
            yield fc_layer
            yield activation_fn

        elif layer_type == "reshape":
            shape = layer["shape"]
            reshape_layer = lambda x: x.view(shape)
            yield reshape_layer

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


class Yolo(nn.Module):
    def __init__(self, config, logger: Logger):
        super().__init__()
        layers = []
        for layer in get_layers(config):
            layers.append(layer)

        self.layers = nn.Sequential(*layers)
        self.logger = logger

    def forward(self, x):
        x = self.layers(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)
        self.logger.info(f"Loaded weights from {weights_path}")
        return self

    def save_weights(self, weights_path):
        torch.save(self.state_dict(), weights_path)
        self.logger.info(f"Saved weights to {weights_path}")
        return self
