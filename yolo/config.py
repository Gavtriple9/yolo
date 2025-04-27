import toml
from yolo.colors import Colors

# ---------------------- DEFAULT PARAMETERS ---------------------------
"""
Note: These values are only defined so a config file is not required
      to run the program. If a config file is provided, the values
      in the config file will override these defaults.
"""
# ---------------------------------------------------------------------


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """
    A singleton for managing configuration parameters.

    :param config_path: The path to the configuration file.
    :type config_path: str

    :example:
        config = Config("config.toml")
        print(config["key"])
    """

    def __init__(self, config_path: str = None):
        self._config = {}
        if config_path:
            self._config = toml.load(config_path)

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def __getitem__(self, key: str):
        return self._config[key]

    def __setitem__(self, key: str, value):
        self._config[key] = value

    def __delitem__(self, key: str):
        del self._config[key]

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __str__(self):
        return str(self._config)

    def __repr__(self):
        return repr(self._config)
