import toml


def open_toml(path):
    """Returns a dict with the contents of the toml file
    :returns: a dict from a toml file
    :rtype: dict
    """
    
    with open(path, "r") as f:
        data = toml.load(f)
    return data
