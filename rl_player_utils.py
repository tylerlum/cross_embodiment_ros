from typing import Optional
import yaml
from omegaconf import OmegaConf, DictConfig
from reformat import omegaconf_to_dict, print_dict


def read_cfg(config_path: str, device: Optional[str] = None) -> dict:
    # Read in isaacgymenvs config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Need to convert back to dict
    cfg = omegaconf_to_dict(omegaconf_cfg)

    # Modify device manually
    if device is not None:
        cfg["train"]["params"]["config"]["device"] = device
        cfg["train"]["params"]["config"]["device_name"] = device

    print("-" * 80)
    print("CONFIGURATION")
    print_dict(cfg)
    print("-" * 80 + "\n")

    return cfg


def read_cfg_omegaconf(config_path: str, device: Optional[str] = None) -> DictConfig:
    # Read in isaacgymenvs config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Modify device manually
    if device is not None:
        omegaconf_cfg.train.params.config.device = device
        omegaconf_cfg.train.params.config.device_name = device

    print("-" * 80)
    print("CONFIGURATION")
    print_dict(omegaconf_cfg)
    print("-" * 80 + "\n")

    return omegaconf_cfg
