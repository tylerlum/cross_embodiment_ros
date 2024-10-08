from typing import Tuple


def _get_entity_project_runid(wandb_url: str) -> Tuple[str, str, str]:
    # Input: https://wandb.ai/tylerlum/MOCCA-development-3/runs/372voqf0/files/stats/tyler_laikago_2022-02-16_11_01/iter17.pt
    # Output: entity="tylerlum", project="MOCCA-development-3", run_id="372voqf0"

    # Input: https://wandb.ai/tylerlum/language_modulated_representation_learning_v2/runs/e5es9mkr
    # Output: entity="tylerlum", project="language_modulated_representation_learning_v2", run_id="e5es9mkr"

    # New case with groups
    # Input: https://wandb.ai/tylerlum/bidex/groups/2024-02-22_Comparisons/files/runs/R_Reach_Absolute_Fixed_2024-02-22_22-38-33/nn/R_Reach_Absolute_Fixed.pth?runName=R_Reach_Absolute_Fixed_2024-02-22_22-38-33
    # Output: entity="tylerlum", project="bidex", run_id="R_Reach_Absolute_Fixed_2024-02-22_22-38-33"
    url_split = wandb_url.split("/")
    if "wandb.ai" not in url_split:
        raise ValueError(f"Invalid wandb url: {wandb_url}")

    start_idx = url_split.index("wandb.ai") + 1
    entity, project, runs_ro_groups, run_id_or_group_name = url_split[
        start_idx : start_idx + 4
    ]
    if runs_ro_groups == "runs":
        run_id = run_id_or_group_name
        print(f"run_id = {run_id}")
    elif runs_ro_groups == "groups":
        group_name = run_id_or_group_name
        print(f"group_name = {group_name}")

        if "?runName=" not in wandb_url:
            raise ValueError(f"Invalid wandb url: {wandb_url}")
        run_id = wandb_url.split("?runName=")[-1]
        print(f"run_id = {run_id}")
    else:
        raise ValueError(f"Invalid wandb url: {wandb_url}")

    print(f"entity={entity}, project={project}, run_id={run_id}")
    return entity, project, run_id


def _get_filepath(wandb_file_url: str) -> str:
    url_split = wandb_file_url.split("/")
    files_idx = url_split.index("files")
    filepath = "/".join(url_split[files_idx + 1 :])

    # Find .pth or .pt file extension
    if ".pth" in filepath:
        filepath = filepath.split(".pth")[0] + ".pth"
    elif ".pt" in filepath:
        filepath = filepath.split(".pt")[0] + ".pt"
    elif ".yaml" in filepath:
        filepath = filepath.split(".yaml")[0] + ".yaml"
    elif ".yml" in filepath:
        filepath = filepath.split(".yml")[0] + ".yml"
    else:
        raise ValueError(f"Invalid wandb_file_url {wandb_file_url}")
    return filepath


def load_model(model, filepath: str, strict: bool = True) -> None:
    import torch
    import os

    # Input: filepath to a .pt file
    # Output: model loaded with state_dict from filepath
    if not os.path.exists(filepath):
        print(f"Filepath {filepath} does not exist")
        filepath = os.path.join(os.getcwd(), filepath)
        print(f"Trying with {filepath}")
        if not os.path.exists(filepath):
            raise ValueError(f"Filepath {filepath} does not exist")

    state_dict = torch.load(filepath)

    # Check if model state dict matches weights
    loaded_state_dict_keys = set(state_dict.keys())
    model_state_dict_keys = set(model.state_dict().keys())
    if strict:
        assert loaded_state_dict_keys == model_state_dict_keys
    elif loaded_state_dict_keys != model_state_dict_keys:
        print("WARNING: loaded_state_dict_keys != model_state_dict_keys")
        print(
            f"Only in loaded_state_dict_keys: {loaded_state_dict_keys.difference(model_state_dict_keys)}"
        )
        print(
            f"Only in model_state_dict_keys: {model_state_dict_keys.difference(loaded_state_dict_keys)}"
        )

    model.load_state_dict(state_dict, strict=strict)
    return


def restore_model_file_from_wandb(
    wandb_file_url: str, strict: bool = True, model=None
) -> str:
    import wandb

    # Get saved file from wandb
    # Input: https://wandb.ai/tylerlum/MOCCA-development-3/runs/372voqf0/files/stats/tyler_laikago_2022-02-16_11_01/iter17.pt
    # Output: filepath to restored model (restore stats/tyler_laikago_2022-02-16_11_01/iter17.pt)
    entity, project, run_id = _get_entity_project_runid(wandb_file_url)
    run_path = "/".join([entity, project, run_id])
    print(f"Restoring model from {run_path}")

    filepath = _get_filepath(wandb_file_url)
    print(f"Model filepath: {filepath}")

    wandb_file = wandb.restore(filepath, run_path=run_path, replace=True)
    if wandb_file is None:
        print("WARNING: wandb_file is None")
    if model is not None:
        load_model(model=model, filepath=wandb_file.name, strict=strict)
    return filepath
