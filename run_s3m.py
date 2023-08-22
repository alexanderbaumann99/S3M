import random
import string
import logging
import sys
import os
from pathlib import Path
import yaml
from pipeline.pipeline_utils import seed_everything, get_train_val_sets
from pipeline.s3m_pipeline import s3m_pipeline


with open(sys.argv[1], "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

RUN_KEY = "".join(random.sample(string.ascii_lowercase, 8))
config["run_key"] = RUN_KEY
seed_everything(config["SEED"])

for val_fold in range(len(config["data"]["data_folds"])):
    # Get train/val split
    train_set, val_set = get_train_val_sets(config, val_fold)
    train_paths = [
        "{:s}/{:03d}.ply".format(config["data"]["data_path"], i) for i in train_set
    ]
    val_paths = [
        "{:s}/{:03d}.ply".format(config["data"]["data_path"], i) for i in val_set
    ]
    # test generates the outputs.. use both train and val
    inference_paths = train_paths + val_paths

    naming = str(config["data"]["data_path"]).split("/")[-1]

    # Define output path
    output_path = (
        Path(config["data"]["output_path"])
        / f"experiment_{RUN_KEY}_{naming}"
        / f'fold_{val_fold}'
    )
    # Create output path
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    # Save config file in output path for reproducibility
    yaml_file = output_path / "config.yml"
    with yaml_file.open("w") as file:
        yaml.dump(config, file)

    if not(os.path.exists(os.path.join(output_path.parent, "s3m_run.log"))):
        logging.basicConfig(
            filename=os.path.join(output_path.parent, "s3m_run.log"), level=logging.INFO
        )
    logging.info(f'Validation fold: {val_fold}')
    # run s3m pipeline
    mat_savedir, ref_shape = s3m_pipeline(
        config["ssm"], train_paths, val_paths, inference_paths, save_path=output_path
    )
