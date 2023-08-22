import shutil
import os
from typing import List, Tuple
import logging
from pathlib import Path
from model.S3M_preprocessing import preprocess_train_set_main, preprocess_test_set_main
from model.S3M_scripts import train_S3M, get_reference_shape, inference_S3M
from model.utils.correspond_vertices import save_corresponding_vertices


def s3m_pipeline(
    cfg: dict,
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    save_path: str,
) -> Tuple[str]:
    """
    Pipeline of our proposed mode. Includes preprocessing, training and
    inference.
    Args:
        cfg:                        Configuration data
        train_paths:                List of paths to train mesh files
        val_paths:                  List of paths to val mesh files
        test_paths:                 List of paths to test mesh files
        save_path:                  Output directory

    Returns:
        mat_savedir:                Directory containing mat files with correspondences
        optimal_reference_shape:    Index of optimal reference shape
    """
    # Logging
    # logging.basicConfig(
    #     filename=os.path.join(save_path, "s3m_run.log"), level=logging.INFO
    # )

    logging.info("Preparing training data...")
    # preprocess train data
    experiment_dir = Path(save_path)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_dir_train = experiment_dir / "train"
    save_dir_train.mkdir()
    preprocess_train_set_main(
        file_paths = train_paths,
        save_dir = save_dir_train,
        n_eigen = cfg["n_lbo_eigenfunctions"],
        n_jobs=cfg["n_jobs"],
        n_points=cfg["n_points"],
        scaling=cfg['scaling']
    )

    # preprocess val data
    save_dir_val = experiment_dir / "val"
    save_dir_val.mkdir()
    preprocess_test_set_main(
        file_paths = val_paths,
        save_dir = save_dir_val,
        n_eigen = cfg["n_lbo_eigenfunctions"],
        n_jobs=cfg["n_jobs"],
        n_points=cfg["n_points"],
        scaling=cfg['scaling'], 
        distances = False
    )

    # Run training
    logging.info('Starting training script')
    cfg["preprocessed_data_root"] = str(save_dir_train)
    cfg["preprocessed_data_root_val"] = str(save_dir_val)
    cfg["TrainingDetails"]["save_dir"] = save_path
    model = train_S3M(cfg)

    logging.info('Preparing data for reference shape extraction')
    # preprocess data for reference shape extraction
    save_dir_train_wo_augmentations = experiment_dir / "train_wo_augmentations"
    preprocess_test_set_main(
        file_paths = train_paths,
        save_dir = save_dir_train_wo_augmentations,
        n_eigen = cfg["n_lbo_eigenfunctions"],
        n_jobs=cfg["n_jobs"],
        n_points=cfg["n_points"],
        scaling=cfg['scaling'],
        distances = False
    )
    cfg["preprocessed_data_root"] = str(save_dir_train_wo_augmentations)
    optimal_reference_shape = get_reference_shape(cfg, model=model)
    logging.info(f"Reference shape: {optimal_reference_shape}")

    logging.info('Preparing inference data')
    # preprocess data for inference
    # inference shapes should generally include all shapes
    save_dir_test = experiment_dir / "test"
    preprocess_test_set_main(
        file_paths = test_paths,
        save_dir = save_dir_test,
        n_eigen = cfg["n_lbo_eigenfunctions"],
        n_jobs=cfg["n_jobs"],
        n_points=cfg["n_points_inference"],
        scaling=cfg['scaling'],
        distances = True
    )

    logging.info('Starting inference script')
    # Run inference
    cfg["preprocessed_data_root"] = str(save_dir_test)
    cfg["TestingDetails"]["save_dir"] = save_path
    cfg["TestingDetails"]["reference_shape"] = optimal_reference_shape
    mat_savedir = inference_S3M(cfg, model=model)

    # Delete preprocessed mat files
    shutil.rmtree(save_dir_train)
    shutil.rmtree(save_dir_val)
    shutil.rmtree(save_dir_train_wo_augmentations)
    shutil.rmtree(save_dir_test)

    # Save corresponded vertices
    save_corresponding_vertices(mat_savedir, int(optimal_reference_shape))
    logging.info(
        f"Saved correspondence files to {mat_savedir} \
                for reference shape {optimal_reference_shape}"
    )

    return mat_savedir, optimal_reference_shape
