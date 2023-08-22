import os
import sys
from pathlib import Path
from copy import deepcopy
import yaml
import numpy as np
import torch
from chamferdist import ChamferDistance

from evaluation.SSM import SSM
from evaluation.evaluation_utils import (
    get_correspondended_vertices,
    get_target_point_cloud,
    get_test_point_cloud,
)


def run_evaluation(save_dir: str):
    """
    Cross-validates the results with respect to generalization and specificty.

    Args:
        save_dir:   Path to an output directory of any fold.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_dir = os.path.join( str(save_dir),"fold_{}")
    save_dir = Path(save_dir)

    with open(os.path.join(fold_dir.format(0), "config.yml"), "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    path_meshes = config["data"]["data_path"]
    path_correspondend_verts_per_fold = os.path.join(
        fold_dir,
        "corres_verts.npy",
    )
    folds = config["data"]["data_folds"]
    # Distance metric
    chamfer_distance = ChamferDistance()

    generalizations = []
    specificities = []
    # number of samples for specifity
    n_samples = 1000

    # Iteration over folds
    for val_fold in range(len(folds)):
        generalizations_per_fold = []
        specificities_per_fold = []

        correspondence_path = path_correspondend_verts_per_fold.format(val_fold)
        copied_folds = deepcopy(folds)
        val_set = np.array(copied_folds.pop(val_fold))
        train_set = np.concatenate(copied_folds)

        # Load corresponded vertices
        corresponded_train_verts, n_particles = get_correspondended_vertices(
            train_set, correspondence_path
        )
        # Define SSM class
        ssm = SSM(np.transpose(corresponded_train_verts, (1, 0)))

        # generalization error
        for i in val_set:
            # Get SSM's reconstruction of test shape
            test_shape = get_test_point_cloud(correspondence_path, i)
            reconstruction = ssm.get_reconstruction(test_shape)
            reconstruction = reconstruction.reshape(1, -1, 3)
            reconstruction = torch.Tensor(reconstruction).to(device)

            target = get_target_point_cloud(path_meshes, i)
            target_cloud = torch.Tensor(target).view(1, -1, 3).to(device)
            # Compute generalization error
            generalizations_per_fold.append(
                np.sqrt(chamfer_distance(reconstruction, target_cloud).cpu().numpy())
                / n_particles
            )
        generalizations.append(np.mean(generalizations_per_fold))

        # specificity error
        # Generate random samples from the SSM
        samples = ssm.generate_random_samples(n_samples = n_samples)

        training_point_clouds = []
        for i in train_set:
            target = get_target_point_cloud(path_meshes, i)
            training_point_clouds.append(torch.Tensor(target).unsqueeze(0).to(device))

        # For each sample...
        for sample in samples:
            sample = torch.Tensor(sample.reshape(-1, 3)).unsqueeze(0).to(device)
            distances_per_sample = []
            # Find closest instance from training set
            for target_cloud in training_point_clouds:
                distances_per_sample.append(
                    np.sqrt(chamfer_distance(sample, target_cloud).cpu().numpy())
                    / n_particles
                )
            specificities_per_fold.append(min(distances_per_sample))

        specificities.append(np.mean(specificities_per_fold))

    print("Generalization: ", np.mean(generalizations), np.std(generalizations))
    print("Specificity: ", np.mean(specificities), np.std(specificities))


if __name__ == "__main__":

    if len(sys.argv) == 2:
        SAVE_DIR = sys.argv[1]  # directory to saved data
        run_evaluation(SAVE_DIR)
    else:
        raise ValueError("Please indicate directory to evaluate")
