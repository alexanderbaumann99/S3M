import os
import logging
from pathlib import Path
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import torch
from torch.multiprocessing import set_start_method
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from .S3M_model import S3MNet
from .S3M_dataset import S3MNetDataset
from .S3M_loss import S3MNetLoss
from .utils.p2p_correspondence import FM_to_p2p
from .utils.visualization import visu
from .utils.pmf import pmf

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def train_S3M(cfg):
    """
    Training procedure of S3MNet
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    save_dir = os.path.join(cfg["TrainingDetails"]["save_dir"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create dataset
    logging.info("Creating dataset")
    dataset = S3MNetDataset(
        root = cfg["preprocessed_data_root"],
        n_eigen = cfg["n_lbo_eigenfunctions"],
        mode = "train",
        n_points=cfg["TrainingDetails"]["n_sample_points"],
        device=device,
    )
    logging.info("Length of train set: %d" % (len(dataset)))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["TrainingDetails"]["batch_size"],
        shuffle=True,
    )
    logging.info("Length of train dataloader: %d" % (len(dataloader)))
    n_points = cfg["TrainingDetails"]["n_sample_points"]
    dataset_val = S3MNetDataset(
        root = cfg["preprocessed_data_root_val"],
        n_eigen = cfg["n_lbo_eigenfunctions"],
        mode = "train",
        n_points=n_points,
        device=device,
    )
    logging.info("Length of validation dataset: %d" % (len(dataset)))
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg["TrainingDetails"]["batch_size"],
        shuffle=True,
    )
    logging.info("Length of validation dataloader: %d" % (len(dataloader_val)))

    # create model
    logging.info("Creating model")
    s3mnet = S3MNet(cfg["feat_size"])
    if cfg["weights"]:
        s3mnet.load_state_dict(torch.load(cfg["weights"]))
    s3mnet = s3mnet.to(device)
    optimizer = torch.optim.AdamW(s3mnet.parameters(), lr=cfg["TrainingDetails"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=20, threshold=1e-2, verbose=1
    )
    criterion = S3MNetLoss()

    writer = SummaryWriter(save_dir)
    logging.info(
        f"Using {sum(p.numel() for p in s3mnet.parameters())} trainable parameters"
    )

    # Training loop
    number_epochs = cfg["TrainingDetails"]["n_epochs"]
    logging.info(f"Starting training for {number_epochs} epochs..")
    iterations_train = 0
    iterations_val = 0
    for epoch in tqdm(range(1, number_epochs + 1)):
        s3mnet.train()
        epoch_loss_train = 0
        for iteration, data in enumerate(dataloader):
            data = [x.to(device) for x in data]
            (
                graph_x,
                evals_x,
                evecs_x,
                evecs_trans_x,
                graph_y,
                evals_y,
                evecs_y,
                evecs_trans_y,
            ) = data
            # do iteration
            c_x, c_y, feat_x, feat_y = s3mnet(
                graph_x, graph_y, evecs_trans_x, evecs_trans_y
            )
            loss = criterion(
                c_x,
                c_y,
                feat_x,
                feat_y,
                evecs_x,
                evecs_y,
                evecs_trans_x,
                evecs_trans_y,
                evals_x,
                evals_y,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # log
            iterations_train += iteration
            epoch_loss_train += loss.item()

            writer.add_scalar("Training Loss per Iter", loss.item(), iterations_train)

        epoch_loss_train /= len(dataloader)
        writer.add_scalar("Training loss per Epoch", epoch_loss_train, epoch)

        s3mnet.eval()
        epoch_loss_val = 0
        for iteration, data in enumerate(dataloader_val):
            data = [x.to(device) for x in data]
            data = [x.to(device) for x in data]
            with torch.no_grad():
                (
                    graph_x,
                    evals_x,
                    evecs_x,
                    evecs_trans_x,
                    graph_y,
                    evals_y,
                    evecs_y,
                    evecs_trans_y,
                ) = data
                # do iteration
                c_x, c_y, feat_x, feat_y = s3mnet(
                    graph_x, graph_y, evecs_trans_x, evecs_trans_y
                )

                loss = criterion(
                    c_x,
                    c_y,
                    feat_x,
                    feat_y,
                    evecs_x,
                    evecs_y,
                    evecs_trans_x,
                    evecs_trans_y,
                    evals_x,
                    evals_y,
                )

            iterations_val += iteration
            epoch_loss_val += loss.item()

            writer.add_scalar("Validation Loss per Iter", loss.item(), iterations_val)

        # save model
        torch.save(s3mnet.state_dict(), os.path.join(save_dir, "model_last.pth"))

        scheduler.step(epoch_loss_val)
        epoch_loss_val /= len(dataloader_val)
        writer.add_scalar("Validation loss per Epoch", epoch_loss_val, epoch)

    return s3mnet


def get_reference_shape(cfg: dict, model: torch.nn.Module = None):
    """
    After training, the function examines the optimal shape as
    reference. It is deduced by the lowest loss to all other
    shapes.
    Args:
        cfg:        Config file
        model:      Trained model
    Returns:
        _ref_id:    ID of the optimal reference shape
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # create dataset
    dataset = S3MNetDataset(
        root = cfg["preprocessed_data_root"],
        n_eigen = cfg["n_lbo_eigenfunctions"],
        mode = "train",
        n_points=cfg["TrainingDetails"]["n_sample_points"],
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if model is not None:
        s3mnet = model
    else:
        s3mnet = S3MNet(cfg["feat_size"])
        if cfg["weights"]:
            s3mnet.load_state_dict(torch.load(cfg["weights"]))
    s3mnet = s3mnet.to(device)

    criterion = S3MNetLoss().to(device)

    s3mnet.eval()
    losses = []
    for _ in range(len(dataset.samples)):
        losses.append([])

    for idx, data in enumerate(dataloader):
        data = [x.to(device) for x in data]

        with torch.no_grad():
            (
                graph_x,
                evals_x,
                evecs_x,
                evecs_trans_x,
                graph_y,
                evals_y,
                evecs_y,
                evecs_trans_y,
            ) = data
            reference_number, _ = dataset.combinations[idx]
            # do iteration
            c_x, c_y, feat_x, feat_y = s3mnet(
                graph_x, graph_y, evecs_trans_x, evecs_trans_y
            )
            loss = criterion(
                c_x,
                c_y,
                feat_x,
                feat_y,
                evecs_x,
                evecs_y,
                evecs_trans_x,
                evecs_trans_y,
                evals_x,
                evals_y,
            )

        losses[reference_number].append(loss.item())

    losses = np.mean(np.array(losses), axis=1)
    optimal_reference_shape = np.argmin(losses)

    def _get_id(path):
        return Path(path).stem.split("_")[-1]

    _ref_id = _get_id(dataset.samples[optimal_reference_shape])

    return _ref_id


def inference_S3M(cfg: dict, model: torch.nn.Module = None):
    """
    Calculates the correspondences between the reference shape
    and all other shapes.
    Args:
        cfg:        Config file
        model:      Trained model
    Returns:
        save_dir:   Directory where mat files are stored
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    save_dir = os.path.join(cfg["TestingDetails"]["save_dir"], "mat")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create dataset
    dataset = S3MNetDataset(
        root = cfg["preprocessed_data_root"],
        n_eigen = cfg["n_lbo_eigenfunctions"],
        mode="test",
        n_points=None,
        ref_shape=cfg["TestingDetails"]["reference_shape"],
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # create model
    if model:
        s3mnet = model
    else:
        s3mnet = S3MNet(cfg["feat_size"])
        s3mnet.load_state_dict(torch.load(cfg["weights"]))
        s3mnet = s3mnet.to(device)
    criterion = S3MNetLoss().to(device)

    # Inference loop
    logging.info("Start testing")
    for iteration, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ref_num = dataset.combinations[iteration][0]
        tar_num = dataset.combinations[iteration][1]

        data = [x.to(device) for x in data]

        (
            graph_x,
            evals_x,
            evecs_x,
            evecs_trans_x,
            dist_x,
            graph_y,
            evals_y,
            evecs_y,
            evecs_trans_y,
            dist_y,
        ) = data

        with torch.no_grad():
            c_x, c_y, feat_x, feat_y = s3mnet(
                graph_x, graph_y, evecs_trans_x, evecs_trans_y
            )
            loss = criterion(
                c_x,
                c_y,
                feat_x,
                feat_y,
                evecs_x,
                evecs_y,
                evecs_trans_x,
                evecs_trans_y,
                evals_x,
                evals_y,
            )

        c_x = c_x.cpu().numpy().squeeze()
        c_y = c_y.cpu().numpy().squeeze()
        evecs_x = evecs_x.cpu().numpy().squeeze()
        evecs_y = evecs_y.cpu().numpy().squeeze()
        verts_x = graph_x.pos.cpu().numpy().squeeze()
        verts_y = graph_y.pos.cpu().numpy().squeeze()
        loss = loss.item()

        y2x = FM_to_p2p(c_x, evecs_x, evecs_y)
        corres_idx = np.empty((verts_x.shape[0], 2), dtype="long")
        corres_idx[:, 1] = np.arange(verts_y.shape[0])
        corres_idx[:, 0] = np.array(y2x)
        dist_x, dist_y = dist_x.squeeze(), dist_y.squeeze()
        y2x_pmf = pmf(
            corres_idx = corres_idx,
            dist_x = dist_x,
            dist_y = dist_y,
            var = cfg['TestingDetails']['variance_pmf']
        )

        cmap1 = visu(verts_x)
        cmap2 = cmap1[y2x]
        cmap2_pmf = cmap1[y2x_pmf]
        dist_x, dist_y = dist_x.cpu().numpy().squeeze(), dist_y.cpu().numpy().squeeze()
        to_save = {
            "verts_x": verts_x,
            "verts_y": verts_y,
            "dist_x": dist_x,
            "dist_y": dist_y,
            "cmap1": cmap1,
            "cmap2": cmap2,
            "cmap2pmf": cmap2_pmf,
            "loss": loss,
            "y2x": y2x,
            "y2x_pmf": y2x_pmf,
        }

        def _get_id(path):
            return Path(path).stem.split("_")[-1]

        _ref_id = _get_id(dataset.samples[ref_num])
        _sample_id = _get_id(dataset.samples[tar_num])
        logging.info(
            "Writing file for shape " + str(_ref_id) + "_" + str(_sample_id) + "..."
        )

        sio.savemat(os.path.join(save_dir, f"out_{_ref_id}_{_sample_id}.mat"), to_save)

    return save_dir
