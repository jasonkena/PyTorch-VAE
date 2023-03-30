from experiment import VAEXperiment
from dataset import VAEDataset
import yaml
import torch
from torchvision import transforms
from models import *

import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

from sklearn.manifold import TSNE

import umap
import umap.plot
import matplotlib
import matplotlib.pyplot as plt

import os
import glob
import h5py

from scipy.spatial import KDTree
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load():
    config = yaml.safe_load(open("./configs/vesicle.yaml"))
    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    ckpt = torch.load("./logs/VanillaVAE/version_0/checkpoints/last.ckpt")
    experiment = VAEXperiment(model, config["exp_params"])
    experiment.load_state_dict(ckpt["state_dict"])

    data = VAEDataset(
        **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
    )
    data.setup(return_indices=True)

    return experiment, data


def get_embeddings(experiment, data):
    train_loader = data.train_dataloader()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            img, labels, idx = batch
            mu, sigma = experiment.model.encode(img)
            embeddings.append((idx, mu, sigma))
    idx, mu, sigma = zip(*embeddings)
    idx, mu, sigma = torch.cat(idx), torch.cat(mu), torch.cat(sigma)
    assert len(torch.unique(idx)) == len(train_loader.dataset)

    argsort = torch.argsort(idx)
    idx, mu, sigma = idx[argsort], mu[argsort], sigma[argsort]
    torch.save((idx, mu, sigma), "embeddings.pt")


def tsne():
    idx, mu, sigma = torch.load("embeddings.pt")
    x = torch.cat([mu, sigma], dim=1).detach().numpy()
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(x)
    np.save("tsne.npy", tsne_results)


def get_umap():
    idx, mu, sigma = torch.load("embeddings.pt")
    x = mu.detach().numpy()
    # x = torch.cat([mu, sigma], dim = 1).detach().numpy()
    embedding = umap.UMAP(verbose=True).fit(x)
    np.save("umap.npy", embedding)


def recons(experiment, img):
    recons = experiment.model.generate(img.unsqueeze(0), labels=0)
    iio.imwrite(
        "recons.png", (recons * 256).detach().numpy().reshape(16, 16).astype(np.uint8)
    )

    return recons


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.nanmin(points[:, 0])
    max_x = np.nanmax(points[:, 0])
    min_y = np.nanmin(points[:, 1])
    max_y = np.nanmax(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def plot():
    matplotlib.use("QtAgg")
    base_path = "/mmfs1/data/bccv/dataset/xiaomeng/ves"
    h5s = sorted(glob.glob(os.path.join(base_path, "*_patch.h5")))
    labels = []
    images = []
    for h5 in h5s:
        with h5py.File(h5, "r") as f:
            if "pos" in h5:
                labels.append(np.ones(f["main"].shape[0], dtype=np.int32))
            else:
                assert "neg" in h5
                labels.append(np.zeros(f["main"].shape[0], dtype=np.int32))
            images.append(f["main"][:])
    labels = np.concatenate(labels)
    images = np.concatenate(images)

    umap_results = np.load("umap.npy", allow_pickle=True)
    mapper = umap_results.item()

    embeddings = mapper.embedding_
    # plt.hist2d(embeddings[:, 0], embeddings[:, 1], bins=100)
    # plt.show()
    extent = [
        np.min(embeddings[:, 0]),
        np.max(embeddings[:, 0]),
        np.min(embeddings[:, 1]),
        np.max(embeddings[:, 1]),
    ]
    print(f"extent: {extent}")
    # extent = _get_extent(embeddings)
    fig_size = (800, 800)

    kd = KDTree(embeddings)
    ax = umap.plot.points(mapper, labels=labels, color_key_cmap="cool")
    fig = ax.get_figure()
    im = OffsetImage(images[0], zoom=5, cmap="gray")

    xybox = (50.0, 50.0)
    ab = AnnotationBbox(
        im,
        (0, 0),
        xybox=xybox,
        xycoords="data",
        boxcoords="offset points",
        pad=0.3,
        arrowprops=dict(arrowstyle="->"),
    )
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def onclick(event):
        if event.inaxes == ax:
            x = event.xdata / fig_size[0] * (extent[1] - extent[0]) + extent[0]
            y = (fig_size[1] - event.ydata) / fig_size[1] * (
                extent[3] - extent[2]
            ) + extent[2]
            print(f"real_x: {x}, real_y: {y}")

            dist, idx = kd.query([x, y])
            print(dist, idx)
            print(embeddings[idx])
            im.set_data(images[idx])

            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
            hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)

            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (event.xdata, event.ydata)

            ab.set_visible(True)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    # recons(experiment, test_img)


if __name__ == "__main__":
    #  setenv LD_LIBRARY_PATH LD_LIBRARY_PATH\:/data/adhinart/.conda/envs/vesicle/lib/
    # test_img = iio.imread("../white.png").astype(np.float32) / 256
    # test_img = torch.from_numpy(test_img).reshape(1, test_img.shape[0], test_img.shape[1])
    #
    # test_img = transforms.Resize((16,16))(test_img)
    #
    # experiment, data = load()
    # get_embeddings(experiment, data)

    # tsne()
    # get_umap()
    plot()
