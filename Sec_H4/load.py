from pathlib import Path
from PIL import Image
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label
import math
import matplotlib.pyplot as plt


class Fingerprint:
    def __init__(self, data_path='./data/', normalize=True, crop = True, partition = "real_easy_np"):
        self.data_path = data_path
        self.data_np = self.load_partition(partition)
        if crop:
            self.data_np = self.data_np[:,:,:,2:98]
        if normalize:
            self.data_np = self.norm(self.data_np)

        self.data_len = self.data_np.shape[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_tensor = torch.tensor(self.data_np, dtype=torch.float32).to(self.device)


    def norm(self, img):
        return 2*img - 1.

    def denorm(self, img):
        return (img+1)/2


    def load_partition(self, partition = "real_easy_np"):
        if partition == "easy_np":
            return np.load(self.data_path+"easy_np.npy")
        elif partition == "medium_np":
            return np.load(self.data_path+"medium_np.npy")
        elif partition == "hard_np":
            return np.load(self.data_path+"hard_np.npy")
        elif partition == "real_np":
            return np.load(self.data_path+"real_np.npy")
        elif partition == "real_easy_np":
            real_np = np.load(self.data_path + "real_np.npy")
            easy_np = np.load(self.data_path + "easy_np.npy")
            # medium_np = np.load(self.data_path + "medium_np.npy")
            # hard_np = np.load(self.data_path + "hard_np.npy")
            # return np.concatenate([real_np, easy_np, medium_np, hard_np])
            return np.concatenate([real_np, easy_np])
        else:
            raise NotImplementedError

    def fingerprint_connected_component_oracle(self, imgs, max_components=80, ridge_dark=True):
        """
        imgs: np.ndarray or torch.Tensor with shape [B, 1, H, W]
        returns:
            feasible: shape [B], bool
            num_components: shape [B], int
        """
        is_tensor = isinstance(imgs, torch.Tensor)
        device = imgs.device if is_tensor else None

        imgs_np = imgs.detach().cpu().numpy() if is_tensor else np.asarray(imgs)
        imgs_np = imgs_np[:, 0] if imgs_np.ndim == 4 else imgs_np  # [B, H, W]

        feasibles, counts = [], []

        for img in imgs_np:
            img = img.astype(np.float32)
            if img.max() > 1:
                img = img / 255.0

            thresh = threshold_otsu(img)
            foreground = img < thresh if ridge_dark else img > thresh

            skel = skeletonize(foreground)
            num_components = int(label(skel, connectivity=2).max())

            counts.append(num_components)
            feasibles.append(num_components <= max_components)

        feasibles = np.asarray(feasibles, dtype=bool)
        counts = np.asarray(counts, dtype=np.int64)

        if is_tensor:
            return (
                torch.as_tensor(feasibles, device=device, dtype=torch.bool),
                torch.as_tensor(counts, device=device, dtype=torch.long),
            )

        return feasibles, counts

    def get_samples(self, batchsize):
        idx = torch.randint(0, self.data_len, (batchsize,), device=self.device)
        return self.data_tensor[idx]

    def viz(self, matrix, index=None, save_name = None, oracle = True):
        is_tensor = isinstance(matrix, torch.Tensor)

        if is_tensor:
            data = matrix.detach().cpu().numpy()
        else:
            data = np.asarray(matrix)

        if data.ndim != 4 or data.shape[1] != 1:
            raise ValueError(f"Expected shape [B, 1, H, W], got {data.shape}")

        B = data.shape[0]

        if index is None:
            n_show = min(5, B)
            index = np.random.choice(B, size=n_show, replace=False)
        else:
            index = np.asarray(index, dtype=int)

        n = len(index)
        ncols = min(5, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))

        axes = np.asarray(axes).reshape(-1)

        for ax_id, ax in enumerate(axes):
            ax.axis("off")

            if ax_id >= n:
                continue

            idx = int(index[ax_id])
            img = data[idx, 0]

            # If image is normalized to [-1, 1], map back to [0, 1] for display.
            if img.min() < 0:
                img = (img + 1.0) / 2.0

            img = np.clip(img, 0.0, 1.0)

            ax.imshow(img, cmap="gray")
            if oracle:
                feasibility, counts = self.fingerprint_connected_component_oracle([img])
                if counts[0]<=50:
                    ax.set_title(f" {counts[0]}", fontsize=18)
                else:
                    ax.set_title(f" {counts[0]}", fontsize=18, color = 'red')

        plt.tight_layout()
        if save_name is not None:
            plt.savefig('./fig/'+save_name+'.png')
        plt.show()