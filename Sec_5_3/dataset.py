import jax
import numpy as np
import exponax as ex
import jax.numpy as jnp
import configs
import matplotlib.pyplot as plt
import torch

def generate_ks_dataset(
    batch_size,
    horizon= configs.dim * configs.horizon_multiplier - 1,
    n=configs.dim,
    dt=configs.global_dt,
    num_spatial_dims=1,
    domain_extent=configs.global_domain_extent,
    cutoff=configs.global_cutoff,
    seed=0,
    order = configs.global_order,
    save = False
):
    ks_stepper = ex.stepper.KuramotoSivashinskyConservative(
        num_spatial_dims=num_spatial_dims,
        domain_extent=domain_extent,
        num_points=n,
        dt=dt,
        order=order
    )

    ic = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=1,
        cutoff=cutoff,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    u0_batch = jax.vmap(lambda k: ic(num_points=n, key=k))(keys)   # (B, 1, n)

    rollout_fn = ex.rollout(ks_stepper, horizon, include_init=True)
    trajectories = jax.vmap(rollout_fn)(u0_batch)                  # (B, horizon+1, 1, n)
    trajectories = np.asarray(jax.device_get(trajectories[:, :, 0, :]))  # (B, horizon+1, n)

    samples = []
    for b in range(batch_size):
        for start in range(0, horizon + 1 - n + 1, n):
            samples.append(trajectories[b, start:start + n, :].T)    # (n, n)

    dataset = np.stack(samples, axis=0)
    result = check_nans(dataset)
    print(result)

    if save:
        np.save("./data/ks_dataset.npy", dataset)

    return dataset

def check_nans(dataset):
    has_nan = np.isnan(dataset).any()
    num_nans = np.isnan(dataset).sum()
    nan_sample_indices = np.where(np.isnan(dataset).any(axis=(1, 2)))[0]

    return {
        "has_nan": has_nan,
        "num_nans": int(num_nans),
        "num_samples_with_nan": int(len(nan_sample_indices)),
        "nan_sample_indices": nan_sample_indices,
    }

def ks_batch_residual(
    samples,
    dt=configs.global_dt,
    num_spatial_dims=1,
    domain_extent=configs.global_domain_extent,
        order=configs.global_order,
):
    batch_size, n, _ = samples.shape

    ks_stepper = ex.stepper.KuramotoSivashinskyConservative(
        num_spatial_dims=num_spatial_dims,
        domain_extent=domain_extent,
        num_points=n,
        dt=dt,
        order=order
    )

    samples = jnp.asarray(samples)            # (B, space, time)
    current = samples[:, :, 0][:, None, :]    # (B, 1, space)

    residuals = []
    for t in range(n - 1):
        pred_next = jax.vmap(ks_stepper)(current)[:, 0, :]   # (B, space)
        residual = samples[:, :, t + 1] - pred_next          # (B, space)
        residuals.append(residual)
        current = samples[:, :, t + 1][:, None, :]           # use sample state

    return np.asarray(jnp.stack(residuals, axis=2))          # (B, space, time-1)




def ks_batch_residual_Ln(
    samples,
    norm=configs.Ln,
    dt=configs.global_dt,
    num_spatial_dims=1,
    domain_extent=configs.global_domain_extent,
        order=configs.global_order,
):
    residuals = ks_batch_residual(
        samples,
        dt=dt,
        num_spatial_dims=num_spatial_dims,
        domain_extent=domain_extent,
        order=order,
    )

    if norm == "L1":
        return np.sum(np.abs(residuals), axis=(1, 2))
    elif norm == "L2":
        return np.sqrt(np.sum(residuals**2, axis=(1, 2)))
    elif norm == "Linf":
        return np.max(np.abs(residuals), axis=(1, 2))

def ks_show(sample):
    plt.imshow(sample, aspect='equal', cmap='RdBu', origin="lower")
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.show()



class KSDataset():
    def __init__(self, dataset_path = configs.dataset_path, device = configs.device, eps=0, degrade_interval = None,
                 discard_infeasible = False, threshold = configs.default_thres):
        self.device = device
        self.eps = eps

        data = np.load(dataset_path).astype(np.float32)
        if degrade_interval is not None:
            data = self.degrade_resolution(data, pattern = degrade_interval)
        
        self.mean = data.mean()
        self.std = data.std()

        if discard_infeasible:
            residual = ks_batch_residual_Ln(data)/(63*64)
            valid_mask = residual<threshold
            print("Total sample num: "+str(data.shape[0]))
            print("Remaining sample num: "+str(np.sum(valid_mask)))
            data = data[valid_mask]

        data = (data - self.mean) / (self.std + self.eps)
        data = data[:, None, :, :]  # [N, D, D] -> [N, 1, D, D]

        self.data = torch.tensor(data, device=self.device)
        self.data_len = self.data.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def denorm(self, x):
        if isinstance(x, np.ndarray):
            return x * (self.std + self.eps) + self.mean
        return x * (self.std + self.eps) + self.mean

    def get_samples(self, batchsize):
        idx = torch.randint(0, self.data_len, (batchsize,), device=self.device)
        return self.data[idx]

    def degrade_resolution(self, x, pattern, s=2):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
    
        if x.ndim != 3 or x.shape[1:] != (64, 64):
            raise ValueError(f"x must have shape (N, 64, 64), got {x.shape}")
    
        if (
            not isinstance(pattern, tuple)
            or len(pattern) != 2
            or not all(isinstance(v, int) for v in pattern)
        ):
            raise TypeError("pattern must be a tuple of two integers: (i, j)")
    
        i, j = pattern
    
        if i <= 0 or j < 0:
            raise ValueError("pattern must satisfy i > 0 and j >= 0")
    
        if s not in (1, 2):
            raise ValueError("currently only s=1 (hold) and s=2 (linear) are supported")
    
        S = 64
        y = x.astype(np.result_type(x, np.float32), copy=True)
    
        if j == 0:
            return x.copy()
    
        L = i + j
        pos = np.arange(S)
    
        # Position within each cycle
        r = pos % L
    
        # Exact points: first i positions of each cycle
        exact_mask = r < i
    
        # Interpolated points: next j positions of each cycle
        interp_mask = ~exact_mask
    
        # Left anchor for every position:
        # last exact pixel in the current cycle
        cycle_start = (pos // L) * L
        left_anchor = cycle_start + (i - 1)
    
        # Right anchor for every position:
        # first exact pixel of the next cycle
        right_anchor = cycle_start + L
    
        # Clamp to valid range
        left_anchor = np.clip(left_anchor, 0, S - 1)
        right_anchor_clipped = np.clip(right_anchor, 0, S - 1)
    
        if s == 1:
            y[:, interp_mask, :] = x[:, left_anchor[interp_mask], :]
        else:
            # Only interpolate where a true next anchor exists
            valid_interp = interp_mask & (right_anchor < S)
    
            p = pos[valid_interp]
            la = left_anchor[valid_interp]
            ra = right_anchor[valid_interp]
    
            # Fraction inside the interpolation segment:
            # for positions la+1, ..., la+j between la and ra=la+j+1
            alpha = (p - la) / (ra - la)
    
            x_left = x[:, la, :]
            x_right = x[:, ra, :]
    
            y[:, valid_interp, :] = (
                (1.0 - alpha)[None, :, None] * x_left
                + alpha[None, :, None] * x_right
            )
    
            # If the final cycle has interpolated positions but no next anchor,
            # keep them unchanged rather than extrapolating.
            invalid_interp = interp_mask & (right_anchor >= S)
            y[:, invalid_interp, :] = x[:, invalid_interp, :]
    
        # Ensure exact points remain exactly unchanged
        y[:, exact_mask, :] = x[:, exact_mask, :]
    
        return y.astype(x.dtype, copy=False)







    