import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs
import tqdm
import math


class DDPM_class:
    def __init__(
        self,
        T = configs.T,
        beta_start = configs.beta_start,
        beta_end = configs.beta_end,
        learning_rate = configs.DDPM_lr,
        training_epochs = configs.DDPM_training_epochs,
        devcie = configs.DDPM_device,
        DDPM_name = configs.DDPM_name
    ):
        # hyperparameters
        super().__init__()

        self.T = int(T)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.learning_rate = float(learning_rate)
        self.training_epochs = int(training_epochs)
        self.device = devcie
        self.DDPM_name = DDPM_name
        self.criterion = torch.nn.MSELoss(reduction='none')

        # linear schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # move to device
        self.betas = betas.to(self.device)                          # [T]
        self.alphas = alphas.to(self.device)                        # [T]
        self.alpha_bar = alpha_bar.to(self.device)                  # [T]
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)            # [T]
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)  # [T]
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)      # [T]


        # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        ######NOTICE
        alpha_bar_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alpha_bar[:-1]], dim=0
        )
        self.posterior_variance = (
            self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)
        self.coeff = (self.betas ** 2 / (2 * self.posterior_variance * self.alphas * (1 - self.alpha_bar)))
        self.coeff = self.coeff.clamp(max = self.coeff[1])
        # model (you already have this)
        self.model = self.get_untrained_model()
        self.model.to(self.device)

    def get_untrained_model(self):
        class EpsMLP(nn.Module):
            def __init__(self, T, device):
                super().__init__()
                self.T = T
                self.time_emb_dim = 32
                hidden = 64

                self.time_mlp = nn.Sequential(
                    nn.Linear(self.time_emb_dim, hidden),
                    nn.SELU(),
                    nn.Linear(hidden, hidden),
                    nn.SELU(),
                )

                self.net = nn.Sequential(
                    nn.Linear(2 + hidden, hidden),
                    nn.SELU(),
                    nn.Linear(hidden, hidden),
                    nn.SELU(),
                    nn.Linear(hidden, hidden),
                    nn.SELU(),
                    nn.Linear(hidden, hidden),
                    nn.SELU(),
                    nn.Linear(hidden, 2),
                ).to(device)

            def sinusoidal_time_embedding(self, t):
                # t: [B]
                half_dim = self.time_emb_dim // 2
                t = t.float() / (self.T - 1)  # normalize to [0, 1]

                freqs = torch.exp(
                    -math.log(10000) * torch.arange(half_dim, device=t.device).float() / (half_dim - 1)
                )  # [half_dim]

                args = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]
                emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, time_emb_dim]
                return emb

            def forward(self, x_t, t):
                t_emb = self.sinusoidal_time_embedding(t)  # [B, 32]
                t_feat = self.time_mlp(t_emb)  # [B, 64]
                inp = torch.cat([x_t, t_feat], dim=1)  # [B, 66]
                return self.net(inp)  # [B, 2]

        return EpsMLP(self.T, self.device)

    def _extract(self, a, t, x_shape):
        # a: [T], t: [B] -> [B, 1, 1, ...] for broadcasting to x_shape
        B = t.shape[0]
        out = a.gather(0, t)  # [B]
        return out.view(B, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise
        return (
            self._extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )


    def train_ddpm(self, dataset, batch_size, epochs = None, log_every = 1000, lr = None):
        self.model.train()
        lr = lr if lr is not None else self.learning_rate
        num_epochs = int(epochs) if epochs is not None else self.training_epochs
        loss_record = np.zeros(num_epochs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for ep in tqdm.tqdm(range(num_epochs)):
            random_indices = np.random.choice(dataset.shape[0], batch_size)
            x0 = dataset[random_indices]
            x0 = x0.to(self.device)

            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)

            eps_pred = self.model(x_t, t)
            loss = (self.criterion(eps_pred, noise) * self._extract(self.coeff, t, x0.shape)).mean()

            loss_record[ep] = loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if (ep + 1) % log_every == 0:
                print(f"[epoch {ep+1}/{num_epochs}] step {ep} loss={loss.item():.6f}")
                torch.save(self.model.state_dict(), './saved_model/' + self.DDPM_name + '_' + str(ep + 1) + '.pth')
                np.save('./saved_model/'+ self.DDPM_name + '_loss_record.npy', loss_record)

    # -------------------- sampling --------------------
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse step:
          mean = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-alpha_bar_t))*eps_theta)
          x_{t-1} = mean + sqrt(posterior_variance_t)*z  (if t>0 else mean)
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_1mab_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)

        eps_theta = self.model(x_t, t)
        mean = sqrt_recip_alpha_t * (x_t - (betas_t / sqrt_1mab_t) * eps_theta)

        var_t = self._extract(self.posterior_variance, t, x_t.shape)
        z = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(var_t) * z

    @torch.no_grad()
    def sample(self, n, sample_shape, batch_size) :

        self.model.eval()

        out = []
        remaining = int(n)
        while remaining > 0:
            b = min(batch_size, remaining)
            x = torch.randn((b, *sample_shape), device=self.device)

            for ti in reversed(range(self.T)):
                t = torch.full((b,), ti, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t)

            out.append(x.detach().cpu())
            remaining -= b

        return torch.cat(out, dim=0)


    def make_ddim_schedule(self, ddim_steps: int):
        """
        Returns a LongTensor of shape [ddim_steps] with indices in [0, T-1],
        decreasing from T-1 to 0.
        Common choice: uniform spacing in index space.
        """
        ddim_steps = int(ddim_steps)

        steps = torch.linspace(self.T - 1, 0, ddim_steps, device=self.device)
        steps = steps.round().long()
        # Ensure strictly decreasing (remove duplicates after rounding)
        steps = torch.unique_consecutive(steps)
        # If rounding collapsed too much, fall back to exact stride
        if steps.numel() < ddim_steps:
            # integer stride fallback
            stride = (self.T - 1) / (ddim_steps - 1)
            steps = torch.tensor(
                [int(round((self.T - 1) - i * stride)) for i in range(ddim_steps)],
                device=self.device,
                dtype=torch.long,
            )
            steps = torch.unique_consecutive(steps)
        # guarantee last is 0
        if steps[-1].item() != 0:
            steps[-1] = 0
        return steps


    @torch.no_grad()
    def ddim_step(self, x_t, t, t_prev, eta: float = 0.0):
        """
        Jump from timestep t to timestep t_prev (where t_prev < t) using DDIM.
        t and t_prev are LongTensor shape [b].
        """
        # Extract alpha_bar at t and t_prev
        a_bar_t = self._extract(self.alpha_bar, t, x_t.shape)          # \bar{alpha}_t
        a_bar_prev = self._extract(self.alpha_bar, t_prev, x_t.shape)  # \bar{alpha}_{t_prev}

        eps_theta = self.model(x_t, t)

        # x0 estimate: (x_t - sqrt(1-a_bar_t)*eps) / sqrt(a_bar_t)
        sqrt_a_bar_t = torch.sqrt(a_bar_t)
        sqrt_1m_a_bar_t = torch.sqrt(1.0 - a_bar_t)
        x0_hat = (x_t - sqrt_1m_a_bar_t * eps_theta) / sqrt_a_bar_t

        # DDIM sigma
        eps = 1e-12
        sigma = eta * torch.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_t + eps)) * torch.sqrt(
            1.0 - (a_bar_t / (a_bar_prev + eps))
        )

        # direction coefficient
        dir_coeff = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma**2, min=0.0))

        x_prev = torch.sqrt(a_bar_prev) * x0_hat + dir_coeff * eps_theta

        # add noise unless t_prev == 0? (in general, you add noise unless you're at final)
        # Here: if t_prev == 0, many implementations set noise to 0; we’ll mask it.
        z = torch.randn_like(x_t)
        nonzero_mask = (t_prev != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        x_prev = x_prev + nonzero_mask * sigma * z

        return x_prev


    @torch.no_grad()
    def sample_ddim(self, n, sample_shape, batch_size, ddim_steps: int = 100, eta: float = 0.0):
        """
        DDIM sampling with fewer than T steps.
        - ddim_steps: number of sampling steps K
        - eta: 0 => deterministic DDIM
        """
        self.model.eval()

        # Create schedule tau: [K] decreasing indices (e.g., [T-1, ..., 0])
        tau = self.make_ddim_schedule(ddim_steps)

        out = []
        remaining = int(n)
        while remaining > 0:
            b = min(batch_size, remaining)
            x = torch.randn((b, *sample_shape), device=self.device)

            # iterate over pairs (tau_i -> tau_{i+1})
            for i in range(tau.numel() - 1):
                ti = tau[i].item()
                ti_prev = tau[i + 1].item()

                t = torch.full((b,), ti, device=self.device, dtype=torch.long)
                t_prev = torch.full((b,), ti_prev, device=self.device, dtype=torch.long)

                x = self.ddim_step(x, t, t_prev, eta=eta)

            out.append(x.detach().cpu())
            remaining -= b

        return torch.cat(out, dim=0)