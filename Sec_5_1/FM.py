import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn

import configs



class FlowMatching:

    def __init__(self, device = configs.FM_device , sig_min: float = 0) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.device = device
        self.model = self.get_untrained_model()
        

    def get_untrained_model(self):
        # torch.manual_seed(42)
        # REF https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
        return nn.Sequential(
            nn.Linear(3, 64),
            nn.SELU(),
            nn.LazyLinear(out_features=64),
            nn.SELU(),
            nn.LazyLinear(64),
            nn.SELU(),
            nn.LazyLinear(out_features=64),
            nn.SELU(),
            nn.LazyLinear(64),
            nn.SELU(),
            nn.LazyLinear(out_features=2)
        ).to(self.device)

    def sample_xt_given_x1_x0(self, x0_ND: torch.Tensor, x1_ND: torch.Tensor, t_N: torch.Tensor):
        # N, D = x1_ND.shape
        std1 = self.sig_min
        return (1 - (1 - std1) * t_N[..., None]) * x0_ND + t_N[..., None] * x1_ND


    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None]

    def get_samples(self, dataset, n_samples):
        dataset_size = dataset.shape[0]
        selected_ind = np.random.randint(0, dataset_size - 1, n_samples)
        return dataset[selected_ind]

    def train(self, dataset, epoches, batch_size_N, save_every, lr = configs.FM_lr, save_name = configs.FM_name):
        # mymodel = self.get_untrained_model()
        optimizer = Adam(self.model.parameters(), lr=lr)
        for j in tqdm.tqdm(range(epoches)):
            x1_ND = self.get_samples(dataset, batch_size_N)
            x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
            vt_ND = self.model(model_input)
            flow_loss = self.crieria(ut_ND, vt_ND)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

            if (j + 1) % save_every == 0 or j == 0:
                print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/'+ save_name+'_' + str(j + 1) + '.pth')

        return self.model


    
    def sampler(self, batch_size, FM_step_num = configs.FM_step_num):
        x_prev = torch.randn(batch_size, 2, dtype=torch.float32, device=self.device)
    
        for i in range(FM_step_num):
            t = i/FM_step_num 
            t_tensor_N = t*torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
            with torch.no_grad():
                z = self.model(input_ND)
            x_prev = x_prev + z * 1/FM_step_num
            # x_prev = project_union_boxes(x_prev)
            # print('x')
        return x_prev