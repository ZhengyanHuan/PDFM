import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
from scipy.optimize import linear_sum_assignment
import configs
import torch.nn.functional as F
import os


def get_linear_layer_block(num_layers, input_dim, hidden_dim, output_dim, activation=nn.SELU, dropout=0.0):

    layers = []
    current_dim = input_dim

    for i in range(num_layers):
        # Determine the output dimension for this layer
        if i == num_layers - 1:
            next_dim = output_dim  # Last layer should output the desired output_dim
        else:
            next_dim = hidden_dim  # Intermediate layers use hidden_dim

        # Add a linear layer
        layers.append(nn.Linear(current_dim, next_dim))

        # Add activation function and dropout if this is not the last layer
        if i < num_layers - 1:
            if activation is not None:
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Update current dimension for the next layer
        current_dim = next_dim

    return nn.Sequential(*layers)

class velocity_net(nn.Module):
    def __init__(self, d_model,t_layer_num = 2, x_layer_num = 3, tx_layer_num = 3, hidden_dim = 64):
        super(velocity_net, self).__init__()
        self.t_net = get_linear_layer_block(t_layer_num, 1, hidden_dim, hidden_dim)
        self.x_net = get_linear_layer_block(x_layer_num, d_model, hidden_dim, hidden_dim)
        self.tx_net = get_linear_layer_block(tx_layer_num, hidden_dim, hidden_dim, d_model)


    def forward(self, x, t):
        combined_tx = self.t_net(t) + self.x_net(x)
        combined_tx = F.selu(combined_tx)
        mean = self.tx_net(combined_tx)
        return mean

class p_model_net(nn.Module):
    def __init__(self, d_model,t_layer_num = 2, x_layer_num = 3, tx_layer_num = 3, hidden_dim = 64):
        super(p_model_net, self).__init__()
        self.t_net = get_linear_layer_block(t_layer_num, 1, hidden_dim, hidden_dim)
        self.x_net = get_linear_layer_block(x_layer_num, d_model, hidden_dim, hidden_dim)
        self.tx_net = get_linear_layer_block(tx_layer_num, hidden_dim, hidden_dim, 1)


    def forward(self, x, t):
        combined_tx = self.t_net(t) + self.x_net(x)
        combined_tx = F.selu(combined_tx)
        logit = self.tx_net(combined_tx).squeeze(-1)
        return logit

    def predict_proba(self, inp_Bd, t):
        proba = torch.sigmoid(self.forward(inp_Bd, t))
        # proba[proba > 0.998] = 1
        return torch.clamp(proba, max=1)

class StableStopper:
    def __init__(self, patience = 100):
        self.patience = patience
        self.step = -1
        self.best_low = float("inf")
        self.best_high = float("-inf")
        self.last_low_step = -1
        self.last_high_step = -1

    def update(self, loss):
        self.step += 1

        if loss < self.best_low:
            self.best_low = loss
            self.last_low_step = self.step

        if loss > self.best_high:
            self.best_high = loss
            self.last_high_step = self.step

        no_new_low = (self.step - self.last_low_step) >= self.patience
        no_new_high = (self.step - self.last_high_step) >= self.patience

        return no_new_low and no_new_high

class PDFlowMatching:

    def __init__(self, device=configs.FM_device, sig_min: float = 0,
                 ) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.criteria = nn.MSELoss()
        self.device = device
        self.model = self.get_untrained_model()
        self.model_p = self.get_untrained_model_prob()


    def get_untrained_model(self):
        return velocity_net(d_model=configs.d_model).to(self.device)


    def get_untrained_model_prob(self):
        return p_model_net(d_model=configs.d_model).to(self.device)


    def train_prob_model(self, batch_size, lr, training_epochs, init_ckpt_path = None, inner_batch_size=None, inner_repeat_num=None, save_name = None,
                         verbose = True, FM_step_num = configs.FM_step_num):

        # self.optimizer_p = torch.optim.AdamW(self.model_p.parameters(), lr=lr)

        if not hasattr(self, "optimizer_p"):
            self.optimizer_p = torch.optim.Adam(self.model_p.parameters(), lr=lr)

        for pg in self.optimizer_p.param_groups:
            if pg["lr"] != lr:
                pg["lr"] = lr

        if init_ckpt_path is not None:
            self.model_p.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
        if inner_batch_size is None:
            inner_batch_size = 10000
        if inner_repeat_num is None:
            inner_repeat_num = 100
        self.model_p.train()

        loss_record = np.zeros(training_epochs)
        pbar = tqdm.tqdm(range(training_epochs), disable=not verbose)
        for ep in pbar:
            sampled_trajectories_TBd = self.sample_recordsteps(batch_size, FM_step_num=FM_step_num) # (T,b,d)
            valid_mask_expand_TB1 = (self.check_inside(sampled_trajectories_TBd[-1,:])
                                 .unsqueeze(0).unsqueeze(-1).expand(sampled_trajectories_TBd.shape[0], -1, 1))  # (T, b, 1)
            sampled_trajectories_wfeedback_TB3 = torch.cat([sampled_trajectories_TBd, valid_mask_expand_TB1], dim = -1)

            triandata_N4 = self.add_first_dim_index(sampled_trajectories_wfeedback_TB3, FM_step_num=FM_step_num) # (T*b,d+1+1)

            loss_sum = 0
            for i in range(inner_repeat_num):
                selected_idx = torch.randperm(triandata_N4.shape[0])[:inner_batch_size]

                x_t = triandata_N4[selected_idx, :configs.d_model]
                y = triandata_N4[selected_idx, configs.d_model]
                t = triandata_N4[selected_idx, configs.d_model+1:]

                # inp_B3 = torch.cat([x_t, t], dim = -1)
                logits = self.model_p(x_t, t)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                self.optimizer_p.zero_grad()
                loss.backward()
                self.optimizer_p.step()
                loss_sum += loss.item()

            loss_avg = loss_sum / inner_repeat_num
            loss_record[ep] = loss_avg
            pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
            pbar.set_postfix({
                "loss": f"{loss_avg:.4f}",
            })

            # if verbose:
            #     print(f"[epoch {ep+1}/{training_epochs}] step {ep} loss={loss_avg:.6f}")
            # pbar.set_description(f"[epoch {ep+1}/{training_epochs}]")
            # pbar.set_postfix({
            #     "loss": f"{loss_avg:.4f}",
            # })

        if save_name is not None:
            torch.save(self.model_p.state_dict(), './saved_model/' +save_name )
            # np.save('./saved_model/' + self.PD_DDPM_name + '_loss_record.npy', DDPM_loss_record)
            np.save('./saved_model/'  + save_name.replace('.pth','_') + 'loss_record.npy', loss_record)


    def pcfm_sampler_unit_ball(
    self,
    batch_size,
    FM_step_num=configs.FM_step_num,
    radius=1.0,
    eps=1e-8,
    final_project=True,
):
        """
        PCFM-style constrained sampler for constraint:
    
            distance(x, C) = max(||x||_2 - radius, 0)
    
        So the feasible set is:
    
            C = {x : ||x||_2 <= radius}
    
        Assumes:
            self.model(x, t) returns velocity z
            x.shape == (batch_size, configs.d_model)
            t.shape == (batch_size, 1)
        """
    
        def distance_to_constraint(x):
            return torch.clamp(torch.norm(x, dim=-1) - radius, min=0.0)
    
        def project_to_unit_ball(x):
            norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
            scale = torch.clamp(radius / norm, max=1.0)
            return x * scale
    
        dt = 1.0 / FM_step_num
    
        # Initial noise u_0
        x0 = torch.randn(
            batch_size,
            configs.d_model,
            dtype=torch.float32,
            device=self.device,
        )
    
        x_prev = x0.clone()
    
        for i in range(FM_step_num):
            t = i / FM_step_num
            t_next = (i + 1) / FM_step_num
    
            t_tensor_N = t * torch.ones(
                x_prev.shape[0],
                1,
                device=self.device,
                dtype=torch.float32,
            )
    
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
    
            # PCFM forward shooting:
            # estimate terminal sample from current point
            x_terminal = x_prev + (1.0 - t) * z
    
            # Project estimated terminal sample onto constraint set
            x_terminal_proj = project_to_unit_ball(x_terminal)
    
            # Reverse/interpolate projected terminal sample back to next flow time
            x_next = (1.0 - t_next) * x0 + t_next * x_terminal_proj
    
            x_prev = x_next.detach()
    
        if final_project:
            x_prev = project_to_unit_ball(x_prev)
    
        # Optional diagnostic
        final_distance = distance_to_constraint(x_prev)
    
        return x_prev.detach()


    def add_first_dim_index(self, x, FM_step_num = configs.FM_step_num):
        # x: (T, b, i)
        T, b, i = x.shape
        t_idx = torch.arange(T, device=x.device).view(T, 1, 1).expand(T, b, 1)/FM_step_num
        return torch.cat([x, t_idx], dim=2).reshape(T * b, i + 1)



    def check_inside(self, mat_Ndim):
        return torch.norm(mat_Ndim, dim=-1)<=1


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




    def minibatch_ot_coupling(self, x0, x1):

        cost = torch.cdist(x0, x1, p=2) ** 2  # [B, B]

        # Solve minimum-cost bipartite matching on CPU
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # row_ind should be [0, 1, ..., B-1] up to permutation
        match = torch.empty(x0.shape[0], dtype=torch.long)
        match[row_ind] = torch.tensor(col_ind, dtype=torch.long)

        return match


    def train_vanilla(self, dataset, epoches, batch_size_N, save_every, lr=configs.FM_lr, save_name=configs.FM_name, OT = False):
        # mymodel = self.get_untrained_model()
        optimizer = Adam(self.model.parameters(), lr=lr)
        for j in tqdm.tqdm(range(epoches)):

            if dataset is not None:
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)

                if OT:
                    match_ind = self.minibatch_ot_coupling(x0_ND, x1_ND)
                    x1_ND = x1_ND[match_ind]
            else:
                raise NotImplementedError

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            # model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
            vt_ND = self.model(xt_ND, t_N)
            flow_loss = self.criteria(ut_ND, vt_ND)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

            if (j + 1) % save_every == 0 or j == 0:
                print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')

        return self.model



    def train_fixed_lambda(self, dataset, epoches, batch_size_N, save_every, para_lambda,lr=configs.FM_lr, save_name=configs.PDFM_name,
                             update_pmodel_every = 1, init_ckpt_path = None, FM_step_num = configs.FM_step_num, tval = -1, OT = False, retrain_pmodel = False,
                           early_stop = False, patience = 50, sample_trajectory = True, p_model_batch_size = 1000,
                           p_model_lr = 1e-4):

        # mymodel = self.get_untrained_model()
        if early_stop:
            stopper_flow = StableStopper(patience)
            stopper_constraint = StableStopper(patience)
            if sample_trajectory:
                stopper_probin = StableStopper(patience)
            # stopped = False

        if init_ckpt_path is not None:
            self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
            folder, file = os.path.split(init_ckpt_path)

            if os.path.exists(folder +'/'+ 'pmodel4'+file) and not retrain_pmodel:
                print('pmodel found')
                self.model_p.load_state_dict(torch.load('./saved_model/' + 'pmodel4'+file  , weights_only=True))
            else:
                print("initializing pmodel")
                self.train_prob_model(batch_size=p_model_batch_size, lr=p_model_lr, training_epochs=300, verbose=True,
                                      save_name='pmodel4' + file)
        elif retrain_pmodel:
            print("initializing pmodel")
            self.train_prob_model(batch_size=p_model_batch_size, lr=p_model_lr, training_epochs=300, verbose=True)
        else:
            print("using the current pmodel without retraining")


        print("training PDFM")

        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches))

        flow_loss_record = []
        constraint_loss_record = []
        prob_in_record = []
        for j in pbar:

            if dataset is not None:
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)

                if OT:
                    match_ind = self.minibatch_ot_coupling(x0_ND, x1_ND)
                    x1_ND = x1_ND[match_ind]
            else:
                raise NotImplementedError

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            # model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
            vt_ND = self.model(xt_ND, t_N[:, None])
            flow_loss = self.criteria(ut_ND, vt_ND)

            if sample_trajectory:
                constraint_loss, in_num = self.sample_w_constraintloss(batch_size_N, tval, FM_step_num)
            else:
                end_mask = t_N > tval
                # p_model_input = torch.cat([xt_ND + vt_ND / FM_step_num, (t_N + 1 / FM_step_num)[:, None]], dim=-1)
                feedback = self.model_p.predict_proba(xt_ND + vt_ND / FM_step_num, (t_N + 1 / FM_step_num)[:, None]) * end_mask
                constraint_loss = -feedback.mean()

            loss = constraint_loss * para_lambda + flow_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow_loss_record.append(flow_loss.item())
            constraint_loss_record.append(constraint_loss.item())
            if sample_trajectory:
                prob_in_record.append(in_num / batch_size_N)

            if early_stop:
                if sample_trajectory:
                    stopped = stopper_flow.update(flow_loss.item()) and stopper_constraint.update(constraint_loss.item()) and stopper_probin.update(in_num / batch_size_N)
                else:
                    stopped = stopper_flow.update(flow_loss.item()) and stopper_constraint.update(
                        constraint_loss.item())
                if stopped:
                    print('Early stopping, lambda=' + str(para_lambda))
                    torch.save(self.model.state_dict(),  save_name  + '.pth')
                    np.savez(  save_name + '_loss_record.npz', flow_loss=flow_loss_record,
                             constraint_loss=constraint_loss_record, prob_in=prob_in_record)

                    eval_batch_num = 10
                    eval_batch_size = 100000
                    eval_sample_num = eval_batch_num * eval_batch_size
                    total_in_num = 0
                    for i in range(eval_batch_num):
                        samples = self.sampler(eval_batch_size)
                        valid_mask, in_num = self.get_terminal_feedback(samples)
                        total_in_num += in_num
                    return para_lambda, total_in_num/eval_sample_num

            if (j + 1) % update_pmodel_every == 0:
                self.train_prob_model(batch_size=p_model_batch_size, lr=p_model_lr, training_epochs=update_pmodel_every, verbose=False)

            if (j + 1) % 10 == 0 or j == 0:
                pbar.set_description(f"Epoch {j + 1}")
                if sample_trajectory:
                    pbar.set_postfix({
                        "flow_loss": f"{flow_loss:.4f}",
                        "constraint_loss": f"{constraint_loss:.4f}",
                        "prob_in": f"{in_num / batch_size_N:.4f}",
                    })
                else:
                    pbar.set_postfix({
                        "flow_loss": f"{flow_loss:.4f}",
                        "constraint_loss": f"{constraint_loss:.4f}",
                    })
            if (j + 1) % save_every == 0 or (j == 0 and early_stop == False):
                # print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')
                np.savez('./loss_record/'+save_name+'.npz', flow_loss=flow_loss_record, constraint_loss=constraint_loss_record, prob_in=prob_in_record)

        return para_lambda, 0


    def get_terminal_feedback(self, cur_state_ND):
        valid_mask = self.check_inside(cur_state_ND)
        in_num = torch.sum(valid_mask).item()
        return valid_mask, in_num


    def sampler(self, batch_size, FM_step_num=configs.FM_step_num):
        x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=self.device)

        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / FM_step_num
            # x_prev = project_union_boxes(x_prev)
            # print('x')
        return x_prev

    def sample_recordsteps(self, batch_size, FM_step_num=configs.FM_step_num):

        output = torch.zeros(FM_step_num+1, batch_size, configs.d_model, dtype=torch.float32, device=self.device)
        x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=self.device)
        output[0] = x_prev

        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / FM_step_num
            output[i+1] = x_prev
            # x_prev = project_union_boxes(x_prev)
            # print('x')
        return output


    def sample_w_constraintloss(self, batch_size, t_val, FM_step_num=configs.FM_step_num, normalize = True):

        # output = torch.zeros(FM_step_num+1, batch_size, 2, dtype=torch.float32, device=self.device)
        x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=self.device)
        # output[0] = x_prev

        step_count = 0
        constraint_loss = 0
        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            # with torch.no_grad():
            z = self.model(x_prev, t_tensor_N[:, None])

            if t>=t_val:
                step_count += 1
                # p_model_input = torch.cat([x_prev + z/ FM_step_num, (t_tensor_N + 1/ FM_step_num)[:, None]], dim=-1)
                feedback = self.model_p.predict_proba(x_prev + z/ FM_step_num, (t_tensor_N + 1/ FM_step_num)[:, None])
                constraint_loss -= feedback.mean()

            x_prev = x_prev + z.detach() / FM_step_num

        if normalize and step_count>0:
            constraint_loss /= step_count

        terminal_feedback, in_num = self.get_terminal_feedback(x_prev.detach())
        return constraint_loss, in_num

    def format_float(self, x: float):
        s = str(x)
        if "." in s:
            left, right = s.split(".", 1)
            return f"{left}p{right[:2]}"
        return s[:3]

    def PDFMtrain(self, dataset, max_epoches_per_lambda, batch_size_N, target_csrate, PD_lr, para_lambda_init = None, PD_lr_decay = 0.8,
                  lr = configs.FM_lr,
                  save_name=configs.PDFM_name,
                  update_pmodel_every=1, init_ckpt_path=None, FM_step_num=configs.FM_step_num, tval=-1, OT=False,
                  retrain_pmodel=False, patience = 50, sample_trajectory = True, p_model_batch_size = 1000,
                  p_model_lr=1e-4
        ):

        PD_iter = 1
        os.makedirs("./saved_model/" + save_name, exist_ok=True)
        epoches = max_epoches_per_lambda
        prob_in = np.nan

        while True:
            if PD_iter == 1:
                if init_ckpt_path is not None:
                    self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
                    ckpt_path = init_ckpt_path
                else:
                    vanilla_epoches = 50000
                    self.train_vanilla(dataset, epoches = vanilla_epoches, batch_size_N = 10000, save_every = vanilla_epoches, lr=configs.FM_lr, save_name=configs.FM_name, OT = True)
                    ckpt_path = './saved_model/' + configs.FM_name + '_' + str(vanilla_epoches) + '.pth'

                if para_lambda_init is not None:
                    paralambda = para_lambda_init

                else:
                    paralambda = 0
                    if init_ckpt_path is not None:
                        self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
                        eval_batch_num = 10
                        eval_batch_size = 100000
                        eval_sample_num = eval_batch_num * eval_batch_size
                        total_in_num = 0
                        for i in range(eval_batch_num):
                            samples = self.sampler(eval_batch_size)
                            valid_mask, in_num = self.get_terminal_feedback(samples)
                            total_in_num += in_num

                        prob_in = total_in_num / eval_sample_num
                        paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                        PD_lr = PD_lr_decay * PD_lr
            else:
                paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                PD_lr = PD_lr_decay * PD_lr
                ckpt_path = None

            # print("Iter:", PD_iter, "lambda:", paralambda)
            save_name_PD = "./saved_model/" + save_name + "/" + save_name + '_' + str(
                PD_iter) + '_' + self.format_float(paralambda)

            print("Iter:", PD_iter, "lambda:", paralambda, "prob_in:", prob_in)
            paralambda, prob_in = self.train_fixed_lambda(dataset, epoches, batch_size_N, save_every=epoches,
                                                          para_lambda=paralambda, lr=lr, save_name=save_name_PD,
                                                          update_pmodel_every=update_pmodel_every,
                                                          init_ckpt_path=ckpt_path, FM_step_num=FM_step_num,
                                                          tval=tval, OT=OT, retrain_pmodel=retrain_pmodel,
                                                          early_stop=True, patience=patience, sample_trajectory = sample_trajectory,
                                                          p_model_batch_size=p_model_batch_size, p_model_lr=p_model_lr)
            PD_iter += 1


                # while True:
                #
                #     PD_iter += 1
                #     paralambda = max(paralambda + PD_lr*(target_csrate-prob_in), 0)
                #     save_name_PD = "./saved_model/"+ save_name +"/"+save_name + '_' + str(PD_iter) + '_' + self.format_float(paralambda)
                #     print("Iter:", PD_iter, "lambda:", paralambda, "prob_in:", prob_in)
                #     paralambda, prob_in = self.train_fixed_lambda(dataset, epoches, batch_size_N, save_every=epoches,
                #                                                   para_lambda=paralambda, lr=lr, save_name=save_name_PD,
                #                                                   update_pmodel_every=update_pmodel_every,
                #                                                   init_ckpt_path=None, FM_step_num=FM_step_num,
                #                                                   tval=tval, OT=OT, retrain_pmodel=retrain_pmodel,
                #                                                   early_stop=True, patience = patience)
                #
                #     save_name_PD = "./saved_model/"+ save_name +"/"+save_name + '_' + str(PD_iter) + '_' +self.format_float(paralambda)
                #     paralambda, prob_in = self.train_fixed_lambda(dataset, epoches, batch_size_N, save_every = epoches, para_lambda = paralambda,lr=lr,
                #                                                   save_name=save_name_PD,
                #                          update_pmodel_every = update_pmodel_every, init_ckpt_path = init_ckpt_path,
                #                                                   FM_step_num = FM_step_num, tval = tval, OT = OT, retrain_pmodel = retrain_pmodel,
                #                        early_stop = True, patience = patience)
                #
                #     PD_lr = PD_lr_decay * PD_lr
