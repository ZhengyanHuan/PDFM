import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
from scipy.optimize import linear_sum_assignment
import configs
import torch.nn.functional as F
import os
import NNstructure
import dataset
import math

def mem_gb():
    return torch.cuda.memory_allocated() / 1024**3


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

    def __init__(self, dataset, thres = configs.default_thres, device = configs.device):
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.sig_min = 0
        self.model = self.get_untrained_model()
        self.model_p = self.get_untrained_model_prob()
        self.criteria = nn.MSELoss()
        self.thres = thres

    def get_untrained_model(self):
        return NNstructure.SpaceTimeModel("velocity_estimator").to(self.device)

    def get_untrained_model_prob(self):
        return NNstructure.SpaceTimeModel("classifier").to(self.device)

    def check_inside(self, cur_state_N1DD, thres, Ln = configs.Ln):
        dataNDD_normed = cur_state_N1DD.cpu().numpy()[:, 0, :, :]
        dataNDD = self.dataset.denorm(dataNDD_normed)
        ln_norm = dataset.ks_batch_residual_Ln(dataNDD) / (configs.dim * (configs.dim -1))
        return torch.tensor(ln_norm < thres, dtype = torch.float32, device = self.device)

    def get_terminal_feedback(self, cur_state_N1DD, thres):
        valid_mask = self.check_inside(cur_state_N1DD, thres)
        in_num = torch.sum(valid_mask).item()
        return valid_mask, in_num

    def ut_given_x1(self, xt_N1TD, x1_N1TD, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_N1TD - diff * xt_N1TD
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None, None, None]

    def sample_xt_given_x1_x0(self, x0_N1TD: torch.Tensor, x1_N1tD: torch.Tensor, t_N: torch.Tensor):
        # N, D = x1_1ND.shape
        std1 = self.sig_min
        return (1 - (1 - std1) * t_N[..., None, None, None]) * x0_N1TD + t_N[..., None, None, None] * x1_N1tD

    def train_FM_lambda0(self, epoches, batch_size_N, save_every, lr=configs.FM_lr, save_name=configs.FM_name, verbose=True):
        # print( mem_gb())
        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches), disable=not verbose)
        loss_record = np.zeros(epoches)
        for j in pbar:
            # print( mem_gb())
            x1_1ND = self.dataset.get_samples( batch_size_N)
            x0_1ND = torch.randn_like(x1_1ND, device=self.device, dtype=torch.float32)

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_1ND = self.sample_xt_given_x1_x0(x0_1ND, x1_1ND, t_N)
            ut_1ND = self.ut_given_x1(xt_1ND, x1_1ND, t_N)
            
            vt_1ND = self.model(xt_1ND, t_N)

            # print( mem_gb())
            flow_loss = self.criteria(ut_1ND, vt_1ND)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

            # print( mem_gb())
            loss_record[j] = flow_loss.item()
            pbar.set_description(f"Epoch {j + 1}/{epoches}")
            pbar.set_postfix({
                "loss": f"{flow_loss:.4f}",
            })

            if (j + 1) % save_every == 0 or j == 0:
                print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')
                np.save('./saved_model/' + save_name + '_loss_record.npy', loss_record)

        return self.model

    def sampler(self, batch_size, default_generation_step=configs.default_generation_step):
        x_prev = torch.randn(batch_size, 1, configs.dim, configs.dim, dtype=torch.float32, device=configs.device)
        for i in range(default_generation_step):
            t = i / default_generation_step
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
                # print(torch.mean(torch.abs(z)))
            x_prev = x_prev + z / default_generation_step
        return x_prev

    def sample_recordsteps(self, batch_size, FM_step_num=configs.default_generation_step):

        output = torch.randn(FM_step_num+1, batch_size, 1, configs.dim, configs.dim, dtype=torch.float32, device=configs.device)
        x_prev = torch.randn(batch_size, 1, configs.dim, configs.dim, dtype=torch.float32, device=configs.device)

        output[0] = x_prev

        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
            x_prev = x_prev + z * 1 / FM_step_num
            output[i+1] = x_prev

        return output

    def time_index(self, x, FM_step_num = configs.default_generation_step):
        # x: TB1DD
        T, b = x.shape[0], x.shape[1]
        t_idx = torch.arange(T, device=x.device).unsqueeze(1).expand(T, b)/FM_step_num
        return t_idx

    def train_prob_model(self, batch_size, lr, training_epochs, init_ckpt_path = None,
                         inner_batch_size=None, inner_repeat_num=None, save_name = None,
                         verbose = True, FM_step_num = configs.default_generation_step, t_val = -1):

        if not hasattr(self, "optimizer_p"):
            self.optimizer_p = torch.optim.Adam(self.model_p.parameters(), lr=lr)

        for pg in self.optimizer_p.param_groups:
            if pg["lr"] != lr:
                pg["lr"] = lr

        if init_ckpt_path is not None:
            self.model_p.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
        if inner_batch_size is None:
            inner_batch_size = 1000
        if inner_repeat_num is None:
            inner_repeat_num = 100
        self.model_p.train()

        loss_record = np.zeros(training_epochs*inner_repeat_num)
        j = 0
        pbar = tqdm.tqdm(range(training_epochs), disable=not verbose)
        for ep in pbar:
            sampled_trajectories_TB1DD = self.sample_recordsteps(inner_batch_size, FM_step_num=FM_step_num)

            valid_mask = self.check_inside(sampled_trajectories_TB1DD[-1,:], thres= self.thres)
                                 
            valid_mask_expand_TB = valid_mask.unsqueeze(0).expand(sampled_trajectories_TB1DD.shape[0], -1)

            t_idx = self.time_index(sampled_trajectories_TB1DD).flatten(0,1) #(T*B)
            # print(t_idx.dtype)
            sampled_trajectories_TB1DD_flatten = sampled_trajectories_TB1DD.flatten(0,1) #(T*B, 1, D, D)
            # print(sampled_trajectories_TB1DD_flatten.dtype)
            valid_mask_expand_TB_flatten = valid_mask_expand_TB.flatten(0,1)
            # print(valid_mask_expand_TB_flatten.dtype)

            t_mask = t_idx>t_val
            t_idx = t_idx[t_mask]
            sampled_trajectories_TB1DD_flatten = sampled_trajectories_TB1DD_flatten[t_mask]
            valid_mask_expand_TB_flatten = valid_mask_expand_TB_flatten[t_mask]
            
            
            loss_sum = 0
            for i in range(inner_repeat_num):
                idx = torch.randint(0, t_idx.shape[0], (batch_size,), device=self.device)
                xB1DD = sampled_trajectories_TB1DD_flatten[idx]
                yB = valid_mask_expand_TB_flatten[idx]
                # print(y)
                tB = t_idx[idx]
                logits = self.model_p(xB1DD, tB)
                loss = F.binary_cross_entropy_with_logits(logits, yB)

                self.optimizer_p.zero_grad()
                loss.backward()
                self.optimizer_p.step()
                loss_sum += loss.item()

                loss_avg = loss_sum / inner_repeat_num
                loss_record[j] = loss.item()
                j+=1
                pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                })
            # pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
            # pbar.set_postfix({
            #     "loss": f"{loss_avg:.4f}",
            # })

        if save_name is not None:
            torch.save(self.model_p.state_dict(), './saved_model/' +save_name )
            # np.save('./saved_model/' + self.PD_DDPM_name + '_loss_record.npy', DDPM_loss_record)
            np.save('./saved_model/'  + save_name.replace('.pth','_') + 'loss_record.npy', loss_record)
        
        in_num = torch.sum(valid_mask).item()
        return valid_mask, in_num

    def format_float(self, x: float):
        s = str(x)
        if "." in s:
            left, right = s.split(".", 1)
            return f"{left}p{right[:2]}"
        return s[:3]

    def train_fixed_lambda(self, epoches, batch_size_N, save_every, para_lambda,lr=configs.FM_lr, save_name=configs.PDFM_name,
                         update_pmodel_every = 10, init_ckpt_path = None, FM_step_num = configs.default_generation_step, tval = -1,
                            early_stop = False, patience = 50, retrain_pmodel = False):
        if early_stop:
            stopper_flow = StableStopper(patience)
            stopper_constraint = StableStopper(patience)
            
        if init_ckpt_path is not None and not retrain_pmodel:
            self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
            folder, file = os.path.split(init_ckpt_path)
            # print(folder + 'pmodel4'+file)

            if os.path.exists(folder + '/pmodel4'+file):
                print('pmodel found')
                self.model_p.load_state_dict(torch.load('./saved_model/' + 'pmodel4'+file , weights_only=True))
            else:
                print("initializing pmodel")
                self.train_prob_model(batch_size=1000, lr=1e-4, training_epochs=100, verbose=True,
                                      save_name='pmodel4' + file, t_val = tval)
        elif retrain_pmodel:
            print("initializing pmodel")
            # print("initializing pmodel")
            self.train_prob_model(batch_size=1000, lr=1e-4, training_epochs=300, verbose=False)
        else:
            print("using the current pmodel without retraining")

        print("training PDFM")

        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches))

        flow_loss_record = np.array([])
        constraint_loss_record = np.array([])
        prob_in_record = np.array([])
        
        for j in pbar:
            x1_1ND = self.dataset.get_samples( batch_size_N)
            x0_1ND = torch.randn_like(x1_1ND, device=self.device, dtype=torch.float32)

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_1ND = self.sample_xt_given_x1_x0(x0_1ND, x1_1ND, t_N)
            ut_1ND = self.ut_given_x1(xt_1ND, x1_1ND, t_N)
            
            vt_1ND = self.model(xt_1ND, t_N)
            flow_loss = self.criteria(ut_1ND, vt_1ND)

            end_mask = t_N>tval
            feedback = self.model_p.predict_proba(xt_1ND + vt_1ND / FM_step_num, t_N+ 1/ FM_step_num) * end_mask
            constraint_loss = -feedback.mean()

            loss = constraint_loss * para_lambda + flow_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                 
                    
            pmodel_inner_batch_size = 300
            if (j + 1) % update_pmodel_every == 0:
                valid_mask, in_num = self.train_prob_model(batch_size=300, lr=5e-5, training_epochs=1, verbose=False,
                                                          inner_batch_size=pmodel_inner_batch_size, inner_repeat_num=30)
                pbar.set_description(f"Epoch {j + 1}")
                pbar.set_postfix({
                    "flow_loss": f"{flow_loss:.4f}",
                    "constraint_loss": f"{constraint_loss:.4f}",
                    "prob_in": f"{in_num / pmodel_inner_batch_size:.4f}",
                })
                flow_loss_record = np.append(flow_loss_record, flow_loss.item())
                constraint_loss_record = np.append(constraint_loss_record, constraint_loss.item())
                prob_in_record = np.append(prob_in_record, (in_num/pmodel_inner_batch_size))
                
            if early_stop:
                stopped = stopper_flow.update(flow_loss.item()) and stopper_constraint.update(constraint_loss.item())     
                if j == epoches - 1:
                    stopped = 1
                if stopped:
                    torch.save(self.model.state_dict(), './saved_model/'+ save_name  + '.pth')          
                    np.savez('./saved_model/' + save_name + '_record.npz', flow_loss_record=flow_loss_record, 
                         constraint_loss_record=constraint_loss_record, prob_in_record=prob_in_record)           
                    eval_batch_num = 10
                    eval_batch_size = 1000
                    eval_sample_num = eval_batch_num * eval_batch_size
                    total_in_num = 0
                    for i in range(eval_batch_num):
                        samples = self.sampler(eval_batch_size)         
                        valid_mask, in_num = self.get_terminal_feedback(samples, thres = self.thres)
                        total_in_num += in_num
                    print('Early stopping, lambda=' + str(para_lambda)+' feasible rate:'+ str(total_in_num/eval_sample_num))
                    return para_lambda, total_in_num/eval_sample_num       

            
            if (j + 1) % save_every == 0:
                # print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')
                np.savez('./saved_model/' + save_name + '_record.npz', flow_loss_record=flow_loss_record, 
                         constraint_loss_record=constraint_loss_record, prob_in_record=prob_in_record)
        return para_lambda, 0

    def PDFMtrain(self, max_epoches_per_lambda, batch_size_N, target_csrate, PD_lr, para_lambda_init = None, PD_lr_decay = 0.8,
                  lr = configs.FM_lr,
                  save_name=configs.PDFM_name,
                  update_pmodel_every=10, init_ckpt_path=None, FM_step_num=configs.default_generation_step, tval=-1,
                   patience = 50
        ):
        
        PD_iter = 1
        os.makedirs("./saved_model/" + save_name, exist_ok=True)
        epoches = max_epoches_per_lambda
        prob_in = np.nan

        while PD_iter<=30:
            if PD_iter == 1:
                if init_ckpt_path is not None:
                    self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
                    ckpt_path = init_ckpt_path
                else:
                    self.train_FM_lambda0(epoches = 10000, batch_size_N = 1000, save_every = 2000, lr = 1e-4)

                if para_lambda_init is not None:
                    paralambda = para_lambda_init
                else:
                    paralambda = 0
                    eval_batch_num = 10
                    eval_batch_size = 1000
                    eval_sample_num = eval_batch_num * eval_batch_size
                    total_in_num = 0
                    for i in range(eval_batch_num):
                        samples = self.sampler(eval_batch_size)         
                        valid_mask, in_num = self.get_terminal_feedback(samples, thres = self.thres)
                        total_in_num += in_num

                    prob_in = total_in_num / eval_sample_num
                    paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                    PD_lr = PD_lr_decay * PD_lr
            else:
                paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                PD_lr = PD_lr_decay * PD_lr
                ckpt_path = None

            save_name_PD =  save_name + "/" + save_name + '_' + str(PD_iter) + '_' + self.format_float(paralambda)
            print("Iter:", PD_iter, "lambda:", paralambda, "prob_in:", prob_in)

            paralambda, prob_in = self.train_fixed_lambda(epoches, batch_size_N, save_every = epoches, para_lambda = paralambda,
                                          lr=lr, save_name=save_name_PD,
                         update_pmodel_every = update_pmodel_every, init_ckpt_path = ckpt_path,
                                          FM_step_num = FM_step_num, tval = tval, early_stop = True, patience=patience)
            PD_iter += 1



    def estimate_nll(self, x1_B1HW, n_steps=100, use_rademacher=True, return_mean=True, return_bpd=False):

        if self.model.model_type != "velocity_estimator":
            raise RuntimeError("estimate_nll() is only valid when model_type='velocity_estimator'")
        x = x1_B1HW.clone().to(self.device)
        B = x.shape[0]
        dt = 1.0 / n_steps
    
        f_B = torch.zeros(B, device=x.device, dtype=x.dtype)
        z_B1HW = (2 * torch.randint_like(x, 0, 2) - 1).to(x.dtype) if use_rademacher else torch.randn_like(x)
    
        for k in range(n_steps):
            t_B = torch.full((B,), 1.0 - k * dt, device=x.device, dtype=x.dtype)
            x.requires_grad_(True)
            v_B1HW = self.model(x, t_B)
            v_dot_z = (v_B1HW * z_B1HW).sum()
            
            Jt_z_B1HW = torch.autograd.grad(v_dot_z, x, create_graph=False, retain_graph=False)[0]
            # print(torch.mean(torch.abs(Jt_z_B1HW)))
            div_B = (Jt_z_B1HW * z_B1HW).flatten(1).sum(dim=1)
            # print(div_B)
    
            x = (x - dt * v_B1HW).detach()
            f_B = f_B + dt * div_B.detach()
            # print(torch.mean(torch.abs(dt * div_B.detach())))
        # print(v_dot_z.shape)
        # print(x)
        x0_B1HW = x
        D = x0_B1HW[0].numel()
        log2pi = torch.log(x0_B1HW.new_tensor(2.0 * math.pi))
        logp0_B = -0.5 * (x0_B1HW.flatten(1).pow(2).sum(dim=1) + D * log2pi)
        nll_B = -(logp0_B - f_B)
    
        if return_bpd:
            bpd_B = nll_B / (D * torch.log(x0_B1HW.new_tensor(2.0)))
            if return_mean:
                return nll_B.mean(), bpd_B.mean()
            return nll_B, bpd_B
    
        return nll_B.mean() if return_mean else nll_B




    def pcfm_sampler(
        self,
        batch_size,
        residual_fn,
        thres,
        default_generation_step=configs.default_generation_step,
        newton_steps=1,
        final_newton_steps=10,
        eps=1e-6,
        damping=1e-5,
        guided_interpolation=False,
        penalty_lam=1.0,
        penalty_step_size=1e-3,
        penalty_steps=10,
    ):
        """
        PCFM sampler for image-shaped samples.
    
        Assumptions:
            x shape: (B, 1, configs.dim, configs.dim)
            self.model(x, t_tensor_N) returns z with same shape as x
            residual_fn(x_single) returns differentiable residual for one sample.
    
        residual_fn should return:
            scalar tensor or vector tensor.
    
        The hard constraint is interpreted as:
            residual_fn(x) <= 0
    
        Internally we project only the positive violation:
            h(x) = relu(residual_fn(x))
        so h(x)=0 means constraint satisfied.
        """
    
        device = configs.device
        dim = configs.dim
        dt = 1.0 / default_generation_step
    
        def h_single(x_single):
            """
            x_single shape: (1, dim, dim)
            returns vector residual h(x) >= 0 where h=0 means feasible.
            """
            r = residual_fn(x_single)
    
            if not torch.is_tensor(r):
                raise TypeError(
                    "residual_fn must return a torch.Tensor. "
                    "A NumPy residual cannot be used for PCFM projection."
                )
    
            r = r.reshape(-1)
    
            # Inequality residual: feasible if r <= 0.
            # PCFM wants h(x)=0, so use positive violation.
            return torch.relu(r)
    
        def project_single(xi, max_iter):
            """
            Damped Newton-Schur projection of one sample xi onto h(x)=0.
            """
            x_anchor = xi.detach()
            u = xi.detach().clone()
    
            for _ in range(max_iter):
                u = u.detach().clone().requires_grad_(True)
    
                h_val = h_single(u)
                h_norm = torch.linalg.norm(h_val.detach())
    
                if h_norm <= eps:
                    u = u.detach()
                    break
    
                flat_u = u.reshape(-1)
                rows = []
    
                for j in range(h_val.numel()):
                    grad_j = torch.autograd.grad(
                        h_val[j],
                        u,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
    
                    if grad_j is None:
                        grad_j = torch.zeros_like(u)
    
                    rows.append(grad_j.reshape(-1))
    
                J = torch.stack(rows, dim=0)  # shape: (m, n)
    
                # If all gradients vanish, projection cannot improve this sample.
                if torch.linalg.norm(J.detach()) <= eps:
                    u = u.detach()
                    break
    
                delta = (x_anchor.reshape(-1) - flat_u).unsqueeze(-1)
                JJt = J @ J.T
                rhs = h_val.unsqueeze(-1) + J @ delta
    
                eye = torch.eye(
                    JJt.shape[0],
                    device=JJt.device,
                    dtype=JJt.dtype,
                )
    
                lam = torch.linalg.solve(JJt + damping * eye, rhs)
    
                flat_new = x_anchor.reshape(-1) - (J.T @ lam).squeeze(-1)
                u_new = flat_new.reshape_as(x_anchor)
    
                if torch.linalg.norm((u_new - u).detach()) <= eps:
                    u = u_new.detach()
                    break
    
                u = u_new.detach()
    
            return u.detach()
    
        def project_batch(x, max_iter):
            out = []
            for b in range(x.shape[0]):
                out.append(project_single(x[b], max_iter=max_iter))
            return torch.stack(out, dim=0)
    
        def relaxed_penalty_single(x_hat, z, t_next):
            """
            Optional relaxed PCFM correction:
                min_x ||x - x_hat||^2 + lambda * ||h(x + gamma z)||^2
            """
            gamma = max(1.0 - float(t_next), 1e-6)
            x = x_hat.detach().clone().requires_grad_(True)
    
            for _ in range(penalty_steps):
                x_ext = x + gamma * z.detach()
                h_val = h_single(x_ext)
    
                loss = (
                    (x - x_hat.detach()).pow(2).sum()
                    + penalty_lam * h_val.pow(2).sum()
                )
    
                grad = torch.autograd.grad(
                    loss,
                    x,
                    retain_graph=False,
                    create_graph=False,
                )[0]
    
                x = (x - penalty_step_size * grad).detach().clone().requires_grad_(True)
    
            return x.detach()
    
        def relaxed_penalty_batch(x_hat, z, t_next):
            out = []
            for b in range(x_hat.shape[0]):
                out.append(relaxed_penalty_single(x_hat[b], z[b], t_next))
            return torch.stack(out, dim=0)
    
        # Initial noise u_0
        x0 = torch.randn(
            batch_size,
            1,
            dim,
            dim,
            dtype=torch.float32,
            device=device,
        )
    
        x_prev = x0.clone()
    
        for i in range(default_generation_step):
            t = i / default_generation_step
            t_next = (i + 1) / default_generation_step
    
            t_tensor_N = t * torch.ones(
                x_prev.shape[0],
                device=device,
                dtype=torch.float32,
            )
    
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
    
            # Forward shooting to terminal time:
            # x_1 ~= x_t + (1 - t) v_theta(x_t, t)
            x_terminal = x_prev + (1.0 - t) * z
    
            # Project terminal prediction toward h(x)=0
            x_terminal_proj = project_batch(
                x_terminal,
                max_iter=newton_steps,
            )
    
            # Reverse/interpolate projected terminal sample back to t_next
            x_hat = (1.0 - t_next) * x0 + t_next * x_terminal_proj
    
            if guided_interpolation:
                x_prev = relaxed_penalty_batch(x_hat, z, t_next)
            else:
                x_prev = x_hat.detach()
    
        # Final hard projection
        x_prev = project_batch(
            x_prev,
            max_iter=final_newton_steps,
        )
    
        return x_prev.detach()
    


                    





            