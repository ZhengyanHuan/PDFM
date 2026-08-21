import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import net
import torch.nn.functional as F
import os
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
    def __init__(self, dataset, default_generation_step = 100, default_thres = 50):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.model = self.get_untrained_model()
        self.model_p = self.get_untrained_model_p()
        self.criteria = nn.MSELoss()
        self.default_generation_step = default_generation_step
        self.dim = 96
        self.default_thres = default_thres

    def get_untrained_model(self):
        return net.FingerprintVelocityNet().to(self.device)

    def get_untrained_model_p(self):
        return net.FingerprintConstraintClassifier().to(self.device)

    def sample_xt_ut_given_x1_x0(self, x0_B1DD: torch.Tensor, x1_B1DD: torch.Tensor, t_B: torch.Tensor):
        return (1 - t_B[..., None, None, None]) * x0_B1DD + t_B[..., None, None, None] * x1_B1DD, x1_B1DD - x0_B1DD

    def train_FM_lambda0(self, lr, save_name, epoches, batch_size_N, save_every,
                         verbose=True, init_path = None):
        if init_path is not None:
            self.model.load_state_dict(torch.load(init_path, weights_only=True))
        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches), disable=not verbose)
        loss_record = np.zeros(epoches)
        for j in pbar:
            x1_B1DD = self.dataset.get_samples( batch_size_N)
            x0_B1DD = torch.randn_like(x1_B1DD, device=self.device, dtype=torch.float32)

            t_B = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_B1DD, ut_B1DD = self.sample_xt_ut_given_x1_x0(x0_B1DD, x1_B1DD,t_B)
            vt_B1DD = self.model(xt_B1DD, t_B)
            flow_loss = self.criteria(ut_B1DD, vt_B1DD)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

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

    def sampler(self, batch_size, default_generation_step=None):
        if default_generation_step is None:
            default_generation_step = self.default_generation_step

        x_prev = torch.randn(batch_size, 1, self.dim, self.dim, dtype=torch.float32, device=self.device)
        for i in range(default_generation_step):
            t = i / default_generation_step
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
            x_prev = x_prev + z / default_generation_step
        return x_prev

    def train_prob_model(self, batch_size, lr, training_epochs, init_ckpt_path = None,
                         inner_batch_size=None, inner_repeat_num=None, save_name = None,
                         verbose = True, FM_step_num = None, t_val = -1):

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

            valid_mask = self.check_inside(sampled_trajectories_TB1DD[-1, :], thres=self.default_thres)

            valid_mask_expand_TB = valid_mask.unsqueeze(0).expand(sampled_trajectories_TB1DD.shape[0], -1)

            t_idx = self.time_index(sampled_trajectories_TB1DD).flatten(0, 1)  # (T*B)
            # print(t_idx.dtype)
            sampled_trajectories_TB1DD_flatten = sampled_trajectories_TB1DD.flatten(0, 1)  # (T*B, 1, D, D)
            # print(sampled_trajectories_TB1DD_flatten.dtype)
            valid_mask_expand_TB_flatten = valid_mask_expand_TB.flatten(0, 1)
            # print(valid_mask_expand_TB_flatten.dtype)

            t_mask = t_idx > t_val
            t_idx = t_idx[t_mask]
            sampled_trajectories_TB1DD_flatten = sampled_trajectories_TB1DD_flatten[t_mask]
            valid_mask_expand_TB_flatten = valid_mask_expand_TB_flatten[t_mask]

            # loss_sum = 0
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
                # loss_sum += loss.item()

                # loss_avg = loss_sum / inner_repeat_num
                loss_record[j] = loss.item()
                j += 1
                pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                })

        if save_name is not None:
            torch.save(self.model_p.state_dict(), './saved_model/' + save_name)
            # np.save('./saved_model/' + self.PD_DDPM_name + '_loss_record.npy', DDPM_loss_record)
            np.save('./saved_model/' + save_name.replace('.pth', '_') + 'loss_record.npy', loss_record)

        in_num = torch.sum(valid_mask).item()
        return valid_mask, in_num

    def sample_recordsteps(self, batch_size, FM_step_num=None):
        if FM_step_num is None:
            FM_step_num = self.default_generation_step

        output = torch.randn(FM_step_num+1, batch_size, 1, self.dim, self.dim, dtype=torch.float32, device=self.device)
        x_prev = torch.randn(batch_size, 1, self.dim, self.dim, dtype=torch.float32, device=self.device)

        output[0] = x_prev

        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                z = self.model(x_prev, t_tensor_N)
            x_prev = x_prev + z * 1 / FM_step_num
            output[i+1] = x_prev

        return output


    def check_inside(self, cur_state_B1DD, thres = None):
        if thres is None:
            thres = self.default_thres

        dataNDD_normed = cur_state_B1DD.cpu().numpy()[:, 0, :, :]
        dataNDD = self.dataset.denorm(dataNDD_normed)
        feasibles, counts = self.dataset.fingerprint_connected_component_oracle(dataNDD, max_components=thres, ridge_dark=True)
        return torch.tensor(feasibles, device=self.device, dtype=torch.float32)

    def get_terminal_feedback(self, cur_state_B1DD, thres):
        if thres is None:
            thres = self.default_thres

        feasibles = self.check_inside(cur_state_B1DD, thres)
        in_num = torch.sum(feasibles).item()
        return feasibles, in_num

    def time_index(self, x, FM_step_num =  None):
        if FM_step_num is None:
            FM_step_num = self.default_generation_step
        # x: TB1DD
        T, b = x.shape[0], x.shape[1]
        t_idx = torch.arange(T, device=x.device).unsqueeze(1).expand(T, b)/FM_step_num
        return t_idx

    def format_float(self, x: float):
        s = str(x)
        if "." in s:
            left, right = s.split(".", 1)
            return f"{left}p{right[:2]}"
        return s[:3]



    def train_fixed_lambda(self, epoches, batch_size_N, save_every, para_lambda,lr, save_name,
                         update_pmodel_every = 10, init_ckpt_path = None, FM_step_num = None, tval = -1,
                            early_stop = False, patience = 50, retrain_pmodel = False):
        if FM_step_num is None:
            FM_step_num = self.default_generation_step
        if early_stop:
            stopper_flow = StableStopper(patience)
            stopper_constraint = StableStopper(patience)

        if init_ckpt_path is not None and not retrain_pmodel:
            self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
            folder, file = os.path.split(init_ckpt_path)
            # print(folder + 'pmodel4'+file)

            if os.path.exists(folder + '/pmodel4' + file):
                print('pmodel found')
                self.model_p.load_state_dict(torch.load('./saved_model/' + 'pmodel4' + file, weights_only=True))
            else:
                print("initializing pmodel")
                self.train_prob_model(batch_size=1000, lr=1e-4, training_epochs=50, verbose=True,
                                      save_name='pmodel4' + file, t_val=tval)
        elif retrain_pmodel:
            print("initializing pmodel")
            # print("initializing pmodel")
            self.train_prob_model(batch_size=1000, lr=1e-4, training_epochs=50, verbose=False)
        else:
            print("using the current pmodel without retraining")

        print("training PDFM")

        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches))

        flow_loss_record = np.array([])
        constraint_loss_record = np.array([])
        prob_in_record = np.array([])

        for j in pbar:
            x1_B1DD = self.dataset.get_samples(batch_size_N)
            x0_B1DD = torch.randn_like(x1_B1DD, device=self.device, dtype=torch.float32)

            t_B = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)
            xt_B1DD, ut_B1DD = self.sample_xt_ut_given_x1_x0(x0_B1DD, x1_B1DD,t_B)

            vt_B1DD = self.model(xt_B1DD, t_B)
            flow_loss = self.criteria(ut_B1DD, vt_B1DD)

            end_mask = t_B > tval
            feedback = self.model_p.predict_proba(xt_B1DD + vt_B1DD / FM_step_num, t_B + 1 / FM_step_num) * end_mask
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
                prob_in_record = np.append(prob_in_record, (in_num / pmodel_inner_batch_size))


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
                        valid_mask, in_num = self.get_terminal_feedback(samples, thres = self.default_thres)
                        total_in_num += in_num
                    print('Early stopping, lambda=' + str(para_lambda)+' feasible rate:'+ str(total_in_num/eval_sample_num))
                    return para_lambda, total_in_num/eval_sample_num

            if (j + 1) % save_every == 0:
                # print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')
                np.savez('./saved_model/' + save_name + '_record.npz', flow_loss_record=flow_loss_record,
                         constraint_loss_record=constraint_loss_record, prob_in_record=prob_in_record)
        return para_lambda, 0

    def PDFMtrain(self, max_epoches_per_lambda, batch_size_N, target_csrate, PD_lr, lr,
                  save_name,
                  FM_step_num=None,
                  para_lambda_init = None, PD_lr_decay = 0.8,
                  update_pmodel_every=10, init_ckpt_path=None,  tval=-1,
                   patience = 50
        ):
        if FM_step_num is None:
            FM_step_num = self.default_generation_step

        PD_iter = 1
        os.makedirs("./saved_model/" + save_name, exist_ok=True)
        epoches = max_epoches_per_lambda
        prob_in = np.nan


        while PD_iter<=20:
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
                        valid_mask, in_num = self.get_terminal_feedback(samples, thres = self.default_thres)
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


    def estimate_nll(
        self,
        x1_B1HW,
        n_steps=100,
        use_rademacher=True,
        return_mean=True,
        return_bpd=False,
    ):

        x = x1_B1HW.clone().to(self.device)
        B = x.shape[0]
        dt = 1.0 / n_steps

        f_B = torch.zeros(B, device=x.device, dtype=x.dtype)

        z_B1HW = (
            (2 * torch.randint_like(x, 0, 2) - 1).to(x.dtype)
            if use_rademacher
            else torch.randn_like(x)
        )

        for k in range(n_steps):
            t_B = torch.full(
                (B,),
                1.0 - k * dt,
                device=x.device,
                dtype=x.dtype,
            )

            x.requires_grad_(True)
            v_B1HW = self.model(x, t_B)

            v_dot_z = (v_B1HW * z_B1HW).sum()

            Jt_z_B1HW = torch.autograd.grad(
                v_dot_z,
                x,
                create_graph=False,
                retain_graph=False,
            )[0]

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

        logp0_B = -0.5 * (
            x0_B1HW.flatten(1).pow(2).sum(dim=1) + D * log2pi
        )

        nll_B = -(logp0_B - f_B)

        if return_bpd:
            bpd_B = nll_B / (D * torch.log(x0_B1HW.new_tensor(2.0)))

            if return_mean:
                return nll_B.mean(), bpd_B.mean()

            return nll_B, bpd_B

        return nll_B.mean()




