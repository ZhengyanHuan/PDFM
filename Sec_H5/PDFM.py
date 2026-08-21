import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
from egnn.models import EGNN_dynamics_QM9, EGNN_stability_QM9, EGNN_stability_QM9_naive
import torch.nn.functional as F
import os
import classifier
import math

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

    def __init__(self, dataset,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), default_steps=100,
                 classifer_type = 'egnn'): #'egnn' / 'interact'
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.classifer_type = classifer_type
        self.model = self.get_untrained_model()
        self.criteria = nn.MSELoss()
        self.model_p = self.get_untrained_model_prob()
        self.default_steps = default_steps
        

    def get_untrained_model(self):
        #     return EGNN_dynamics_QM9(in_node_nf=5+1, context_node_nf=0, n_dims=3, hidden_nf=128,
        # device=self.device, n_layers=8, attention=False, condition_time=True, mode="egnn_dynamics")
        #     return EGNN_dynamics_QM9(in_node_nf=5+1, context_node_nf=0, n_dims=3, hidden_nf=128,
        # device=self.device, n_layers=6, attention=True, condition_time=True, inv_sublayers=1, mode="egnn_dynamics", normalization_factor=1, tanh=True,
        # )
        return EGNN_dynamics_QM9(in_node_nf=5 + 1, context_node_nf=0, n_dims=3, hidden_nf=192,
                                 device=self.device, n_layers=8, attention=True, condition_time=True,
                                 inv_sublayers=2, mode="egnn_dynamics", normalization_factor=1, tanh=True, time_cond_dim = 64
                                 ).to(self.device)

    def get_untrained_model_prob(self):
        if self.classifer_type == 'egnn':
            return EGNN_stability_QM9(in_node_nf=5 + 1, context_node_nf=0, n_dims=3, hidden_nf=128,
                                     device=self.device, n_layers=8, attention=True, condition_time=True,
                                     inv_sublayers=1, normalization_factor=1, tanh=True, time_cond_dim = 64
                                     ).to(self.device)
        elif self.classifer_type == 'interact':
            return classifier.PairwiseStabilityClassifier().to(self.device)
        else:
            raise NotImplementedError


    def get_init_noise(self, xh_BD8, node_mask_BD1):
        xh0_BD8 = torch.randn_like(xh_BD8).to(self.device)
        xh0_BD8 = xh0_BD8 * node_mask_BD1
        xh0_BD8 = self.dataset.center_pos(xh0_BD8, node_mask_BD1)
        return xh0_BD8

    def get_xt_ut(self, xh0_BD8, xh1_BD8, t_B):
        t_B11 = t_B.view(-1, 1, 1)
        xt_BD8 = (1.0 - t_B11) * xh0_BD8 + t_B11 * xh1_BD8
        ut_BD8 = xh1_BD8 - xh0_BD8
        return xt_BD8, ut_BD8

    def train_FM_lambda0(self, epoches, batch_size_N, save_every, lr, save_name,
                         verbose=True, init_ckpt_path = None):
        if init_ckpt_path is not None:
            ckpt = torch.load(init_ckpt_path, weights_only=True)
            self.model.load_state_dict(ckpt)

        optimizer = Adam(self.model.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(epoches), disable=not verbose)
        loss_record = np.zeros(epoches)
        for j in pbar:
            # self.xh[idx], self.num_atoms[idx], self.node_mask[idx], self.edge_mask[idx]
            xh1_BD8, num_atoms_B, node_mask_BD1, edge_mask_BDD1 = self.dataset.get_samples(batch_size_N)
            xh0_BD8 = self.get_init_noise(xh1_BD8, node_mask_BD1)
            t_B = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)

            xht_BD8, ut_BD8 = self.get_xt_ut(xh0_BD8, xh1_BD8, t_B)
            vt_BD8 = self.model._forward(t_B, xht_BD8, node_mask_BD1, edge_mask_BDD1, context=None)

            flow_loss = ((ut_BD8 - vt_BD8) ** 2 * node_mask_BD1).sum() / (node_mask_BD1.sum() * ut_BD8.shape[-1])

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

    def sampler(self, batch_size, num_atoms = None, default_generation_step=None):

        if default_generation_step is None:
            default_generation_step = self.default_steps
        if num_atoms is None:
            num_atoms = self.dataset.sample_num_atoms(batch_size)
        node_mask_BD1, edge_mask_BDD1 = self.dataset.build_masks_from_num_atoms( num_atoms)
        x_prev_BD8 = self.get_init_noise(
            torch.empty(batch_size, self.dataset.max_nodes, 8, dtype=torch.float32, device=self.device),
            node_mask_BD1
        )

        for i in range(default_generation_step):
            t = i / default_generation_step
            t_tensor_B = t * torch.ones(x_prev_BD8.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
            with torch.no_grad():
                z = self.model._forward(t_tensor_B, x_prev_BD8, node_mask_BD1, edge_mask_BDD1, context=None)
                # print(torch.mean(torch.abs(z)))
            x_prev_BD8 = x_prev_BD8 + z / default_generation_step
        return x_prev_BD8

    def sample_recordsteps(self, batch_size, num_atoms = None, FM_step_num=None):
        if FM_step_num is None:
            FM_step_num = self.default_steps
        if num_atoms is None:
            num_atoms = self.dataset.sample_num_atoms(batch_size)
        output = torch.empty(FM_step_num+1, batch_size, self.dataset.max_nodes, 8, dtype=torch.float32, device=self.device)
        node_mask_BD1, edge_mask_BDD1 = self.dataset.build_masks_from_num_atoms(num_atoms)
        x_prev_BD8 = self.get_init_noise(
            torch.empty(batch_size, self.dataset.max_nodes, 8, dtype=torch.float32, device=self.device),
            node_mask_BD1
        )

        output[0] = x_prev_BD8
        for i in range(FM_step_num):
            t = i / FM_step_num
            t_tensor_B = t * torch.ones(x_prev_BD8.shape[0], device=self.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
            with torch.no_grad():
                z = self.model._forward(t_tensor_B, x_prev_BD8, node_mask_BD1, edge_mask_BDD1, context=None)
                # print(torch.mean(torch.abs(z)))
            x_prev_BD8 = x_prev_BD8 + z / FM_step_num
            output[i+1] = x_prev_BD8

        return output, num_atoms

    def check_inside(self, xh_BD8, num_atoms_B=None):
        ##TODO!!!!!!!!!!!!!!!!!!!
        valid_mask, valid_prop = self.dataset.check_molecule_stable_batch(xh_BD8, num_atoms_B=num_atoms_B)
        return valid_mask

    def time_index(self, x, FM_step_num = None):
        # x: TBD8
        if FM_step_num is None:
            FM_step_num = self.default_steps

        T, b = x.shape[0], x.shape[1]
        t_idx = torch.arange(T, device=x.device).unsqueeze(1).expand(T, b)/FM_step_num
        return t_idx

    def train_prob_model(self, batch_size, lr, training_epochs, init_ckpt_path = None,
                         inner_batch_size=None, inner_repeat_num=None, save_name = None,
                         verbose = True, FM_step_num = None, t_val = -1):
        # inner_batch_size: number of sampled trajectories
        # batch_size: training batch_size
        if FM_step_num is None:
            FM_step_num = self.default_steps
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
            inner_repeat_num = 150
        self.model_p.train()

        loss_record = np.zeros(training_epochs*inner_repeat_num)
        j = 0
        pbar = tqdm.tqdm(range(training_epochs), disable=not verbose)
        for ep in pbar:
            sampled_trajectories_TBD8, num_atoms = self.sample_recordsteps(inner_batch_size, FM_step_num=FM_step_num)
            #Note T = FM_steps+1
            valid_mask_B = self.check_inside(sampled_trajectories_TBD8[-1,:], num_atoms_B=num_atoms)

            valid_mask_expand_TB = valid_mask_B.unsqueeze(0).expand(sampled_trajectories_TBD8.shape[0], -1)

            num_atoms_expand_TB = num_atoms.unsqueeze(0).expand(sampled_trajectories_TBD8.shape[0], -1)
            num_atoms_expand_TB_flatten = num_atoms_expand_TB.flatten(0, 1)

            t_idx = self.time_index(sampled_trajectories_TBD8, FM_step_num = FM_step_num).flatten(0,1) #(T*B)
            sampled_trajectories_TBD8_flatten = sampled_trajectories_TBD8.flatten(0, 1)  # (T*B, D, 8)
            valid_mask_expand_TB_flatten = valid_mask_expand_TB.flatten(0, 1)

            t_mask = t_idx>t_val
            t_idx = t_idx[t_mask]
            sampled_trajectories_TBD8_flatten = sampled_trajectories_TBD8_flatten[t_mask]
            valid_mask_expand_TB_flatten = valid_mask_expand_TB_flatten[t_mask]

            loss_sum = 0
            for i in range(inner_repeat_num):
                idx = torch.randint(0, t_idx.shape[0], (batch_size,), device=self.device)
                xBD8 = sampled_trajectories_TBD8_flatten[idx]
                yB = valid_mask_expand_TB_flatten[idx]
                # print(yB)
                tB = t_idx[idx]

                node_mask, edge_mask = self.dataset.build_masks_from_num_atoms(num_atoms_expand_TB_flatten[idx])

                logits = self.model_p(tB, xBD8, node_mask, edge_mask, context=None)
                loss = F.binary_cross_entropy_with_logits(logits, yB.float())

                self.optimizer_p.zero_grad()
                loss.backward()
                self.optimizer_p.step()

                loss_sum += loss.item()

                # loss_avg = loss_sum / inner_repeat_num
                loss_record[j] = loss.item()
                j += 1
                pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                })
            # pbar.set_description(f"Epoch {ep + 1}/{training_epochs}")
            # pbar.set_postfix({
            #     "loss": f"{loss_avg:.4f}",
            # })

        if save_name is not None:
            torch.save(self.model_p.state_dict(), './saved_model/' + save_name)
            # np.save('./saved_model/' + self.PD_DDPM_name + '_loss_record.npy', DDPM_loss_record)
            np.save('./saved_model/' + save_name.replace('.pth', '_') + 'loss_record.npy', loss_record)

        in_num = torch.sum(valid_mask_B).item()
        return in_num

    def train_fixed_lambda(self, epoches, batch_size_N, save_every, para_lambda,lr, save_name,
                         update_pmodel_every = 10, init_ckpt_path = None, FM_step_num = None, tval = -1,
                            early_stop = False, patience = 50, retrain_pmodel = False):

        if FM_step_num is None:
            FM_step_num = self.default_steps

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
            xh1_BD8, num_atoms_B, node_mask_BD1, edge_mask_BDD1 = self.dataset.get_samples(batch_size_N)
            xh0_BD8 = self.get_init_noise(xh1_BD8, node_mask_BD1)
            t_B = torch.rand(batch_size_N, dtype=torch.float32, device=self.device)

            xht_BD8, ut_BD8 = self.get_xt_ut(xh0_BD8, xh1_BD8, t_B)
            vt_BD8 = self.model._forward(t_B, xht_BD8, node_mask_BD1, edge_mask_BDD1, context=None)

            flow_loss = ((ut_BD8 - vt_BD8) ** 2 * node_mask_BD1).sum() / (node_mask_BD1.sum() * ut_BD8.shape[-1])

            end_mask = t_B > tval
            feedback = self.model_p.predict_proba(t_B + 1 / FM_step_num, xht_BD8 + vt_BD8 / FM_step_num,  node_mask_BD1, edge_mask_BDD1) * end_mask
            constraint_loss = -feedback.mean()

            loss = constraint_loss * para_lambda + flow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pmodel_inner_batch_size = 300
            if (j + 1) % update_pmodel_every == 0:
                in_num = self.train_prob_model(batch_size=300, lr=5e-5, training_epochs=1, verbose=False,
                                                          inner_batch_size=pmodel_inner_batch_size, inner_repeat_num=50)
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
                        valid_mask = self.check_inside(samples)
                        in_num = torch.sum(valid_mask).item()
                        total_in_num += in_num
                    print('Early stopping, lambda=' + str(para_lambda)+' feasible rate:'+ str(total_in_num/eval_sample_num))
                    return para_lambda, total_in_num/eval_sample_num

            if (j + 1) % save_every == 0:
                # print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(self.model.state_dict(), './saved_model/' + save_name + '_' + str(j + 1) + '.pth')
                np.savez('./saved_model/' + save_name + '_record.npz', flow_loss_record=flow_loss_record,
                         constraint_loss_record=constraint_loss_record, prob_in_record=prob_in_record)
        return para_lambda, 0

    def format_float(self, x: float):
        s = str(x)
        if "." in s:
            left, right = s.split(".", 1)
            return f"{left}p{right[:2]}"
        return s[:3]

    def PDFMtrain(self, max_epoches_per_lambda, batch_size_N, target_csrate, PD_lr, lr, save_name,
                  para_lambda_init=None,
                  PD_lr_decay=0.8,
                  update_pmodel_every=10, init_ckpt_path=None, FM_step_num=None, tval=-1,
                  patience=50
                  ):

        if FM_step_num is None:
            FM_step_num = self.default_steps
        PD_iter = 1
        os.makedirs("./saved_model/" + save_name, exist_ok=True)
        epoches = max_epoches_per_lambda
        prob_in = np.nan

        while PD_iter<=300:
            if PD_iter == 1:
                if init_ckpt_path is not None:
                    self.model.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
                    ckpt_path = init_ckpt_path
                else:
                    self.train_FM_lambda0(epoches = 50000, batch_size_N = 500, save_every = 5000, lr = 4e-5, save_name = "./"+save_name+"/"+save_name, init_ckpt_path = None)

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
                        valid_mask = self.check_inside(samples)
                        total_in_num += torch.sum(valid_mask).item()

                    prob_in = total_in_num / eval_sample_num
                    paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                    PD_lr = PD_lr_decay * PD_lr

            else:
                paralambda = max(paralambda + PD_lr * (target_csrate - prob_in), 0)
                PD_lr = PD_lr_decay * PD_lr
                ckpt_path = None

            save_name_PD =  save_name + "/" + save_name + '_' + str(PD_iter) + '_' + self.format_float(paralambda)
            print("Iter:", PD_iter, "lambda:", paralambda, "prob_in:", prob_in)

            paralambda, prob_in = self.train_fixed_lambda(epoches, batch_size_N, save_every=epoches, para_lambda=paralambda,
                                                          lr=lr, save_name=save_name_PD,
                                                          update_pmodel_every=update_pmodel_every, init_ckpt_path=ckpt_path,
                                                          FM_step_num=FM_step_num, tval=tval, early_stop=True,
                                                          patience=patience)
            PD_iter += 1



    def estimate_nll(
        self,
        traindata,
        batch_size,
        n_steps=None,
        use_rademacher=True,
        return_mean=True,
        return_bpd=False,
    ):
        if n_steps is None:
            n_steps = self.default_steps
    
        # Draw real samples and their corresponding graph information.
        x1_BD8, num_atoms_B, node_mask_BD1, edge_mask_BDD1 = (
            traindata.get_samples(batch_size)
        )
    
        x = x1_BD8.clone().to(self.device)
        num_atoms_B = num_atoms_B.to(self.device)
        node_mask_BD1 = node_mask_BD1.to(
            device=self.device,
            dtype=x.dtype,
        )
        edge_mask_BDD1 = edge_mask_BDD1.to(
            device=self.device,
            dtype=x.dtype,
        )
    
        B, D, F = x.shape
        dt = 1.0 / n_steps
    
        # Ensure padded nodes remain zero.
        x = x * node_mask_BD1
    
        f_B = torch.zeros(
            B,
            device=x.device,
            dtype=x.dtype,
        )
    
        # Hutchinson trace-estimation vector.
        if use_rademacher:
            z_BD8 = (
                2 * torch.randint_like(x, low=0, high=2) - 1
            ).to(x.dtype)
        else:
            z_BD8 = torch.randn_like(x)
    
        # Ignore padded nodes in the divergence estimate.
        z_BD8 = z_BD8 * node_mask_BD1
    
        # Integrate backward from data at t=1 to noise at t=0.
        for k in range(n_steps):
            t = 1.0 - k * dt
    
            t_tensor_B = torch.full(
                (B,),
                t,
                device=x.device,
                dtype=x.dtype,
            )
    
            x.requires_grad_(True)
    
            v_BD8 = self.model._forward(
                t_tensor_B,
                x,
                node_mask_BD1,
                edge_mask_BDD1,
                context=None,
            )
    
            v_BD8 = v_BD8 * node_mask_BD1
    
            v_dot_z = (v_BD8 * z_BD8).sum()
    
            Jt_z_BD8 = torch.autograd.grad(
                v_dot_z,
                x,
                create_graph=False,
                retain_graph=False,
            )[0]
    
            div_B = (
                Jt_z_BD8
                * z_BD8
                * node_mask_BD1
            ).flatten(1).sum(dim=1)
    
            x = (x - dt * v_BD8).detach()
            x = x * node_mask_BD1
    
            f_B = f_B + dt * div_B.detach()
    
        x0_BD8 = x
    
        # Each active atom has F=8 scalar dimensions.
        dimensionality_B = num_atoms_B.to(x.dtype) * F
    
        log2pi = torch.log(
            x0_BD8.new_tensor(2.0 * math.pi)
        )
    
        squared_norm_B = (
            x0_BD8.pow(2) * node_mask_BD1
        ).flatten(1).sum(dim=1)
    
        # Standard Gaussian log density over active dimensions.
        logp0_B = -0.5 * (
            squared_norm_B
            + dimensionality_B * log2pi
        )
    
        logp1_B = logp0_B - f_B
        nll_B = -logp1_B/(num_atoms_B*8)
    
        if return_bpd:
            bpd_B = nll_B / (
                dimensionality_B
                * torch.log(x0_BD8.new_tensor(2.0))
            )
    
            if return_mean:
                return nll_B.mean(), bpd_B.mean()
    
            return nll_B, bpd_B
    
        if return_mean:
            return nll_B.mean()
    
        return nll_B
                

