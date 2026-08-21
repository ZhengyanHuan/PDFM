import numpy as np
import torch
import qm9.visualizer as visualizer
import qm9.analyze as analyze


class qm9dataset():
    def __init__(self, subset='train', data_path='./downloaded_data/qm9/'
                 , device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_nodes=29):
        complete_path = data_path + subset + '_processed.npz'
        data_npz = np.load(complete_path)
        self.max_nodes = max_nodes
        # self.xh = data_npz['xh']

        self.num_atoms = data_npz['num_atoms']
        self.node_mask, self.edge_mask = self.build_masks_from_num_atoms(self.num_atoms)
        self.xh = self.center_pos(data_npz['xh'], self.node_mask)
        self.data_size = self.xh.shape[0]
        self.device = device

        self.xh = torch.tensor(self.xh, device=self.device)
        self.node_mask = torch.tensor(self.node_mask, device=self.device)
        self.edge_mask = torch.tensor(self.edge_mask, device=self.device)
        self.num_atoms = torch.tensor(self.num_atoms, device=self.device)
        self.dataset_info = {
            'name': 'qm9',
            'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
            'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
            'colors_dic': ['#FFFFFF', '#909090', '#3050F8', '#FF0D0D', '#90E050'],
            'radius_dic': [0.46, 0.77, 0.75, 0.73, 0.71],
        }

    def build_masks_from_num_atoms(self, num_atoms, max_nodes=None):
        """
        Rebuild masks when needed.

        Args:
            num_atoms: [B], numpy array / list / torch tensor
            max_nodes: int

        Returns:
            node_mask: [B, max_nodes, 1]
            edge_mask: [B, max_nodes, max_nodes, 1]
            same backend as num_atoms
        """
        if max_nodes is None:
            max_nodes = self.max_nodes

        if isinstance(num_atoms, torch.Tensor):
            node_mask = (torch.arange(max_nodes, device=num_atoms.device)[None, :] < num_atoms[:, None]).float()
            node_mask = node_mask.unsqueeze(-1)  # [B, max_nodes, 1]
            edge_mask = node_mask[:, :, None, :] * node_mask[:, None, :, :]  # [B, max_nodes, max_nodes, 1]
            return node_mask, edge_mask
        else:
            num_atoms = np.asarray(num_atoms, dtype=np.int64)
            node_mask = (np.arange(max_nodes)[None, :] < num_atoms[:, None]).astype(np.float32)
            node_mask = node_mask[..., None]  # [B, max_nodes, 1]
            edge_mask = node_mask[:, :, None, :] * node_mask[:, None, :, :]  # [B, max_nodes, max_nodes, 1]
            return node_mask, edge_mask


    def center_pos(self, xh, node_mask):
        if isinstance(xh, torch.Tensor):
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)
            x = xh[:, :, :3] * node_mask
            h = xh[:, :, 3:]
            mean = x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            x = (x - mean) * node_mask
            return torch.cat([x, h], dim=-1)
        else:
            if node_mask.ndim == 2:
                node_mask = node_mask[..., None]
            x = xh[:, :, :3] * node_mask
            h = xh[:, :, 3:]
            mean = x.sum(axis=1, keepdims=True) / np.clip(node_mask.sum(axis=1, keepdims=True), 1.0, None)
            x = (x - mean) * node_mask
            return np.concatenate([x, h], axis=-1)

    def get_samples(self, batch_size_N):
        idx = torch.randint(0, self.data_size, (batch_size_N,), device=self.device)
        return self.xh[idx], self.num_atoms[idx], self.node_mask[idx], self.edge_mask[idx]

    def xh_to_positions_charges(self, xh, num_atoms=None):
        pos = xh[:, :, :3]
        atom_idx = xh[:, :, 3:8].argmax(dim=-1) if isinstance(xh, torch.Tensor) else np.argmax(xh[:, :, 3:8], axis=-1)

        if isinstance(xh, torch.Tensor):
            idx_to_charge = torch.tensor([1, 6, 7, 8, 9], device=xh.device, dtype=torch.long)
            charges = idx_to_charge[atom_idx]
            if num_atoms is None:
                num_atoms = (xh[:, :, 3:8].abs().sum(dim=-1) > 0).sum(dim=-1)
            return [pos[i, :int(num_atoms[i])] for i in range(xh.shape[0])], \
                [charges[i, :int(num_atoms[i])] for i in range(xh.shape[0])]
        else:
            idx_to_charge = np.array([1, 6, 7, 8, 9], dtype=np.int64)
            charges = idx_to_charge[atom_idx]
            if num_atoms is None:
                num_atoms = (np.abs(xh[:, :, 3:8]).sum(axis=-1) > 0).sum(axis=-1)
            return [pos[i, :int(num_atoms[i])] for i in range(xh.shape[0])], \
                [charges[i, :int(num_atoms[i])] for i in range(xh.shape[0])]

    def viz(self, positions, atom_type):
        visualizer.plot_data3d(positions, atom_type, dataset_info=self.dataset_info, spheres_3d=True
                               )

    def sample_num_atoms(self, batch_size):
        idx = np.random.randint(0, len(self.num_atoms), size=batch_size)
        return self.num_atoms[idx]

    def viz_xh(self, xh_BD8, num_atoms=None):
        positions_list, charges_list = self.xh_to_positions_charges(xh_BD8, num_atoms)

        charge_to_atom_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        for positions, charges in zip(positions_list, charges_list):
            if isinstance(charges, torch.Tensor):
                atom_type = torch.tensor(
                    [charge_to_atom_type[int(z)] for z in charges],
                    device=charges.device
                ).cpu().numpy()
            else:
                atom_type = np.array([charge_to_atom_type[int(z)] for z in charges], dtype=np.int64)

            if isinstance(positions, torch.Tensor):
                positions = positions.cpu()

            self.viz(positions, atom_type)

    def check_molecule_stable_batch(self, xh_BD8, num_atoms_B=None, debug = False):

        if isinstance(xh_BD8, np.ndarray):
            xh_BD8 = torch.from_numpy(xh_BD8)
        if num_atoms_B is not None and isinstance(num_atoms_B, np.ndarray):
            num_atoms_B = torch.from_numpy(num_atoms_B)

        positions_list, charges_list = self.xh_to_positions_charges(xh_BD8, num_atoms_B)
        charge_to_atom_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        out = []
        out2 = []
        for positions, charges in zip(positions_list, charges_list):
            if isinstance(positions, torch.Tensor):
                positions = positions.detach().cpu().numpy()
            if isinstance(charges, torch.Tensor):
                charges = charges.detach().cpu().numpy()

            atom_type = np.array([charge_to_atom_type[int(z)] for z in charges], dtype=np.int64)
            molecule_stable, nr_stable_bonds, lenx = analyze.check_stability(
                positions=positions,
                atom_type=atom_type,
                dataset_info=self.dataset_info,
                debug=debug,
            )
            out.append(bool(molecule_stable)) #bool
            out2.append(nr_stable_bonds/lenx) #propotion
            if debug:
                print(nr_stable_bonds/lenx)

        return torch.tensor(out, dtype=torch.bool, device=self.device), torch.tensor(out2, dtype=torch.float32, device=self.device)