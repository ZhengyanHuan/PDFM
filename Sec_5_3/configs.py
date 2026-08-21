import torch

######## PDE settings #############

global_dt = 0.2
global_domain_extent = 64.0
global_cutoff = 5
global_order = 3

dim = 64
horizon_multiplier = 50
dataset_path = './data/ks_dataset.npy'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Ln = "L1" #L1, L2, Linf
default_thres = 0.033 # 0.033 0.05

#####

FM_lr = 1e-3
FM_name = "PDE_FM"
default_generation_step = 100


###
PDFM_name = "PDFM"