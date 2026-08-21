import numpy as np
import torch

####### dataset settings#####

box_bound = 4 # box is [-box_bound, box_bound] x [-box_bound, box_bound]
box2_bound = 2
box2_center = 3

dataset_type = "uniform2" #uniform2

####### Costraint settings######

constraint_type = "box" ### ring box boxandline
constraint_bound = 3.5

ring_radius_lower = 0
ring_radius_upper = ring_radius_lower + 1.5

xysum = 8
####### DDPM settings#######

T = 100
beta_start = 1e-3
beta_end = 2e-1 #2e-2
DDPM_lr = 2e-4
DDPM_training_epochs = 50
DDPM_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DDPM_name = 'DDPM_s100'


##########PD_DDPM settings#######
PD_DDPM_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PD_DDPM_name = 'PD_DDPM'
DDIM_steps = 50


############FM settings#############

FM_lr = 2e-4
FM_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FM_step_num = 100
FM_name = 'OTFM'

############PDFM #############
PDFM_name = 'OTPDFM'
