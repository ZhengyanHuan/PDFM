import numpy as np
import torch

####### dataset settings#####

d_model = 8

####### Costraint settings######


####### DDPM settings#######



##########PD_DDPM settings#######


############FM settings#############

FM_lr = 2e-4
FM_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FM_step_num = 100
FM_name = 'OTFM'

############PDFM #############
PDFM_name = 'OTPDFM'
