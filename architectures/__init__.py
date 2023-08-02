# INIT FILE
import sys
sys.path.insert(1,'../')

from .ae import main as train_ae 
from .ae_ssim import main as train_ae_ssim
from .dae import main as train_dae
from .fine_tune import main as train_fine_tune

from .unet import main as train_unet
from .rnet import main as train_rnet
from .rfi_net import main as train_rfi_net
from .resnet import main as train_resnet

from .helper import end_routine
