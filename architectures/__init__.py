# INIT FILE
import sys
sys.path.insert(1,'../')

from .ae import main as train_ae 
from .aae import main as train_aae
from .dae import main as train_dae
from .vae import main as train_vae
from .ganomaly import main as train_ganomaly

from .unet import main as train_unet
from .rnet import main as train_rnet
from .resnet import main as train_resnet
