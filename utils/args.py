import argparse
import os
from utils.data import sizes 
from coolname import generate_slug as new_name
"""
    Pretty self explanatory, gets arguments for training and adds them to config
"""
parser = argparse.ArgumentParser(description='Train generative anomaly detection models')

parser.add_argument('-model',metavar='-m', type=str, default='AE',
                    choices={'UNET','AE', 'VAE', 'DAE', 'DKNN','AE-res', 'AAE'}, help = 'Which model to train and evaluate')
parser.add_argument('-limit',metavar='-l', type=str, default='None',
                    help = 'Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class',metavar='-a', type=str,  default=2,
                    help = 'The labels of the anomalous class')
parser.add_argument('-anomaly_type',metavar='-at', type=str,  default='MISO',
                    choices={'MISO','SIMO'},help = 'The anomaly scheme whether it is MISO or SIMO')
parser.add_argument('-percentage_anomaly', metavar='-p', type=float, default=0,
                    help = 'The percentage of anomalies in the training set')
parser.add_argument('-epochs', metavar='-e', type=int, default=100,
                    help = 'The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2,
                    help = 'The latent dimension size of the AE based models')
parser.add_argument('-alphas', metavar='-alph', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1.0],
                    help = 'The maximum number of neighbours for latent reconstruction')
parser.add_argument('-neighbors', metavar='-n', type=int, nargs='+', default=[2,4,5,6,7,8,9],
                    help = 'The maximum number of neighbours for latent reconstruction')
parser.add_argument('-radius', metavar='-r', type=float, nargs='+', default=[0.1,0.5,1,2,5,10],
                    help = 'The radius of the unit circle for finding neighbours in frNN')
parser.add_argument('-algorithm', metavar='-nn', type=str, choices={"frnn", "knn"}, 
                    default='knn', help = 'The algorithm for calculating neighbours')
parser.add_argument('-data', metavar='-d', type=str, default='MNIST',
                    help = 'The dataset for training and testing the model on')

parser.add_argument('-data_path', metavar='-mp', type=str, default='datasets/MVTecAD/',
                    help = 'Path to MVTecAD training data')

parser.add_argument('-seed', metavar='-s', type=str, 
                    help = 'The random seed used for naming output files')
parser.add_argument('-debug', metavar='-de', type=str, default='0', 
                    choices={'0', '1', '2', '3'}, help = 'TF debug level')
parser.add_argument('-rotate', metavar='-rot', type=bool,default=False, 
                    help = 'Train on rotated augmentations?')
parser.add_argument('-crop', metavar='-cr', type=bool,default=False, 
                    help = 'Train on crops?')
parser.add_argument('-crop_x', metavar='-cx', type=int,
                    help = 'x-dimension of crop')
parser.add_argument('-crop_y', metavar='-cy', type=int,
                    help = 'y-dimension of crop')
parser.add_argument('-patches', metavar='-ptch', type=bool,default=False, 
                    help = 'Train on patches?')
parser.add_argument('-patch_x', metavar='-px', type=int, default=-1,
                    help = 'x-dimension of patchsize ')
parser.add_argument('-patch_y', metavar='-py', type=int, 
                    help = 'y-dimension of patchsize ')
parser.add_argument('-patch_stride_x', metavar='-psx', type=int, 
                    help = 'x-dimension of strides of patches')
parser.add_argument('-patch_stride_y', metavar='-psy', type=int, 
                    help = 'y-dimension of strides of patches')
parser.add_argument('-rfi', metavar='-rfi', type=str, default=None, 
                    help = 'HERA RFI label to exclude from training')
parser.add_argument('-rfi_threshold', metavar='-rfi_threshold', type=float, default=None, 
                    help = 'AOFlagger base threshold')

args = parser.parse_args()
args.model_name = new_name()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.debug

if args.data == 'MNIST' or args.data == 'FASHION_MNIST':
    args.input_shape =(28,28,1)

elif args.data == 'CIFAR10':
    args.input_shape =(32,32,3)

elif args.data == 'MVTEC':
    if (('grid' in args.anomaly_class) or
        ('screw' in args.anomaly_class) or 
        ('zipper' in args.anomaly_class)): 
        args.input_shape =(1024,1024,1)
    else:
        args.input_shape =(1024,1024,3)

elif args.data == 'HERA':
    args.input_shape =(512, 512,1)

elif args.data == 'HIDE':
    args.input_shape =(256, 256,1)

elif args.data == 'LOFAR':
    args.input_shape =(128,64,1)


if args.patches:
    args.input_shape = (args.patch_x,args.patch_y,args.input_shape[-1])

else:
    args.patch_x = args.input_shape[0]
    args.patch_y = args.input_shape[1]

if args.crop:
    args.input_shape = (args.crop_x,args.crop_y,args.input_shape[-1])

if args.limit == 'None':
    args.limit = None
else: 
    args.limit =  int(args.limit)

if ((args.data == 'MNIST') or 
    (args.data == 'CIFAR10') or 
    (args.data == 'FASHION_MNIST')): 

    args.anomaly_class = int(args.anomaly_class)
