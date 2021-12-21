import tensorflow as tf
import numpy as np
from sklearn import neighbors
from matplotlib import pyplot as plt
import time
from models import Autoencoder

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine
from inference import infer

optimizer = tf.keras.optimizers.Adam()
NNEIGHBOURS= 5

def l2_loss(x,x_hat):
    return bce(x,x_hat)

@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        loss = l2_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
            auto_loss  =  train_step(ae,image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'AE',
                                 args)
        save_checkpoint(ae,epoch, args,'AE','ae')

        ae_loss.append(auto_loss)

        print_epoch('AE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)

    generate_and_save_training([ae_loss],
                                ['ae loss'],
                                'AE',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'AE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,test_masks_orig,args):
    if args.data == 'MVTEC':
        ae = Autoencoder_MVTEC(args)
    else:
        ae = Autoencoder(args)

    ae = train(ae,train_dataset, train_images,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [ae], 'AE', args)

    
if __name__  == '__main__':
    main()
