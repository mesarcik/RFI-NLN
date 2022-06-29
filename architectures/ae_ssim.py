import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                   Decoder, 
                   Autoencoder)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine

optimizer = tf.keras.optimizers.Adam(1e-4)

def ssim_loss(x,x_hat):
    return 1/2 - tf.reduce_mean(tf.image.ssim(x, x_hat, max_val =1.0))/2

@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        loss = ssim_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch in train_dataset:
            auto_loss  =  train_step(ae,image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'AE_SSIM',
                                 args)
        save_checkpoint(ae,epoch, args,'AE_SSIM','ae')

        ae_loss.append(auto_loss)

        print_epoch('AE_SSIM',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)

    generate_and_save_training([ae_loss],
                                ['ae loss'],
                                'AE_SSIM',args)
    generate_and_save_images(ae,epoch,image_batch[:25,...],'AE_SSIM',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,test_masks_orig,args):
    ae = Autoencoder(args)
    ae = train(ae,train_dataset,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [ae], 'AE_SSIM', args)
    end_routine(train_images, test_images, test_labels, test_masks, [ae], 'AE_SSIM', args)

    
if __name__  == '__main__':
    main()
