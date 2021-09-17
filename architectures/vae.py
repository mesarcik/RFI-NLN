import tensorflow as tf
import numpy as np
import time
from models import VAE
from models_mvtec import VAE as VAE_MVTEC
from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)
from utils.training import print_epoch,save_checkpoint
from model_config import  *

from .helper import end_routine

optimizer = tf.keras.optimizers.Adam()

def l2_loss(x,x_hat):
    return mse(tf.keras.backend.flatten(x),
                                  tf.keras.backend.flatten(x_hat))

def KL_loss(logvar, mean):
    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = -0.5*tf.reduce_sum(kl_loss,axis=-1)
    return kl_loss

def compute_loss(model, x):
    mean, logvar = model.encoder(x,vae=True)
    z = model.reparameterize(mean, logvar)
    x_hat = model.decode(z)

    reconstruction_loss = l2_loss(x,x_hat) *32*32
    var_loss = KL_loss(logvar,mean)

    return tf.keras.backend.mean(var_loss + reconstruction_loss),var_loss, reconstruction_loss



@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss,var_loss,reconstruction_loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, var_loss,tf.reduce_mean(reconstruction_loss)

def train(vae,train_dataset,test_images,test_labels,args):
    ae_loss = []
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch in train_dataset:
           auto_loss,var_loss,reconstruction_loss  =  train_step(vae,image_batch)

        generate_and_save_images(vae,epoch + 1,image_batch[:25,...],'VAE', args)

        save_checkpoint(vae,epoch,args,'VAE','ae')

        ae_loss.append(auto_loss)
        print_epoch('VAE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)
    generate_and_save_training([ae_loss],
                                ['ae loss'],
                                'VAE',args)
    generate_and_save_images(vae,args.epochs,image_batch[:25,...],'VAE',args)
    return vae 


def main(train_dataset,train_images,train_labels,test_images, test_labels, test_masks, args):
    if args.data == 'MVTEC':
        vae = VAE_MVTEC(args)
    else:
        vae = VAE(args)
    vae = train(vae,train_dataset,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, [vae], 'VAE', args)

if __name__  == '__main__':
    main()
