import tensorflow as tf
import numpy as np
import time
from models import (Encoder, 
                    Autoencoder, 
                    Discriminator_x)
from models_mvtec import Encoder as Encoder_MVTEC
from models_mvtec import Autoencoder as Autoencoder_MVTEC 
from models_mvtec import Discriminator_x as Discriminator_x_MVTEC

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)
from utils.training import print_epoch,save_checkpoint
from model_config import *

from .helper import end_routine

ae_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
encoder_optimizer = tf.keras.optimizers.Adam(1e-4)

def ae_loss(x,x_hat,loss_weight):
    return loss_weight*mse(x,x_hat)

def discriminator_loss(real_output, fake_output,loss_weight):
    return loss_weight*mse(real_output, fake_output)

def encoder_loss(z,z_hat, loss_weight):
    return loss_weight*mse(z,z_hat)

@tf.function
def train_step(ae,encoder,discriminator,images):

    with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as en_tape:

      x_hat  = ae(images,training=True)  
      z = ae.encoder(images)
      z_hat = encoder(x_hat)

      real_output,c0 = discriminator(images, training=True)
      fake_output,c1 = discriminator(x_hat, training=True)

      auto_loss = ae_loss(images,x_hat,1)
      disc_loss = discriminator_loss(real_output, fake_output,50)
      enc_loss = encoder_loss(z,z_hat,1)

    gradients_of_ae= ae_tape.gradient(auto_loss, ae.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_encoder= en_tape.gradient(enc_loss, encoder.trainable_variables)

    ae_optimizer.apply_gradients(zip(gradients_of_ae, ae.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    return auto_loss, disc_loss,enc_loss

def train(ae,encoder,discriminator,dataset,test_images,test_labels, args):
    ae_loss, d_loss,e_loss,aucs = [],[],[],[]
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in dataset:
           auto_loss, disc_loss, encoder_loss =  train_step(ae,
                                                            encoder,
                                                            discriminator,
                                                            image_batch)

        generate_and_save_images(ae,epoch + 1,image_batch[:25,...],'GANomaly',args)

        save_checkpoint(ae, epoch, args,'GANomaly','ae')
        save_checkpoint(discriminator, epoch, args, 'GANomaly','disc')
        save_checkpoint(encoder, epoch, args, 'GANomaly', 'encoder')

        ae_loss.append(auto_loss)
        e_loss.append(encoder_loss)
        d_loss.append(disc_loss)

        print_epoch('GANomaly',
                    epoch,
                    time.time()-start,
                    {'AE Loss':auto_loss.numpy(),
                     'Discriminator loss': disc_loss.numpy(),
                     'Encoder loss':encoder_loss.numpy()},
                    None)


    generate_and_save_training([ae_loss,d_loss,e_loss],
                                ['ae loss', 
                                'discriminator loss', 
                                'encoder loss'],
                                'GANomaly',
                                args)

    generate_and_save_images(ae,epoch,image_batch[:25,...],'GANomaly',args)

    return ae, discriminator,encoder

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, args):
    if args.data == 'MVTEC':
        ae = Autoencoder_MVTEC(args)
        discriminator = Discriminator_x_MVTEC(args)
        encoder = tf.keras.Sequential(Encoder_MVTEC(args))
    else:
        ae = Autoencoder(args)
        discriminator = Discriminator_x(args)
        encoder = tf.keras.Sequential(Encoder(args))

    ae,discriminator,encoder = train(ae,
                                     encoder,
                                     discriminator,
                                     train_dataset,
                                     test_images,
                                     test_labels,
                                     args)

    end_routine(train_images, test_images, test_labels, test_masks, [ae,discriminator,encoder], 'GANomaly', args)

    
if __name__  == '__main__':
    main()


