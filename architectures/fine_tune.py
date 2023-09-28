import tensorflow as tf
import numpy as np
import time
from models import (Autoencoder,
                   MLP,
                   UNET)

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training,
                             save_training_curves)

from utils.training import print_epoch,save_checkpoint
from utils.metrics import nln, get_nln_errors
from model_config import *
from inference import infer

from .helper import end_routine

mlp_optimizer = tf.keras.optimizers.Adam()

@tf.function
def fine_tune_step(mlp, x, y, error):# xn):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as ft_tape:

        p = mlp(error,training=True)
        loss = bce(p,y)

    gradients_of_mlp = ft_tape.gradient(loss, mlp.trainable_variables)
    mlp_optimizer.apply_gradients(zip(gradients_of_mlp, mlp.trainable_variables))
    return loss

def fine_tune(ae, mlp, ae_train_data, train_images, train_masks, args):
    train_mask_dataset = tf.data.Dataset.from_tensor_slices(train_masks.astype('float32')).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    train_data_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    fine_tune_loss = []
    z = infer(ae.encoder, ae_train_data, args, 'encoder')
    x_hat_train  = infer(ae, ae_train_data, args, 'AE')
    z_query = infer(ae.encoder, train_images, args, 'encoder')
    neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, 
                                                                        z_query, 
                                                                        x_hat_train, 
                                                                        args.algorithm, 
                                                                        10,
                                                                        -1)
    nln_error = get_nln_errors([ae],
                               'AE',
                               z_query,
                               z,
                               train_images,
                               x_hat_train,
                               neighbours_idx,
                               neighbour_mask,
                               args)
    error_dataset = tf.data.Dataset.from_tensor_slices(nln_error).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch, mask_batch, error_batch in zip(train_data_dataset, train_mask_dataset, error_dataset):
            ft_loss =  fine_tune_step(mlp, image_batch.numpy(), mask_batch, error_batch)

        generate_and_save_images(mlp,
                                 epoch + 1,
                                 error_batch.numpy()[:25,...],
                                 'fine_tune',
                                 args)
        save_checkpoint(mlp,epoch,args,'fine_tune','mlp')

        fine_tune_loss.append(ft_loss)


        print_epoch('fine_tune',
                     epoch,
                     time.time()-start,
                     {'ft loss':ft_loss.numpy()},
                     None)

    generate_and_save_training([fine_tune_loss],
                                ['ft loss'],
                                'fine_tune',args)

    return mlp

def main(train_dataset, ae_train_data, train_images,train_labels,train_masks,test_images,test_labels, test_masks,test_masks_orig, args):

    #indx = np.random.randint(0,len(ae_train_data), 50000)
    rng = np.random.default_rng()
    indx = rng.choice(len(ae_train_data), size=int(8e5), replace=False)
    ae_train_data = ae_train_data[indx]


   # indx = np.random.randint(0,len(train_images), int(1e6))
    indx = rng.choice(len(train_images), size=int(8e5), replace=False)
    train_images = train_images[indx]
    train_labels = train_labels[indx]
    train_masks = train_masks[indx]

    mlp = MLP(args)
    ae = Autoencoder(args)
    ae.load_weights('outputs/DAE_disc/rfi/temp/training_checkpoints/checkpoint_full_model_ae')

    mlp = fine_tune(ae, mlp, ae_train_data, train_images, train_masks, args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [ae,mlp], 'fine_tune', args)
    
if __name__  == '__main__':
    main()
