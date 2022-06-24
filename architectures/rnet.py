import tensorflow as tf
import numpy as np
from sklearn import neighbors
from matplotlib import pyplot as plt
import time
from models import RNET

from utils.plotting  import  (generate_and_save_images,
                             generate_and_save_training)

from utils.training import print_epoch,save_checkpoint
from model_config import *
from .helper import end_routine
from inference import infer

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, x, y):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x,training=True)
        loss = bce(x_hat,y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(rnet,train_dataset,train_images, train_masks, test_images,test_labels,test_masks,args,verbose=True,save=True):
    rnet_loss= []
    train_mask_dataset = tf.data.Dataset.from_tensor_slices(train_masks.astype('float32')).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    train_data_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
            auto_loss  =  train_step(rnet,image_batch, mask_batch)

        generate_and_save_images(rnet,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'RNET',
                                 args)
        save_checkpoint(rnet,epoch, args,'RNET','rnet')

        rnet_loss.append(auto_loss)

        print_epoch('RNET',epoch,time.time()-start,{'RNET Loss':auto_loss.numpy()},None)

    generate_and_save_training([rnet_loss],
                                ['rnet loss'],
                                'RNET',args)
    generate_and_save_images(rnet,epoch,image_batch[:25,...],'RNET',args)

    return rnet

def main(train_dataset,train_images,train_labels,train_masks,test_images,test_labels, test_masks,test_masks_orig, args):
    rnet= RNET(args)

    rnet = train(rnet,train_dataset, train_images, train_masks,test_images,test_labels,test_masks,args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [rnet], 'RNET', args)

    
if __name__  == '__main__':
    main()
