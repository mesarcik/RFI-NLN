from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc 
from utils.metrics import *
from inference import infer, get_error
import os
import numpy as np
import pandas as pd

def generate_and_save_training(losses, legend,name,args): #ae_loss,d_loss,e_loss):
    """
        Shows line plot of of the training curves

        losses (list): list of losses 
        legend (list): name of each loss
        name (str): model name
        args (Namespace): arguments from cmd_args

    """
    epochs = [e for e in range(len(losses[0]))]
    fig = plt.figure(figsize=(10,10))
    for loss, label in zip(losses,legend):
        plt.plot(epochs, loss, label=label)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('outputs/{}/{}/{}/loss.png'.format(name,
                                                   args.anomaly_class,
                                                   args.model_name))
    plt.close('all')


def generate_and_save_images(model, epoch, test_input,name,args):
    """
        Shows input vs output plot for AE while trainging
        
        model (tf.keras.Model): model
        epoch (int): current epoch number 
        test_input (np.array): testing images input
        name (str): model name
        args (Namespace): arguments from cmd_args

    """
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    if name== 'VAE' or name == 'VAEGAN':
        mean, logvar = model.encoder(test_input,vae=True)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
    elif name=='BIGAN':
        predictions = test_input
    else:
        predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
      plt.subplot(5, 5, i+1)
      if predictions.shape[-1] == 1:#1 channel only
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)

      if predictions.shape[-1] == 3: #RGB
          plt.imshow(predictions[i,...], vmin=0, vmax=1)
      plt.axis('off')
    
    if not os.path.exists('outputs/{}/{}/{}/epochs/'.format(name,
                                                         args.anomaly_class,
                                                         args.model_name)):

        os.makedirs('outputs/{}/{}/{}/epochs/'.format(name,
                                                      args.anomaly_class,
                                                      args.model_name))

    plt.tight_layout()
    plt.savefig('outputs/{}/{}/{}/epochs/image_at_epoch_{:04d}.png'.format(name,
                                                            args.anomaly_class,
                                                            args.model_name,
                                                            epoch))
    plt.close('all')

def save_training_curves(model,args,test_images,test_labels,name):
    """
        Shows input vs output for each class for AE after training  
        
        model (tf.keras.Model): model
        args (Namespace): arguments from cmd_args
        test_images (np.array): testing images input
        test_labels (np.array): labels testing images input
        name (str): model name

    """
    model_output = infer(model[0], test_images, args, 'AE') 
    error = get_error('AE',test_images, model_output)
    df = pd.DataFrame(columns=['Reconstruction'])

    labels = pd.unique(test_labels)
    fig, ax = plt.subplots(len(labels),3,figsize=(10,20))
    
    ax[0,1].title.set_text('Model Output')
    ax[0,2].title.set_text('Input - Output')

    for i,lbl in enumerate(labels):
        df.loc[lbl] =  error[test_labels==lbl].mean()
        ind = np.where(test_labels==lbl)[0][0]
        if test_images.shape[-1] == 1: #Mag only
            ax[i,0].imshow(test_images[ind,...,0]);
            ax[i,1].imshow(model_output[ind,...,0]);  
            ax[i,2].imshow(test_images[ind,...,0] - model_output[ind,...,0]);
        if test_images.shape[-1] == 3: #RGB
            ax[i,0].imshow(test_images[ind,...],vmin =0, vmax=1);
            ax[i,1].imshow(model_output[ind,...],vmin=0, vmax=1);  
            ax[i,2].imshow(test_images[ind,...] - model_output[ind,...],vmin=0, vmax=1);

        ax[i,0].title.set_text(lbl)
        ax[i,1].title.set_text(error[ind].mean())
    if not os.path.exists('outputs/{}/{}/{}'.format(name,
                                                    args.anomaly_class,
                                                    args.model_name)):

        os.makedirs('outputs/{}/{}/{}'.format(name,
                                              args.anomaly_class,
                                              args.model_name))
    plt.suptitle('Anomaly = {}'.format(args.anomaly_class))
    plt.tight_layout()
    fig.savefig('outputs/{}/{}/{}/io.png'.format(name,
                                                 args.anomaly_class,
                                                 args.model_name))
    plt.close('all')

    error = (error - np.min(error))/(np.max(error) - np.min(error))
    fpr, tpr, _ = roc_curve(test_labels==args.anomaly_class, error)
    a = auc(fpr,tpr)

    plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % a)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/{}/{}/{}/hist.png'.format(name,
                                                   args.anomaly_class,
                                                   args.model_name))

    df.to_csv('outputs/{}/{}/{}/data.csv'.format(name,
                                                args.anomaly_class,
                                                args.model_name))

