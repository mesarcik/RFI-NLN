from models import *
import tensorflow as tf
from model_config import BATCH_SIZE 


def infer(model, data, args, arch):
    """
        Performs inference in batches for given model on supplied data

        Parameters
        ----------
        model (tf.keras.Model or tf.keras.layers.Layer) 
        data (np.array) or [x,nln] in the case of NNAE 
        args (Namespace)
        arch (str)
        
        Returns
        -------
        np.array

    """
    data_tensor = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

    if arch =='AE' or arch == 'encoder':
        if arch=='encoder':
            output = np.empty([len(data), args.latent_dim])
        else:
            output = np.empty(data.shape)
        strt, fnnsh = 0, BATCH_SIZE
        for batch in data_tensor:
            output[strt:fnnsh,...] = model(batch).numpy() 
            strt = fnnsh
            fnnsh +=BATCH_SIZE
    
    else:
        output = np.empty([len(data), args.latent_dim])
        strt, fnnsh = 0, BATCH_SIZE
        for batch in data_tensor:
            output[strt:fnnsh,...] = model(batch)[0].numpy() # for disc
            strt = fnnsh
            fnnsh +=BATCH_SIZE

    return output


def get_error(model_type, 
              x, 
              x_hat, 
              z=None, 
              z_hat=None, 
              d_x=None, 
              d_x_hat=None, 
              ab=True,
              mean=True):
    """
        Gets the reconstruction error of a given model 

        Parameters
        ----------
        model_type (str) 
        x (np.array) 
        x_hat (np.array) 
        z (optional np.array) 
        z_hat (optional np.array) 
        d_x (optional np.array) 
        d_x_hat (optional np.array) 
        ab (bool) default True
        mean (bool) default True

        Returns
        -------
        np.array

    """

    if ((model_type == 'AE') or 
            (model_type == 'AAE') or
            (model_type == 'AE_SSIM') or
            (model_type == 'DAE') or
            (model_type == 'VAE')):
        error = x - x_hat 

    elif model_type == 'DAE_disc':
        discriminator_error  = d_x - d_x_hat
        if abs: discriminator_error = abs(discriminator_error)
        error = discriminator_error

    elif model_type == 'GANomaly':
        error = z- z_hat

    if ab:
        error = np.abs(error,dtype=np.float32)

    if mean:
        error =  error.mean(axis=tuple(range(1,error.ndim)))
    
    return error 
