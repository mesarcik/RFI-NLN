def print_epoch(model_type,epoch,time,losses,AUC):
    """
        Messages to print while training

        model_type (str): type of model_type
        epoch (int): The current epoch
        time (int): The time elapsed per Epoch
        losses (dict): the losses  of the model 
        AUC (double): AUROC score of the model

    """
    print ('__________________')
    print('Epoch {} at {} sec \n{} losses: {} \nAUC = {}'.format(epoch,
                                                               time,
                                                               model_type,
                                                               losses,
                                                               AUC))

