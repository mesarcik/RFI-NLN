# RFI mitigation using novelty detection in autoencoders
A repository containing the implementation of the paper entitled "//////"

## Installation 
Install conda environment by:
``` 
    conda create --name rfi-ae python=3.7
``` 
Run conda environment by:
``` 
    conda activate rfi-ae 
``` 

Install dependancies by running:
``` 
    pip install -r dependancies
``` 

Additionally for training on a GPU run:
``` 
    conda install -c anaconda tensorflow-gpu=2.2.0
``` 


## Replication of results in paper 
Run the following to replicate the results for HERA or LOFAR
```
    sh experiments/run_lofar.sh
    sh experiments/run_hera.sh
```

### Dataset [TODO] 
You will need to download the [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and specify the its path using `-mvtec_path` command line option.

## Training [TODO]
Run the following: 
```
    python main.py -anomaly_class <0,1,2,3,4,5,6,7,8,9,bottle,cable,...> \
                   -percentage_anomaly <float> \
                   -limit <int> \
                   -epochs <int> \
                   -latent_dim <int> \
                   -data <MNIST,FASHION_MNIST,CIFAR10,MVTEC> \
                   -mvtec_path <str>\
                   -neighbors <int(s)> \
                   -algorithm <knn> \
		   -patches <True, False> \
		   -crop <True, False> \
		   -rotate <True, False> \
		   -patch_x <int> \    
		   -patch_y <int> \    
		   -patch_x_stride <int> \    
		   -patch_y_stride <int> \    
		   -crop_x <int> \    
		   -crop_y <int> \    
```
## Reporting Results [TODO]
Run the following given the correctly generated results files:
```
    python report.py -data <LOFAR/HERA> -seed <filepath-seed>
```

## Licensing
Source code of RFI-AE is licensed under the MIT License.
