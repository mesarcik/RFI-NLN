# Learning to detect RFI in radio astronomy without seeing it 
A repository containing the implementation of the paper entitled "Learning to detect RFI in radio astronomy without seeing it"


## Installation 
Install conda environment by:
``` 
    conda create --name rfi python=3.7
``` 
Run conda environment by:
``` 
    conda activate rfi
``` 

Install dependancies by running:
``` 
    pip install -r requirements
``` 


## Replication of results in paper 
Run the following to replicate the results for HERA or LOFAR
```
    sh experiments/run_lofar.sh
    sh experiments/run_hera.sh
```

### Dataset  
You will need to download the [LOFAR and HERA datasets](https://zenodo.org/record/6724065) and specify the its path using `-data_path` command line option.

## Reproduce results
For HERA dataset run the following: 
```
  ./experiments/run_hera.sh
```

For the LOFAR dataset run the following: 
```
  ./experiments/run_hera.sh
```


## Dependencies
**NOTE** This project makes use of the python wrapping for AOFlagger, for detailed instructions on the installation and usage see [the AOFlagger documentation](https://aoflagger.readthedocs.io/en/latest/)

## Licensing
Source code of RFI-AE is licensed under the MIT License.
