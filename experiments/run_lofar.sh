#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None 
epochs=50
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for ld in 128 
do
		for i in rfi 
		do
				python -u main.py -limit $limit \
								  -anomaly_class $i \
								  -anomaly_type $atype\
								  -percentage_anomaly $percentage \
								  -epochs $epochs \
								  -latent_dim $ld \
								  -data LOFAR\
								  -data_path pathnotinuse\
								  -neighbors 1 2 5 10 16 20\
								  -algorithm knn\
								  -seed $d$seed | tee -a lofar.log 
		done
done 

#TODO
#python report.py -data LOFAR\
#				 -seed $d$seed\
#				 -anomaly_type $atype\
#				 -data_path /data/mmesarcik/LOFAR/LOFAR_training_data/datasets/LOFAR_dataset_14-09-2021.pkl
