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
								  -data HERA\
								  -data_path /data/mmesarcik/hera/HERA_6_24-09-2021.pkl\
								  -neighbors 1 2 5 10 16 20\
								  -algorithm knn\
								  -seed $d$seed | tee -a hera.log 
		done
done 

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
