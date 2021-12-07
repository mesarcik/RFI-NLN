#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=1000
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
patch=64
ld=16

for percentage in 0.1 0.2 0.3 0.4 
do
		python -u main.py -limit $limit \
						  -data_path /home/mmesarcik/data/HERA/HERA_6_29-11-2021_MIXED.pkl\
						  -anomaly_class rfi\
						  -anomaly_type $atype\
						  -percentage_anomaly $percentage \
						  -epochs $epochs \
						  -latent_dim $ld \
						  -patches True \
						  -crop_x $patch\
						  -crop_y $patch\
						  -patch_x $patch \
						  -patch_y $patch \
						  -patch_stride_x $patch \
						  -patch_stride_y $patch \
						  -data HERA \
						  -neighbors 1 2 5 10 \
						  -algorithm knn\
						  -seed $d$seed | tee -a hera.log 
done 

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
