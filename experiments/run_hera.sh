#!/bin/sh
echo "Logging for run_mnist.sh at time: $(date)." >> log.log

limit=None 
epochs=50 
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
patch=64
anomaly=rfi
ld=128

for percentage in 0.01 0.05 0.1 0.2 0.5 0.75 0.9
do
		python -u main.py -limit $limit \
						  -anomaly_class $i \
						  -anomaly_type $atype\
						  -percentage_anomaly $percentage \
						  -epochs $epochs \
						  -latent_dim $ld \
						  -patches True\
						  -crop_x $patch\
						  -crop_y $patch\
						  -patch_x $patch \
						  -patch_y $patch \
						  -patch_stride_x $patch \
						  -patch_stride_y $patch \
						  -data HERA\
						  -data_path /data/mmesarcik/hera/HERA_6_24-09-2021.pkl\
						  -neighbors 1 2 5 10 \
						  -algorithm knn\
						  -seed $d$seed | tee -a hera.log 
done 

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
