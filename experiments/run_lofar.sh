#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=None 
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for patch in 16 32 64 128 
do
		for ld in 16 64 128
		do
				python -u main.py -limit $limit \
								  -anomaly_class rfi\
								  -rfi 0\
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
								  -data LOFAR\
								  -data_path /home/mmesarcik/data/LOFAR/uncompressed/\
								  -neighbors 1 2 5 10 16 20\
								  -algorithm knn\
								  -seed $d$seed | tee -a lofar.log 
		done
done 
