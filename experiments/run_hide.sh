#!/bin/sh
echo "Logging for run_hide.sh at time: $(date)." >> log.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
perc=0

for patch in 8 16 32
do
		for ld in 16 32 64 128
		do
				python -u main.py -limit $limit \
								  -data_path /home/mmesarcik/data/HIDE/full_year/seek_cache\
								  -anomaly_class rfi\
								  -anomaly_type $atype\
								  -percentage_anomaly $perc\
								  -epochs $epochs \
								  -latent_dim $ld \
								  -patches True \
								  -crop_x $patch\
								  -crop_y $patch\
								  -patch_x $patch \
								  -patch_y $patch \
								  -patch_stride_x $patch \
								  -patch_stride_y $patch \
								  -data HIDE\
								  -neighbors 1 2 5 10 15\
								  -algorithm knn\
								  -seed $d$seed | tee -a hide.log 
		done 
done 
