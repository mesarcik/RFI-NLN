#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=10
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for patch in 8 16 32 64 128 
do
		for ld in 64 128
		do
				python -u main.py -limit $limit \
								  -anomaly_class rfi\
								  -rfi None\
								  -anomaly_type $atype\
								  -percentage_anomaly $percentage \
								  -epochs $epochs \
								  -latent_dim $ld \
								  -crop True\
								  -patches True\
								  -crop_x $patch\
								  -crop_y $patch\
								  -patch_x $patch\
								  -patch_y $patch\
								  -patch_stride_x $patch \
								  -patch_stride_y $patch \
								  -data LOFAR\
								  -data_path /home/mmesarcik/data/LOFAR/uncompressed/\
								  -neighbors 1 2 5 10\
								  -alpha 0 0.25 0.5 0.75 0.9 1.0\
								  -algorithm knn\
								  -seed $d$seed | tee -a lofar.log 
		done
done 
