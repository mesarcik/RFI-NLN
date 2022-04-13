#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=1000
epochs=200
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=1024
patch=64

for sigma in 0.95 1 
do
		for model in AE UNET DAE 
		do
				python -u main.py -model $model\
								  -limit $limit \
								  -anomaly_class rfi\
								  -rfi None\
								  -rfi_threshold $sigma\
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
								  -data_path /data/mmesarcik/LOFAR/uncompressed/\
								  -neighbors 5 20\
								  -alpha 0.1\
								  -algorithm knn\
								  -seed 04-12-2022-09-33_a124b6 | tee -a lofar.log 
								  #-seed $d$seed | tee -a lofar.log 
		done
done 
