#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=500
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=128
patch=32

for sigma in 5 7 10 15 20 50
do
		for model in AE DAE_disc UNET 
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
								  -neighbors 20\
								  -alpha 0.25\
								  -algorithm knn\
								  -seed $d$seed | tee -a lofar.log 
		done
done 
