#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=None
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=128

for patch in 64 32 16
do
		for model in UNET AE DAE VAE 
		do
				for threshold in 0.5 0.95 1 2 5 10
				do
						python -u main.py -model $model\
										  -limit $limit \
										  -anomaly_class rfi\
										  -rfi None\
										  -rfi_threshold $threshold\
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
										  -alpha 0.1 0.5 0.9\
										  -algorithm knn\
										  -seed $d$seed | tee -a lofar.log 
										  #-seed 03-30-2022-06-26_7f4f1c | tee -a lofar.log 
				  done
		done
done 
