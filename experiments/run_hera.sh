#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> hera.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=64


for model in UNET
do
		for rad in 1 
		do
				for patch in 32 
				do
						for threshold in  0.5 1 3 5 7 9 10 20 50 100 200 
						do
								python -u main.py -model $model\
												  -limit $limit\
												  -data HERA\
												  -data_path /data/mmesarcik/HERA/HERA_04-03-2022_all.pkl\
												  -radius $rad\
												  -anomaly_class rfi\
												  -rfi_threshold $threshold\
												  -anomaly_type $atype\
												  -percentage_anomaly 0\
												  -epochs $epochs \
												  -latent_dim $ld \
												  -patches True\
												  -crop_x $patch\
												  -crop_y $patch\
												  -patch_x $patch \
												  -patch_y $patch \
												  -patch_stride_x $patch \
												  -patch_stride_y $patch \
												  -neighbors 20\
												  -algorithm knn\
												  -alpha 0.9\
												  -seed $d$seed | tee -a hera.log 
												  #-seed $d$seed | tee -a hera.log 
						done
				done										  
		done										  	  
done 
