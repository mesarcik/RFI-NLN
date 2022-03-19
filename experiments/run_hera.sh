#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> hera.log

limit=None
epochs=50
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for model in DAE AE VAE
do
		for ld in 8 32 64 128 1024 
		do
				for patch in 16 32  
				do
						for rfi in rfi_stations rfi_dtv rfi_impulse rfi_scatter 
						do
								python -u main.py -model $model\
												  -limit $limit\
												  -data HERA\
												  -data_path /home/mmesarcik/data/HERA/HERA_04-03-2022_all.pkl\
												  -rfi $rfi\
												  -anomaly_class rfi\
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
												  #-seed 03-10-2022-06-42_8dae4a | tee -a hera.log
						done
				done
		done										  
done										  	  

for patch in 16 32 64 128 
do
		for rfi in rfi_stations rfi_dtv rfi_impulse rfi_scatter 
		do
				python -u main.py -model UNET\
						  -limit $limit\
						  -data HERA\
						  -data_path /home/mmesarcik/data/HERA/HERA_04-03-2022_all.pkl\
						  -rfi $rfi\
						  -anomaly_class rfi\
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
		 done
done 
#-rfi_threshold $threshold\
