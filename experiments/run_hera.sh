#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> hera.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=8
patch=32
threshold=10

for model in DAE UNET RNET RFI_NET
do
		for repeat in 1 2 3 
		do
				python -u main.py -model $model\
								  -limit $limit\
								  -data HERA\
								  -data_path /data/mmesarcik/HERA\
								  -anomaly_class rfi\
								  -anomaly_type $atype\
								  -epochs $epochs \
								  -latent_dim $ld \
								  -rfi_threshold $threshold\
								  -patches True\
								  -crop_x $patch\
								  -crop_y $patch\
								  -patch_x $patch \
								  -patch_y $patch \
								  -patch_stride_x $patch \
								  -patch_stride_y $patch \
								  -neighbors 20\
								  -algorithm knn\
								  -seed $d$seed | tee -a hera.log 
		done
done										  
