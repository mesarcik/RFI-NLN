#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=None
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=32
patch=32

for model in DAE RFI_NET UNET RNET
do
		for repeat in 1
		do
		python -u main.py -model $model\
						  -limit $limit \
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
						  -data_path /data/mmesarcik/LOFAR/uncompressed/\
						  -neighbors 20\
						  -algorithm knn\
						  -seed $d$seed | tee -a lofar.log 
		done 
done
