#!/bin/sh
echo "Logging for run_lofar.sh at time: $(date)." >> log.log

limit=None
epochs=100
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=8
clip=1.75e6

#patch=256 
#model=RNET
#python -u main.py -model $model\
#				  -limit $limit \
#				  -anomaly_class rfi\
#				  -rfi None\
#				  -clip $clip\
#				  -anomaly_type $atype\
#				  -percentage_anomaly $percentage \
#				  -epochs $epochs \
#				  -latent_dim $ld \
#				  -crop True\
#				  -patches True\
#				  -crop_x $patch\
#				  -crop_y $patch\
#				  -patch_x $patch\
#				  -patch_y $patch\
#				  -patch_stride_x $patch \
#				  -patch_stride_y $patch \
#				  -data LOFAR\
#				  -data_path /data/mmesarcik/LOFAR/uncompressed/\
#				  -neighbors 5 20 50\
#				  -alpha 0.02\
#				  -algorithm knn\
#				  -seed 05-02-2022-05-13_c154d2 | tee -a lofar.log 
#
model=UNET
for patch in 32 128 256
do
		python -u main.py -model $model\
						  -limit $limit \
						  -anomaly_class rfi\
						  -rfi None\
						  -clip $clip\
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
						  -neighbors 5 20 50\
						  -alpha 0.02\
						  -algorithm knn\
						  -seed 05-02-2022-05-13_c154d2 | tee -a lofar.log 
						  #-seed $d$seed | tee -a lofar.log 
done

model=DAE
patch=32
python -u main.py -model $model\
				  -limit $limit \
				  -anomaly_class rfi\
				  -rfi None\
				  -clip $clip\
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
				  -neighbors 5 10 20 50\
				  -alpha 0.02\
				  -algorithm knn\
				  -seed 05-02-2022-05-13_c154d2 | tee -a lofar.log 
				  #-seed $d$seed | tee -a lofar.log 
