#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> log.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for ld in 16 32 64 
do
		for patch in 16 32 64
		do
				for rfi in rfi_dtv rfi_stations rfi_impulse 
				do
						python -u main.py -limit $limit\
										  -data HERA\
										  -data_path /home/mmesarcik/data/HERA/HERA_6_28-01-2022_MIXED.pkl\
										  -anomaly_class rfi\
										  -rfi_threshold 666\
										  -rfi $rfi\
										  -anomaly_type $atype\
										  -percentage_anomaly 0\
										  -epochs $epochs \
										  -latent_dim $ld \
										  -patches True \
										  -crop_x $patch\
										  -crop_y $patch\
										  -patch_x $patch \
										  -patch_y $patch \
										  -patch_stride_x $patch \
										  -patch_stride_y $patch \
										  -neighbors 1 2 5 10 \
										  -algorithm knn\
										  -seed $d$seed | tee -a hera.log 
				done 
		done
done

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
