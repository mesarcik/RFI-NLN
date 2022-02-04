#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> log.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO

for ld in 32 64 128
do
		for patch in 8 16 32 64
		do
				python -u main.py -limit $limit\
								  -data HERA\
								  -data_path /home/mmesarcik/data/HERA/HERA_6_31-01-2022_MIXED.pkl\
								  -anomaly_class rfi\
								  -rfi_threshold 666\
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
								  -alpha 0 0.25 0.5 0.75 0.9 1.0\
								  -algorithm knn\
								  -seed $d$seed | tee -a hera.log 
		done
done

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
