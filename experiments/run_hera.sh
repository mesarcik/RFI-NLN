#!/bin/sh
echo "Logging for run_hera.sh at time: $(date)." >> log.log

limit=None
epochs=100
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=128

for patch in 8 16 32 64 
do
		python -u main.py -limit $limit\
						  -data HERA\
						  -data_path /home/mmesarcik/data/HERA/HERA_03-03-2022_all.pkl\
						  -anomaly_class rfi\
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
						  -neighbors 1 5 10 16 20\
						  -algorithm knn\
						  -alpha 0.0 0.1 0.5 0.9 1.0\
						  -seed 03-02-2022-03-02_ef136a | tee -a hera.log 
						  #-seed $d$seed | tee -a hera.log 
done

#patch=16
#for rfi in rfi_stations rfi_dtv rfi_impulse 
#do
#		python -u main.py -limit $limit\
#						  -data HERA\
#						  -data_path /home/mmesarcik/data/HERA/HERA_02-03-2022_all.pkl\
#						  -rfi $rfi \
#						  -anomaly_class rfi\
#						  -anomaly_type $atype\
#						  -percentage_anomaly 0\
#						  -epochs $epochs \
#						  -latent_dim $ld \
#						  -patches True \
#						  -crop_x $patch\
#						  -crop_y $patch\
#						  -patch_x $patch \
#						  -patch_y $patch \
#						  -patch_stride_x $patch \
#						  -patch_stride_y $patch \
#						  -neighbors 1 10 20 30\
#						  -algorithm knn\
#						  -alpha 0.0 0.5 0.9 1.0\
#						  -seed $d$seed | tee -a hera.log 
#						 # -seed 02-18-2022-11-09_290324 | tee -a hera.log 
#done

#python report.py -data HERA -seed $d$seed -anomaly_type $atype
