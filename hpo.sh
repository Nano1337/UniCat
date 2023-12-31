# train

root='root'
for lr in .032
do
    for bs in 256
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--nproc_per_node=4 --master_port 1234 train.py \
	--config_file configs/RGBNT100/unis-mmc.yml \
	MODEL.DIST_TRAIN True SOLVER.BASE_LR $lr SOLVER.IMS_PER_BATCH $bs \
	OUTPUT_DIR "${root}_bs_${bs}_lr_${lr}" \

    done
done


#CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/Flare/dc_former.yml 
# test
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
