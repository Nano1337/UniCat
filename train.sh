# distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port 1234 train.py \
--config_file configs/RGBNT201/dc_former.yml \
MODEL.DIST_TRAIN True

# single gpu training
# python train.py --config_file configs/RGBNT100/resnet.yml 

# testing
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
