import torch
import torch.distributed as dist
import numpy as np
import os
import argparse
import random

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from processor import do_train
from solver import make_optimizer, scheduler_factory as create_scheduler
from utils.logger import setup_logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    # parse input args from config file and command line
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # setup logger
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.local_rank == 0:
        logger = setup_logger("transreid", output_dir, if_train=True)
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)
    else:
        logger = None
    if logger and args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    if logger:
        logger.info("Running with config:\n{}".format(cfg))

    # init distributed training
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Prepare data loaders, model, loss function, optimizers, and scheduler
    train_loader, train_loader_normal, val_loader, num_query, num_mode, num_classes, camera_num = make_dataloader(cfg)
    model = make_model(cfg, num_mode, num_class=num_classes, camera_num=camera_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    # start training
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank, num_classes, camera_num
    )
