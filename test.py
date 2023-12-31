import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference, check_kl, check_ortho, do_ranking_history
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #output_dir = cfg.OUTPUT_DIR
    #if output_dir and not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

 #   logger = setup_logger("transreid", output_dir, if_train=False)
  #  logger.info(args)

   # if args.config_file != "":
    #    logger.info("Loaded configuration file {}".format(args.config_file))
     #   with open(args.config_file, 'r') as cf:
     #       config_str = "\n" + cf.read()
     #       logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_mode, num_classes, camera_num = make_dataloader(cfg)

    model = make_model(cfg, num_mode, num_class=num_classes, camera_num=camera_num)
    model.load_param(cfg.TEST.WEIGHT)

       #do_ranking_history(cfg, model, val_loader, num_query)
    #check_ortho(cfg, model, val_loader, num_query, camera_num)
    #check_kl(cfg, model, val_loader, num_query, camera_num)
    do_inference(cfg,
                model,
                 val_loader,
                 num_query, camera_num, num_mode, eval_mode='single')

