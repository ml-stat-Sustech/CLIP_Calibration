import argparse
import torch
import os
import json


from dassl.utils import set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from tools.logger import setup_logger


# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

# few-shot CLIP
import trainers.classification.base_learner
import trainers.classification.coop
import trainers.classification.cocoop
import trainers.classification.zsclip
import trainers.classification.maple
import trainers.classification.vpt
import trainers.classification.kgcoop
import trainers.classification.proda
import trainers.classification.taskres
import trainers.classification.prograd
import trainers.classification.promptsrc
import trainers.classification.clip_adapter

# calibration
import trainers.calibration.tempscaling

# evaluation
import evaluators.vl_evaluator


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # replace base classfication evaluator with V-L evaluator
    cfg.TEST.EVALUATOR = 'VLClassification'

    # calibration 
    if args.calibration_config:
        
        calibration_cfgs = json.loads(args.calibration_config)
        args.calibration_config = calibration_cfgs
        print(calibration_cfgs, 'calibration_cfgs')

        if calibration_cfgs['BASE_CALIBRATION_MODE']:
            cfg.CALIBRATION.BASE_CALIBRATION_MODE = calibration_cfgs['BASE_CALIBRATION_MODE']

            if calibration_cfgs['SCALING_CONFIG']:
                cfg.merge_from_file(calibration_cfgs['SCALING_CONFIG'])
                fix_cfg_from_calibraion(cfg)
                cfg.CALIBRATION.SCALING.IF_SCALING = True

            if calibration_cfgs['BIN_CALIBRATOR_NAME']:
                cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME = calibration_cfgs['BIN_CALIBRATOR_NAME']
            
        # scaling
        if args.base_dir:
            cfg.CALIBRATION.SCALING.BASE_DIR = args.base_dir

        if args.base_learner:
            cfg.CALIBRATION.SCALING.BASE_LEARNER = args.base_learner

        if calibration_cfgs['IF_DAC']:
            cfg.CALIBRATION.DAC.IF_DAC= calibration_cfgs['IF_DAC']

        if calibration_cfgs['IF_PROCAL']:
            cfg.CALIBRATION.PROCAL.IF_PROCAL= calibration_cfgs['IF_PROCAL']




def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for CoCoOp
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for Prograd
    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PROGRAD.CTX_INIT = True  # initialization words
    cfg.TRAINER.PROGRAD.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROGRAD.LAMBDA = 1.0
    cfg.TRAINER.PROGRAD.T = 1.0
    cfg.TRAINER.PROGRAD.LOSS_NAME = "prograd" 

    # Config for KgCoOp
    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.KGCOOP.CTX_INIT = True  # initialization words
    cfg.TRAINER.KGCOOP.W = 8.0  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"

    # Config for ProDA
    cfg.TRAINER.PRODA = CN()
    cfg.TRAINER.PRODA.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PRODA.N_PROMPT = 32 
    cfg.TRAINER.PRODA.PROMPT_BS = 4 
    cfg.TRAINER.PRODA.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PRODA.ALPHA = 0.1
    # cfg.TRAINER.PRODA.CSC = False  # class-specific context
    # cfg.TRAINER.PRODA.CLASS_TOKEN_POSITION = "end"

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for TaskRes
    cfg.TRAINER.TaskRes = CN()
    cfg.TRAINER.TaskRes.N_CTX = 16  # number of context vectors
    cfg.TRAINER.TaskRes.CSC = False  # class-specific context
    cfg.TRAINER.TaskRes.CTX_INIT = ""  # initialization words
    cfg.TRAINER.TaskRes.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.TaskRes.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.TaskRes.RESIDUAL_SCALE = 1.0
    cfg.TRAINER.TaskRes.ENHANCED_BASE = 'none'

    # Config for adapter
    cfg.TRAINER.CLIP_ADAPTER = CN()
    cfg.TRAINER.CLIP_ADAPTER.RATIO = 0.2
    cfg.TRAINER.CLIP_ADAPTER.CTX_INIT = "a photo of a"  # initialization words

    # Config for calibration
    cfg.CALIBRATION = CN()
    cfg.CALIBRATION.BASE_CALIBRATION_MODE = None # scaling_based, bin_based

    # config for scaling-based calibration
    cfg.CALIBRATION.SCALING = CN()
    cfg.CALIBRATION.SCALING.IF_SCALING = False
    cfg.CALIBRATION.SCALING.BASE_DIR = ""
    cfg.CALIBRATION.SCALING.INIT_TEMP = 4.6052 # original CLIP temp
    cfg.CALIBRATION.SCALING.BASE_LEARNER = 'CoOp' # CoOp CoCoOp,....
    cfg.CALIBRATION.SCALING.MODE = 'TempScaling' # TempScaling/ ParameterizedTempScaling
    cfg.CALIBRATION.SCALING.BASE_EPOCH = 1 # origin tuned epoch for loade the model
    cfg.CALIBRATION.SCALING.EPOCH = 20 # epoch for scaling calirbation
    cfg.CALIBRATION.SCALING.LR = 5e-2 # learning rate for scaling calibration

    # config for parameterized temp scaling calibration
    cfg.CALIBRATION.P_TS = CN()
    cfg.CALIBRATION.P_TS.N_LAYERS = 2
    cfg.CALIBRATION.P_TS.N_NODES = 5
    cfg.CALIBRATION.P_TS.TOP_K_LOGITS = 10 


    # config for bin-based calibration
    cfg.CALIBRATION.BIN = CN()
    cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME = None # histogram_binning, isotonic_regression, multi_isotonic_regression


    # Config for task difficulty aware calibration
    cfg.CALIBRATION.DAC = CN()
    cfg.CALIBRATION.DAC.IF_DAC = False
    cfg.CALIBRATION.DAC.K = 5 # K text nearest neighbor text

    # Config for proximity-informed calibration
    cfg.CALIBRATION.PROCAL = CN()
    cfg.CALIBRATION.PROCAL.IF_PROCAL = False #  density estimator / bin-mean-shift
    cfg.CALIBRATION.PROCAL.IMAGE_K = 5 # calculation the knn distance for proximity-base calibration, for small dataset, set 5 , for imagenet, set 10

    # config for calibration metrics
    cfg.CALIBRATION.METRICS = CN()
    cfg.CALIBRATION.METRICS.ECE_BINS = 10 # the number of bins for ece calculation
    cfg.CALIBRATION.METRICS.PIECE_BINS = 10 # the number of nearest neighbor in piece calculation


def fix_cfg_from_calibraion(cfg): 
    cfg.OPTIM.LR = cfg.CALIBRATION.SCALING.LR
    cfg.CALIBRATION.SCALING.BASE_EPOCH = cfg.OPTIM.MAX_EPOCH
    cfg.OPTIM.MAX_EPOCH = cfg.CALIBRATION.SCALING.EPOCH



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the tuning method config file
    if args.config_file:
        print(args.config_file, 'args.config_file')
        cfg.merge_from_file(args.config_file)
        
    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    
    base_dir = cfg.OUTPUT_DIR
    base_name = 'log'

    if cfg.CALIBRATION.SCALING.IF_SCALING:
        base_name = base_name + '_' + str(cfg.CALIBRATION.SCALING.MODE)

    if cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME:
        base_name = base_name + '_' + str(cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME)

    if cfg.CALIBRATION.DAC.IF_DAC:
        base_name = base_name + '_dac'

    if cfg.CALIBRATION.PROCAL.IF_PROCAL:
        base_name = base_name + '_procal'
        
    base_name = base_name +'.txt'

    setup_logger(os.path.join(base_dir, base_name))


    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # calibration or not
    if cfg.CALIBRATION.SCALING.IF_SCALING:
        cfg = cfg.clone()
        cfg.defrost()
        base_learner = cfg.TRAINER.NAME
        cfg.CALIBRATION.SCALING.BASE_LEARNER = base_learner
        cfg.TRAINER.NAME = cfg.CALIBRATION.SCALING.MODE # use calibration trainer instand of base few-shot trainer
        trainer = build_trainer(cfg)
        cfg.TRAINER.NAME = args.trainer # replace with origin trainer
    else:
        trainer = build_trainer(cfg)

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))


    if args.eval_only:
        # trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.load_model(args.model_dir, epoch=cfg.OPTIM.MAX_EPOCH)
        print(args.load_epoch, 'load_epochload_epochload_epoch')
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--calibration-config-file",
        type=str,
        default="",
        help="path to config file for calibration",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="load model from few-shot learner",
    )
    parser.add_argument(
        "--base-learner",
        type=str,
        default="",
        help="base learner",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--calibration-config", type=str, help="calibration config"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
