import os
import time

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
# _C.DATA.DATA_PATH = ''
# Dataset name, could be overwritten by command line argument. Choose from:'PHEME', 'Twitter15_16', 'Twitter', 'Weibo'
# _C.DATA.DATASET = ''
# -----------------------------------------------------------------------------
# Data Preprocess
# -----------------------------------------------------------------------------
_C.PRE = CN()
# Model type
_C.PRE.EMBEDDING_MODEL = 'ernie-3.0-base' # xlm-roberta-base
# Choose from:'PHEME', 'Twitter15_16', 'Twitter', 'Weibo'
_C.PRE.DATASET_NAME = 'Weibo'


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.INIT_MODEL_PATH = ''
_C.MODEL.NAME = 'GCN'


# -----------------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------------
_C.hyper_parameters = CN()
_C.hyper_parameters.datasetname = 'Weibo' 
# Choose from:'PHEME', 'Twitter15_16', 'Twitter', 'Weibo','SCL_PHEME', 'SCL_Twitter15_16', 'SCL_Twitter', 'SCL_Weibo'
_C.hyper_parameters.lr = 0.0001
_C.hyper_parameters.EPOCHS = 300
_C.hyper_parameters.TDdroprate = 0.2
_C.hyper_parameters.BUdroprate = 0.2
_C.hyper_parameters.noise_rate = 0.2
_C.hyper_parameters.WEIGHT_DECAY = 1e-4
_C.hyper_parameters.Iterations = 1
_C.hyper_parameters.lThreshold = 0.6
_C.hyper_parameters.hThreshold = 0.8
_C.hyper_parameters.lWeight = 1
_C.hyper_parameters.hWeight = 3
_C.hyper_parameters.EMA_m = 0.985
_C.hyper_parameters.patience = 5
_C.hyper_parameters.init_model_path = 'model/Twitter15_16/Twitter15_16_2_glove_init.m' #'model/Weibo/Weibo_0_init.m'#'model/PHEME/BiGCNPHEME.m'#'model/PHEME/0_PHEME_0_init.m'#'model/Twitter/Twitter_2_init.m' #'model/Twitter/BiGCNTwitter_2_labeled.m' #'model/Weibo/BiGCNWeibo_2_labeled.m'


def update_config(config, args, process):
    
    config.defrost()
    if process == 'source_train':
        if args.lr:
            _C.hyper_parameters.lr = args.lr
        if args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size
        if args.epochs:
            _C.hyper_parameters.EPOCHS = args.epochs
        if args.TDdroprate:
            _C.hyper_parameters.TDdroprate = args.TDdroprate
        if args.BUdroprate:
            _C.hyper_parameters.BUdroprate = args.BUdroprate      
        if args.weight_decay:
            _C.hyper_parameters.WEIGHT_DECAY = args.weight_decay
        if args.patience:
            _C.hyper_parameters.patience = args.patience
        if args.datasetname:
            _C.hyper_parameters.datasetname = args.datasetname
        if args.iterations:
            _C.hyper_parameters.iterations = args.iterations
            
    if process == 'target_train':
    # merge from specific arguments
        if args.lr:
            _C.hyper_parameters.lr = args.lr
        if args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size
        if args.epochs:
            _C.hyper_parameters.EPOCHS = args.epochs
        if args.TDdroprate:
            _C.hyper_parameters.TDdroprate = args.TDdroprate
        if args.BUdroprate:
            _C.hyper_parameters.BUdroprate = args.BUdroprate
        if args.noise_rate:
            _C.hyper_parameters.noise_rate = args.noise_rate
        if args.weight_decay:
            _C.hyper_parameters.WEIGHT_DECAY = args.weight_decay
        if args.lThreshold:
            _C.hyper_parameters.lThreshold = args.lThreshold
        if args.hThreshold:
            _C.hyper_parameters.hThreshold = args.hThreshold
        if args.patience:
            _C.hyper_parameters.patience = args.patience
        if args.lWeight:
            _C.hyper_parameters.lWeight = args.lWeight
        if args.hWeight:
            _C.hyper_parameters.hWeight = args.hWeight
        if args.EMA_m:
            _C.hyper_parameters.EMA_m = args.EMA_m
        if args.init_model_path:
            _C.hyper_parameters.init_model_path = args.init_model_path
        if args.datasetname:
            _C.hyper_parameters.datasetname = args.datasetname
        if args.iterations:
            _C.hyper_parameters.iterations = args.iterations
    
    if process == 'PRE':
        if args.processdataset:
            _C.PRE.DATASET_NAME = args.processdataset
    config.freeze()


def get_config(args, process='PRE'):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, process)

    return config
