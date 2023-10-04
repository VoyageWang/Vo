from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import argparse
import glob
import multiprocessing as mp
import torch
from torch import nn
from detectron2.config import configurable

# 默认的
def add_mlp_config(cfg):
    cfg.MODEL.INDIM = 30
    cfg.MODEL.OUTDIM = 50
    cfg.MODEL.HIDEN = 20

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="/root/voyage_pro/mlp_config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_mlp_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze() #冻结掉防止后续的改动
    return cfg




class Model(nn.Module):
    @configurable
    def __init__(self,*, indim: int, hiden: int, outdim: int):
        super().__init__()
        self.hiden = hiden
        self.ln = nn.Linear(indim,self.hiden)
        self.act = nn.ReLU()
        self.out = nn.Linear(self.hiden, outdim)
    
    @classmethod
    def from_config(cls,cfg):
        hiden = cfg.MODEL.HIDEN
        indim = cfg.MODEL.INDIM
        outdim = cfg.MODEL.OUTDIM    
        
        return {
            'hiden': hiden,
            'indim': indim,
            'outdim': outdim
        }
    
    def forward(self,x):
        return self.out(self.act(self.ln(x)))




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    x = torch.randn(1,50,30)
    mymodel = Model(cfg)
    print(mymodel(x).shape)


