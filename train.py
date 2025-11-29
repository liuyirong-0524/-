import os, torch.multiprocessing as mp
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
import yaml
import argparse
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    add_dict_to_argparser
)

import torch
import os
from mpi4py import MPI
# Import the main training function (assumed to be defined elsewhere)
from train_part import train_fun

# --- Distributed Setup (MPI/CUDA) ---
comm =MPI.COMM_WORLD
rank = comm.Get_rank()
gpu_ids = [0]
# Set the specific GPU for the current MPI process
torch.cuda.set_device(gpu_ids[rank])

def main():
    """
    Functionality: Main execution function. Parses arguments, sets up 
    distributed communication, and starts the core training process.
    """
    # 1. Parse all arguments (including those loaded from YAML)
    args = create_argparser().parse_args()
    
    # 2. Setup distributed training backend (e.g., NCCL/Gloo)
    dist_util.setup_dist()
    
    # 3. Call the core training function
    train_fun(args, args.log_dir, args.n , args.pre_model_dir)

def create_argparser():
    """
    Functionality: Creates an argument parser and loads all configuration 
    parameters from the 'config_train.yaml' file.
    
    Outputs:
    :return: Configured argparse object.
    """
    parser = argparse.ArgumentParser()
    # Note: The first add_argument call appears to be a typo in the original source, 
    # but the subsequent code correctly loads the file.
    parser.add_argument("--./code/config/config_train.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('./code/config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":
    
    main()
