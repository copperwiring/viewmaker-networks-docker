""" Return the state dictionary of the ResNet encoder with the saved weights
"""

import os
import torch

from dotmap import DotMap

from src.systems.eval_image_systems import TransferEvalExpertSystem
from src.utils.utils import load_json

viewmaker_ckpt_path = "../experiments/experiments/pretrain_viewmaker_cifar_simclr_resnet18/checkpoints"
expert_ckpt_path = "../experiments/experiments/pretrain_expert_cifar_simclr_resnet18/checkpoints"

def ckpt_name(epoch):
    checkpoint_name = "epoch=" + str(epoch) + ".ckpt"
    return checkpoint_name


def get_encoder_system(mode, model_path, device, epoch_num):
    if mode == 'viewmaker':
        config_file = '../experiments/experiments/pretrain_viewmaker_cifar_simclr_resnet18/config.json'
    elif mode == 'expert':
        config_file = '../experiments/experiments/pretrain_expert_cifar_simclr_resnet18/config.json'
    
    config_json = load_json(config_file)

    config = DotMap(config_json)
    checkpoint_name = ckpt_name(epoch_num)
    config_ckpt_list = [config, checkpoint_name]
    system = TransferEvalExpertSystem(config_ckpt_list)
    checkpoint = torch.load(model_path, map_location=device)
    system.load_state_dict(checkpoint['state_dict'], strict=False)
    encoder = system.model.eval()
    return system, encoder

def return_system_encoder(mode, epoch_num):
    if mode == 'viewmaker':
        ckpt_path = viewmaker_ckpt_path
    elif mode == 'expert':
        ckpt_path = expert_ckpt_path

    checkpoint_name = ckpt_name(epoch_num)
    model_path = os.path.join(ckpt_path,checkpoint_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system, _ = get_encoder_system(mode, model_path, device, epoch_num)
    return system.encoder
