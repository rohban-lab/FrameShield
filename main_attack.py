import os
import shutil
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from datasets.build import build_dataloader
from models import xclip
from utils.config import get_config
from utils.tools import generate_text, evaluate_result
from utils.logger import create_logger


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    # model parameters
    parser.add_argument("--local-rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.001, type=float, help='weight of sparse loss')
    
    args = parser.parse_args()

    config = get_config(args)

    return args, config

def main(config, num_attack_steps=10):
    train_data, _, _, _, _, test_loader, _ = build_dataloader(config)
    model, _, model_path = xclip.load(config.MODEL.PRETRAINED,
                                      config.MODEL.ARCH, device="cpu", 
                                      jit=False, 
                                      T=config.DATA.NUM_FRAMES,
                                      droppath=config.MODEL.DROP_PATH_RATE,
                                      use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                      use_cache=config.MODEL.FIX_TEXT,
                                      logger=logger)
    model = model.cuda()
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    
    eps = config.ADV_TRAIN.EPS / 255.0
    step_size = 2.5 * (eps / num_attack_steps)
    
    vid_list = get_vid_list(config)
    scores_dict = dict()
    scores_dict['prd'] = dict()    

    text_labels = generate_text(train_data)
    texts = text_labels.cuda(non_blocking=True)
    
    gt = get_gt(config)
    vid2key = {vid: key for vid, key in enumerate(gt.keys())}

    curr_vid = -1
    vid_gt = None
    for _, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = batch_data["imgs"].cuda()
        b, n, c, t, h, w = images.size()
        
        labels = []
        for b_idx in range(b):
            if int(batch_data['vid'][b_idx]) != curr_vid:
                curr_vid = int(batch_data['vid'][b_idx])
                vid_gt = gt[vid2key[curr_vid]]
                
            labels.append(max(vid_gt[:t * config.DATA.FRAME_INTERVAL:config.DATA.FRAME_INTERVAL]))
            vid_gt = vid_gt[t * config.DATA.FRAME_INTERVAL:]
        
        images = rearrange(images, 'b n c t h w -> (b n) t c h w')
                    
        original_images = images.clone().detach()
        adv_images = images.clone().detach()        
                
        for _ in range(num_attack_steps):
            adv_images.requires_grad_()
            
            outputs = model(adv_images, texts)
            scores = F.softmax(outputs['y'], dim=-1)
            scores = rearrange(scores, '(b n) c -> b n c', b=b)
            logits = scores[:, :, 1].reshape(-1)
            
            labels_binary = torch.tensor(labels).cuda().float()
            
            coef = torch.where(labels_binary == 0.0, torch.ones_like(labels_binary), -torch.ones_like(labels_binary))
            cost = torch.dot(coef, logits)
            cost.backward()

            grad_sign = adv_images.grad.sign()
            adv_images = adv_images.detach() + step_size * grad_sign
            perturbation = torch.clamp(adv_images - original_images, min=-eps, max=eps)
            adv_images = torch.clamp(original_images + perturbation, 0, 1)
                
        with torch.no_grad():
            final_outputs = model(adv_images, texts)
            
            final_scores = F.softmax(final_outputs['y'], dim=-1)
            final_scores = rearrange(final_scores, '(b n) c -> b n c', b=b)
            final_scores_np_prd = final_scores.cpu().data.numpy()

            for ind in range(final_scores_np_prd.shape[0]):
                v_name = vid_list[batch_data["vid"][ind]]
                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                scores_dict['prd'][v_name].append(final_scores_np_prd[ind])
    
    tmp_dict = {}
    for v_name in scores_dict["prd"].keys():
        p_scores = np.array(scores_dict["prd"][v_name]).copy()
        if p_scores.shape[0] == 1:
            # 1,T,2
            tmp_dict[v_name] = [p_scores[0, :, 1]]
        else:
            # T,1,2
            tmp_dict[v_name] = [p_scores[:, 0, 1]]

    auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)
    logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t')


def get_gt(config):
    videos = {}
    for video in open(config.DATA.VAL_FILE):
        vid = video.strip().split(' ')[0].split('/')[-1]
        video_len = int(video.strip().split(' ')[1])
        sub_video_gt = np.zeros((video_len,), dtype=np.int8)
        anomaly_tuple = video.split(' ')[3:]
        for ind in range(len(anomaly_tuple) // 2):
            start = int(anomaly_tuple[2 * ind])
            end = int(anomaly_tuple[2 * ind + 1])
            if start > 0:
                sub_video_gt[start:end] = 1
        videos[vid] = sub_video_gt
        
    return videos


def get_vid_list(config):
    vid_list = []
    with open(config.DATA.VAL_FILE, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)
    return vid_list


def save_image(image, name):
    image = image.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    path = os.path.join(config.OUTPUT, f"{name}.png")
    plt.imsave(path, image)
    logger.info(f"Saved image to {path}")


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"Working dir: {config.OUTPUT}")
    
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
