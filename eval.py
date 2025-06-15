import os
import sys
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import umap

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu

sys.path.append('..')
from semilearn.core.utils import get_net_builder, get_dataset, over_write_args_from_file
from semilearn.algorithms.openmatch.openmatch import OpenMatchNet
from semilearn.algorithms.dac.dac import OSCNet
from einops import rearrange

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='')


def load_model_at(step='best', use_ema=True):
    args.step = step
    if step == 'best':
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_best.pth"
    else:
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_at_{args.step}_step.pth"
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    if use_ema:
        load_model = checkpoint['ema_model']
    else:
        load_model = checkpoint['model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if step == 'best':
        args.save_dir = os.path.join(save_dir, f"model_best")
    else:
        args.save_dir = os.path.join(save_dir, f"step_{args.step}")
    # os.makedirs(args.save_dir, exist_ok=True)
    _net_builder = get_net_builder(args.net, args.net_from_name)
    net = _net_builder(num_classes=args.num_classes)
    if args.algorithm == 'openmatch':
        net = OpenMatchNet(net, args.num_classes)
    elif args.algorithm == 'dac':
        net = OSCNet(net, args.num_classes, num_heads=args.num_heads, proj_dim=args.proj_dim)
    keys = net.load_state_dict(load_state_dict, strict=False)
    print(f'Model at step {args.step} loaded!')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    return net


def calculate_ED(logits):
    return torch.logsumexp(logits, dim=1) * (1 - 1 / torch.exp(logits).max(dim=1)[0])

def calculate_consensus(probs_comm, temp_d=1.0, mode='l1'):
    nb, nh, nc = probs_comm.shape
    if mode == "l1":
        consensus = (probs_comm.unsqueeze(1) - probs_comm.unsqueeze(2)).abs().mean([-3, -2, -1])
        consensus = torch.exp(-consensus / temp_d)
    elif mode == "kl":
        marginal_p = probs_comm.mean(dim=0)
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> 1 (h g) (d e)")  # 1, (H, H), (D, D)
        pointwise_p = torch.einsum("bhd,bge->bhgde", probs_comm, probs_comm)  # B, H, H, D, D
        pointwise_p = rearrange(pointwise_p, "b h g d e -> b (h g) (d e)")  # B, (H, H), (D, D)
        kl_computed = pointwise_p * (pointwise_p.log() - marginal_p.log())
        kl_grid = rearrange(kl_computed.sum(-1), "b (h g) -> b h g", h=nh)  # B, H, H
        consensus = torch.triu(kl_grid, diagonal=1).mean([-1, -2])
    return consensus

def evaluate_recall(args, net, dataset_dict, extended_test=False, testset="test"):
    if testset == "unlabeled":
        test_loader = DataLoader(dataset_dict['train_ulb'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    elif testset == "test":
        test_loader = DataLoader(dataset_dict['test']['full'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
        if extended_test:
            test_loader = DataLoader(dataset_dict['test']['extended'], batch_size=1024, drop_last=False, shuffle=False, num_workers=4)
    labels_all = []
    preds_all = []
    probs_all = []
    scores_all = []
    o_scores_all = []
    preds_open_all = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            if testset == "unlabeled":
                if args.algorithm == 'mtc' or args.algorithm == 'openmatch':
                    x = data['x_ulb_w_0']
                else:
                    x = data['x_ulb_w']
                y = data['y_ulb']
            elif testset == "test":
                x = data['x_lb']
                y = data['y_lb']

            if isinstance(x, dict):
                x = {k: v.cuda() for k, v in x.items()}
            else:
                x = x.cuda()
            y = y.cuda()

            outputs = net(x)
            logits = outputs['logits']
            max_probs_idx, preds_idx = torch.max(F.softmax(logits, dim=-1), dim=-1)

            if args.algorithm == 'openmatch':
                logits_open = outputs['logits_open']
                probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda()
                score_idx = probs_open[tmp_range, 0, preds_idx]

            elif args.algorithm == 'dac':
                logits_comm = outputs['logits_comm'].view(-1, args.num_heads, args.num_classes)
                probs_comm = torch.softmax(logits_comm, dim=-1)
                score_idx = calculate_consensus(probs_comm)
                o_score = probs_comm.mean(dim=1).max(dim=-1)[0]
                o_scores_all.extend(o_score.cpu().tolist())

            else:
                score_idx = max_probs_idx

            labels_all.extend(y.cpu().tolist())
            preds_all.extend(preds_idx.cpu().tolist())
            probs_all.extend(max_probs_idx.cpu().tolist())
            scores_all.extend(score_idx.cpu().tolist())

        labels = torch.Tensor(labels_all).long()
        id_mask = labels.lt(args.num_classes)
        ood_mask = labels.ge(args.num_classes)

        preds = torch.Tensor(preds_all).long()
        probs = torch.Tensor(probs_all).float()
        scores = torch.Tensor(scores_all).float()

        labels[ood_mask] = args.num_classes
        true_mask_close = (labels == preds)

        if args.algorithm == 'openmatch':
            type = "otsu"
            if type == "binary":
                pred_pos_mask = scores.lt(0.5)
                pred_neg_mask = scores.ge(0.5)
                preds_open = preds.clone()
                preds_open[pred_neg_mask] = args.num_classes
            elif type == "otsu":
                otsu_th = threshold_otsu(scores.cpu().numpy())
                pred_pos_mask = scores.lt(otsu_th)
                pred_neg_mask = scores.ge(otsu_th)
                preds_open = preds.clone()
                preds_open[pred_neg_mask] = args.num_classes

        elif args.algorithm == 'dac':
            c_scores = scores.clone()
            o_scores = torch.Tensor(o_scores_all).float()
            # use_norm = False
            use_norm = True
            if use_norm:
                c_scores = ((c_scores - c_scores.min()) / (c_scores.max() - c_scores.min() + 1e-8)).abs()
                o_scores = ((o_scores - o_scores.min()) / (o_scores.max() - o_scores.min() + 1e-8)).abs()
            # joint_scores = True
            joint_scores = False
            if joint_scores:
                scores = c_scores * o_scores
            else:
                scores = c_scores
            otsu_th = threshold_otsu(scores.cpu().numpy())
            print(f"OTSU_THRESHOLD_VALUE: {otsu_th}")
            hard_pos_mask = scores.ge(otsu_th)
            hard_neg_mask = scores.lt(otsu_th)
            soft_mask = scores / (otsu_th + 1e-8)
            temp_w = 0.1
            soft_pos_mask = torch.clamp(soft_mask ** temp_w, max=1.0)
            preds_open = preds.clone()
            preds_open[hard_neg_mask] = args.num_classes

        else:
            pred_pos_mask = scores.ge(0.95)
            pred_neg_mask = scores.lt(0.95)
            preds_open = preds.clone()
            preds_open[pred_neg_mask] = args.num_classes
        # 
        np.set_printoptions(precision=3, suppress=True)
        close_acc = accuracy_score(labels[id_mask].cpu().numpy(), preds[id_mask].cpu().numpy())
        open_acc = balanced_accuracy_score(labels.cpu().numpy(), preds_open.cpu().numpy())
        # 
        if args.algorithm == 'dac':
            print(f"#############################################################\n"
                f" Method:               {args.algorithm}\n"
                f" Dataset:              {args.dataset}\n"
                f" Num_classes:          {args.num_classes}\n"
                f" Num_labels:           {args.num_labels}\n"
                f" Closed-set Accuracy:  {close_acc * 100:.2f}\n"
                f" Open-set Accuracy:    {open_acc * 100:.2f}\n"
                f"#############################################################\n"
                )
        else:
            print(f"#############################################################\n"
              f" Method:               {args.algorithm}\n"
              f" Dataset:              {args.dataset}\n"
              f" Num_classes:          {args.num_classes}\n"
              f" Num_labels:           {args.num_labels}\n"
              f" Closed-set Accuracy:  {close_acc * 100:.2f}\n"
              f" Open-set Accuracy:    {open_acc * 100:.2f}\n"
              f"#############################################################\n"
            )


""" ABCD """
# seed_list = [0, 1, 2]
# dataset_list = ["cifar10", "cifar100", "imagenet30"]
seed_list = [1]
dataset_list = ["cifar100"]
algorithm_list = ["dac"]

for seed in seed_list:
    for aidx in algorithm_list:
        for didx in dataset_list:
            # 
            if didx == "cifar10":
                num_class_list = [6]
                num_label_list = [5, 10, 25]
            elif didx == "cifar100":
                num_class_list = [20, 50]
                num_label_list = [5, 10, 25]
            elif didx == "imagenet30":
                num_class_list = [20]
                num_label_list = ["p1", "p5"]
            # 
            for num_class in num_class_list:
                for num_label in num_label_list:
                    if type(num_label) == int:
                        nidx = num_label * num_class
                    else:
                        pidx = num_label

                    if didx == "imagenet30":
                        args = parser.parse_args(args=['--c', f'config/openset_cv/{aidx}/{aidx}_in30_{pidx}_{seed}.yaml'])
                    else:
                        args = parser.parse_args(args=['--c', f'config/openset_cv/{aidx}/{aidx}_{didx}_{num_class}_{nidx}_{seed}.yaml'])

                    args.correlated_ood = False
                    args.mm_ablation = False
                    args.ratio = None
                    args.num_heads = 10
                    over_write_args_from_file(args, args.c)
                    args.data_dir = "data"
                    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, eval_open=True)
                    base_root = "./saved_models"
                    if didx == "imagenet30":
                        args.load_path = f"{base_root}/openset_cv/{aidx}/{aidx}_in30_{pidx}_{seed}/model_best.pth"
                    else:
                        args.load_path = f"{base_root}/openset_cv/{aidx}/{aidx}_{didx}_{num_class}_{nidx}_{seed}/model_best.pth"
                    if not os.path.exists(args.load_path):
                        continue
                    best_net = load_model_at('best')
                    evaluate_recall(args, best_net, dataset_dict, extended_test=False)
