import os
import copy
from PIL import Image
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import get_data_loader
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss,  SSL_Argument, str2bool, concat_all_gather
from .utils import SoftWeightingHook, DiverseLoss

from semilearn.core.utils import ALGORITHMS


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class OSCDataset(BasicDataset):
    def __init__(self, dset, name):
        self.data = copy.deepcopy(dset.data)
        self.targets = copy.deepcopy(dset.targets)
        super(OSCDataset, self).__init__(alg='dac', data=self.data, targets=self.targets, num_classes=dset.num_classes,
                                         transform=dset.transform, strong_transform=dset.strong_transform)
        self.name = name
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        target = self.targets_index[idx] if self.targets is not None else None
        img = self.data_index[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # shape of img should be [H, W, C]
        if isinstance(img, str):
            img = pil_loader(img)
        return img, target

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)
        img_w = self.transform(img)
        if self.name == 'train_lb':
            return {'idx_lb': idx, 'x_lb': img_w, 'x_lb_w_0': img_w, 'x_lb_w_1': self.transform(img), 'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img), 'y_ulb': target}


class OSCNet(nn.Module):
    def __init__(self, base, num_classes, num_heads, proj_dim, use_rot=False):
        super(OSCNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        # 
        self.proj_dim = proj_dim
        self.proj = nn.Sequential(
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_planes, self.proj_dim),)
        self.fc_comm = nn.Linear(self.proj_dim, num_classes * num_heads)
        nn.init.xavier_normal_(self.fc_comm.weight.data)
        self.fc_comm.bias.data.zero_()
        # 
        use_norm = True
        self.norm = nn.LayerNorm(self.feat_planes) if use_norm else nn.Identity()
        # 
        self.use_rot = use_rot
        if self.use_rot:
            self.rot_classifier = nn.Linear(self.feat_planes, 4, bias=False)
            nn.init.xavier_normal_(self.rot_classifier.weight.data)

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x):
        emb = self.backbone(x, only_feat=True)
        emb = self.norm(emb)
        logits = self.backbone(emb, only_fc=True)
        logits_comm = self.fc_comm(self.proj(emb.detach()))
        emb_proj = self.proj(emb)
        return_dict = {'logits': logits, 'logits_comm': logits_comm, 'feat': self.l2norm(emb), 'feat_proj': self.l2norm(emb_proj)}
        # 
        if self.use_rot:
            logits_rot = self.rot_classifier(emb)
            return_dict['logits_rot'] = logits_rot
        return return_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('dac')
class DAC(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.div_criterion = DiverseLoss()

    def set_dataset(self):
        dataset_dict = super(DAC, self).set_dataset()
        dataset_dict['train_lb'] = OSCDataset(dset=dataset_dict['train_lb'], name='train_lb')
        dataset_dict['train_ulb'] = OSCDataset(dset=dataset_dict['train_ulb'], name='train_ulb')
        return dataset_dict

    def set_hooks(self):
        self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'), "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(SoftWeightingHook(num_data=len(self.dataset_dict['train_ulb']), ema_alpha=self.args.ema_alpha, temp_d=self.args.temp_d, 
                                             use_joint=self.args.use_joint, device=self.gpu, temp_w=self.args.temp_w), "WeightingHook")
        # memory bank to store the embeding for labeled & unlabeled data
        self.queue_size = int(self.args.K * (self.args.uratio) * self.args.batch_size) if self.args.dataset != 'imagenet' else self.args.K
        self.u_feats_bank = torch.randn(self.queue_size, self.args.proj_dim).to(self.gpu)
        self.u_probs_bank = torch.zeros(self.queue_size, self.num_classes + 1).to(self.gpu) / (self.num_classes+1)
        self.ptr = torch.zeros(1, dtype=torch.long).to(self.gpu)
        self.u_feats_bank = nn.functional.normalize(self.u_feats_bank)
        # 
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = OSCNet(model, num_classes=self.num_classes, num_heads=self.args.num_heads, proj_dim=self.args.proj_dim, use_rot=self.args.use_rot,)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = OSCNet(ema_model, num_classes=self.num_classes, num_heads=self.args.num_heads, proj_dim=self.args.proj_dim, use_rot=self.args.use_rot,)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def _update_bank(self, feats, probs):
        feats = concat_all_gather(feats)
        probs = concat_all_gather(probs)
        batch_size = feats.size(0)
        ptr = int(self.ptr[0])
        assert self.queue_size % batch_size == 0
        self.u_feats_bank[ptr:ptr + batch_size] = feats
        self.u_probs_bank[ptr:ptr + batch_size] = probs
        self.ptr[0] = (ptr + batch_size) % self.queue_size

    def train(self):
        """ train function """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb'],):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            self.call_hook("after_train_epoch")
            # if self.epoch % 5 == 0 and self.epoch != 0:
            #     save_path = os.path.join(self.save_dir, self.save_name)
            #     self.save_model(f'model_checkpoint_{self.epoch}.pth', save_path)
        self.call_hook("after_run")


    def train_step(self, x_lb_w_0, x_lb_w_1, y_lb, x_ulb_w, x_ulb_s, idx_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w_0, x_lb_w_1, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb * 2]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb * 2:].chunk(2)
                logits_comm = outputs['logits_comm'].view(-1, self.args.num_heads, self.args.num_classes)
                logits_comm_x_lb = logits_comm[:num_lb * 2]
                logits_comm_x_ulb_w, logits_comm_x_ulb_s = logits_comm[num_lb * 2:].chunk(2)
                feat_proj_x_ulb_w, feat_proj_x_ulb_s = outputs['feat_proj'][num_lb * 2:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            ## SUPERIVSED LOSS FOR LABELED SAMPLES ##
            loss_s = ce_loss(logits_x_lb, y_lb.repeat(2), reduction='mean')
            loss_unk_s = 0.
            for hidx in range(self.args.num_heads):
                loss_unk_s += ce_loss(logits_comm_x_lb[:,hidx,:], y_lb.repeat(2), reduction='mean')
            loss_unk_s /= self.args.num_heads

            ## WEIGHT UPDATE ##
            weight = self.call_hook("weighting", "WeightingHook", logits_comm=logits_comm_x_ulb_w.detach(), idx=idx_ulb)
            with torch.no_grad():
                targets_p = F.softmax(logits_x_ulb_w, dim=-1).detach()
                targets_comm = F.softmax(logits_comm_x_ulb_w.view(-1, self.args.num_classes), dim=-1).detach()
                if self.args.dist_align:
                    targets_p = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p)
                    targets_comm = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_comm)
                targets_open = torch.einsum('bc, b -> bc', targets_p, weight)
                targets_open = torch.cat([targets_open, (1. - weight).abs().unsqueeze(-1)], dim=1)

            ## DIVERSIFYING LOSS FOR PSEUDO_OUTLIERS ##
            loss_mi = self.div_criterion(logits_comm_x_ulb_w)

            ## REGULARIZATION LOSS FOR UNLABELED DATA ##
            mask_comm = self.call_hook("masking", "MaskingHook", cutoff=self.args.c_cutoff, logits_x_ulb=targets_comm, softmax_x_ulb=False)
            yh_comm_ulb = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=targets_comm, use_hard_label=True, softmax=False)
            loss_unk_reg = consistency_loss(logits_comm_x_ulb_s.view(-1, self.args.num_classes), yh_comm_ulb, 'ce', mask=mask_comm)

            ### TRAIN UNSUPERVISED LOSS ###
            if self.epoch >= self.args.start_fix:
                # GET SOFT-MASK
                mask_id, weight = self.call_hook("masking", "WeightingHook", idx=idx_ulb)

                ## ##
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                mask_p = self.call_hook("masking", "MaskingHook", cutoff=self.args.p_cutoff, logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                yh_ulb = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=True, softmax=False)
                if self.args.mask_type == 'hard':
                    loss_u = consistency_loss(logits_x_ulb_s, yh_ulb, 'ce', mask=mask_p * mask_id)
                elif self.args.mask_type == 'soft':
                    loss_u = consistency_loss(logits_x_ulb_s, yh_ulb, 'ce', mask=mask_p * weight)

                ## FOR OPEN_SET CLASSIFIER ##
                u_feats_bank = self.u_feats_bank.clone().detach()
                u_probs_bank = self.u_probs_bank.clone().detach()
                relation_qu = F.softmax(feat_proj_x_ulb_s @ u_feats_bank.T / self.args.temp_s, dim=-1)
                nn_qu = relation_qu @ u_probs_bank
                loss_kd = torch.sum(-nn_qu.log() * targets_open.detach(), dim=1).mean()

            else:
                mask_p = torch.zeros(num_ulb).to(self.gpu)
                loss_u = torch.tensor(0).to(self.gpu)
                loss_kd = torch.tensor(0).to(self.gpu)

            loss_unk = loss_unk_s + self.args.lambda_reg * loss_unk_reg + self.args.lambda_mi * loss_mi + self.args.lambda_kd * loss_kd
            total_loss = loss_s + self.args.lambda_u * loss_u + loss_unk 

        self._update_bank(feat_proj_x_ulb_w, targets_open)
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {'train/sup_loss': loss_s.item(),
                    'train/unsup_loss': loss_u.item(),
                    'train/total_loss': total_loss.item(), 
                    'train/p_mask_ratio': mask_p.float().mean().item(),
                    'train/comm_mask_ratio': mask_comm.float().mean().item(),
                    'train/weighted_mask': (mask_p * weight).float().mean().item(),
                    }
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            # [OPTIONAL]
            SSL_Argument('--dist_align', str2bool, False),
            SSL_Argument('--da_len', int, 128),
            SSL_Argument('--use_rot', str2bool, False),
            # [COEFFICIENTS]
            SSL_Argument('--lambda_u', float, 1.),
            SSL_Argument('--lambda_reg', float, 1.),
            SSL_Argument('--lambda_mi', float, 1.),
            SSL_Argument('--lambda_kd', float, 1.),
            # [STRUCTURES]
            SSL_Argument('--num_heads', int, 10),
            SSL_Argument('--proj_dim', int, 128),
            SSL_Argument('--use_joint', str2bool, True),
            SSL_Argument('--mask_type', str, 'soft'),
            # [HYPER_PARAMETERS-1]
            SSL_Argument('--start_fix', int, 0),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--c_cutoff', float, 0.95),
            # [HYPER_PARAMETERS-2]
            SSL_Argument('--ema_alpha', float, 0.9),
            SSL_Argument('--temp_d', float, 1.),
            SSL_Argument('--temp_w', float, 1.),
            # [HYPER_PARAMETERS-3]
            SSL_Argument('--temp_s', float, 0.1),
            SSL_Argument('--K', int, 256),
        ]
