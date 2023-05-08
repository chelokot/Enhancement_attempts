# -*- coding: utf-8 -*-

import os.path as osp

import cv2
import ipdb
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import LCDPNet.src.utils.util as util
from LCDPNet.src.globalenv import *
from .basemodel import BaseModel


# dict_merge = lambda a, b: a.update(b) or a


class SingleNetBaseModel(BaseModel):
    # for models with only one self.net
    def __init__(self, opt, net, running_modes, valid_ssim=False, print_arch=True):
        super().__init__(opt, running_modes)
        self.net = net
        self.net.train()

        self.valid_ssim = valid_ssim  # weather to compute ssim in validation
        self.tonemapper = cv2.createTonemapReinhard(2.2)

        self.psnr_func = torchmetrics.PeakSignalNoiseRatio(data_range=1)
        self.ssim_func = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1)

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR])

        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [schedular]

    def forward(self, x):
        return self.net(x)

    def training_step_forward(self, batch, batch_idx):
        if not self.MODEL_WATCHED and not self.opt[DEBUG] and self.opt.logger == 'wandb':
            self.logger.experiment.watch(
                self.net, log_freq=self.opt[LOG_EVERY] * 2, log_graph=True)
            self.MODEL_WATCHED = True
            # self.show_flops_and_param_num([batch[INPUT]])

        input_batch, gt_batch = batch[INPUT], batch[GT]
        output_batch = self(input_batch)
        self.iogt = {
            INPUT: input_batch,
            OUTPUT: output_batch,
            GT: gt_batch,
        }
        return input_batch, gt_batch, output_batch

    def validation_step(self, batch, batch_idx):
        input_batch, gt_batch = batch[INPUT], batch[GT]
        output_batch = self(input_batch)

        # log psnr
        output_ = util.cuda_tensor_to_ndarray(output_batch)
        y_ = util.cuda_tensor_to_ndarray(gt_batch)
        try:
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
        except:
            ipdb.set_trace()
        self.log(PSNR, psnr)

        # log SSIM (optional)
        if self.valid_ssim:
            ssim = util.ImageProcessing.compute_ssim(output_batch, gt_batch)
            self.log(SSIM, ssim)

        # log images
        if self.global_valid_step % self.opt.log_every == 0:
            self.log_images_dict(
                VALID,
                osp.basename(batch[INPUT_FPATH][0]),
                {
                    INPUT: input_batch,
                    OUTPUT: output_batch,
                    GT: gt_batch,
                },
                gt_fname=osp.basename(batch[GT_FPATH][0])
            )
        self.global_valid_step += 1
        return output_batch

    def log_training_iogt_img(self, batch, extra_img_dict=None):
        """
        Only used in training_step
        """
        if extra_img_dict:
            img_dict = {**self.iogt, **extra_img_dict}
        else:
            img_dict = self.iogt

        if self.global_step % self.opt.log_every == 0:
            self.log_images_dict(
                TRAIN,
                osp.basename(batch[INPUT_FPATH][0]),
                img_dict,
                gt_fname=osp.basename(batch[GT_FPATH][0])
            )

    @staticmethod
    def logdomain2hdr(ldr_batch):
        return 10 ** ldr_batch - 1

    def on_test_start(self):
        self.total_psnr = 0
        self.total_ssim = 0
        self.global_test_step = 0

    def on_test_end(self):
        pass

    def test_step(self, batch, batch_ix):
        """
        save test result and calculate PSNR and SSIM for `self.net` (when have GT)
        """
        # test without GT image:
        assert batch.shape[0] == 1
        self(batch)
