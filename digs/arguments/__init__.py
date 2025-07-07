#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from digs.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.00005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = False
        self.v_grid_spacing = [8,8,8]
        self.t_grid_spacing = 4
        self.num_rank = 2
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        self.unified = False
        self.no_bspline = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 50_000
        self.coarse_iterations = 0
        self.loaded_iteration = -2
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = self.iterations
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = self.iterations
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = self.iterations
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = self.iterations
        self.v_grid_lr_init = 0.0001
        self.v_grid_lr_final = 0.00001
        self.v_grid_lr_max_steps = self.iterations
        self.t_grid_lr_init = 0.01
        self.t_grid_lr_final = 0.001
        self.t_grid_lr_max_steps = self.iterations
        self.loss_type = "l2_loss"
        self.lambda_dssim = 0.0
        self.lambda_tv = 0.0
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 1e-8
        self.densify_grad_threshold_init = 1e-8
        self.densify_grad_threshold_final = 2e-9
        self.densify_scale_threshold = 0.01  # percent of volume size
        self.max_screen_size = 0.0
        self.max_scale = 0.0  # percent of volume size
        self.max_num_gaussians = 500_000
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
