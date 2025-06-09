import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter, SimplePointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import Where2comm
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class PointPillarCommMultiscale(nn.Module):
    """
    Where2comm implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarCommMultiscale, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        
        self.simple_scatter = SimplePointPillarScatter(
            feature_dim=1, grid_size=args['point_pillar_scatter']['grid_size'])
        
        self.req_points_threshold = -1
        if 'req_points_threshold' in args:
            self.req_points_threshold = args['req_points_threshold']
            print(f"req_points_threshold = {args['req_points_threshold']}")
        
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            self.fusion_net.append(Where2comm(args['where2comm'], dim=args['feat_dim'][i]))
        
        if 'k_ratio' in args['where2comm']['communication']:
            print(f"k_ratio: {args['where2comm']['communication']['k_ratio']}")
        elif 'threshold' in args['where2comm']['communication']:
            print(f"threshold: {args['where2comm']['communication']['threshold']}")
        
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        self.compression = False
        self.compression_ratio = 1
        if "compression" in args:
            self.compression = True
            self.compression_ratio = args['compression']
            self.naive_compressor_list = nn.ModuleList()
            for i in range(len(args['feat_dim'])):
                self.naive_compressor_list.append(NaiveCompressor(args['feat_dim'][i],
                                                                    args['compression']))
            print(f"compression_ratio: {self.compression_ratio}")

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        
        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
        print("Backbone fixed.")
    
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        
        ego_flag = True
        if '_ego_flag' in data_dict:
            # ego_flag indicates whether the data is from ego vehicle
            # in reference, if not ego_flag, no need to fuse feature, just use the single detection results
            ego_flag = data_dict['_ego_flag']
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        ###############################################
        # for intermediate-late fusion, get the single detection results
        if not ego_flag and not self.training:
            no_fusion_feature_list = self.backbone.get_multiscale_feature(batch_dict['spatial_features'])
            no_fusion_feature_after_compress = []
            for i, fuse_module in enumerate(self.fusion_net):
                feature_i = no_fusion_feature_list[i]
                if self.compression:
                    feature_i = self.naive_compressor_list[i](feature_i, use_fp16=False)
                no_fusion_feature_after_compress.append(feature_i)
            spatial_features_2d = self.backbone.decode_multiscale_feature(no_fusion_feature_after_compress)
            
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
            spatial_features_2d = spatial_features_2d[0].unsqueeze(0)
        
            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)

            output_dict = {'cls_preds': psm,
                        'reg_preds': rm}

            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})
            
            return output_dict
        ###############################################
        
        req_mask = None
        if not self.training and self.req_points_threshold > 0:
            # n, -> N, 1, H, W
            points_map = self.simple_scatter(voxel_num_points.unsqueeze(1), voxel_coords).float()
            smoothed_points_map = points_map
            req_mask = (smoothed_points_map < self.req_points_threshold).float()
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']
        spatial_features_2d_single = batch_dict['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d_single = self.shrink_conv(spatial_features_2d_single)
        
        psm_single = self.cls_head(spatial_features_2d_single)
        
        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        
        for i, fuse_module in enumerate(self.fusion_net):
            feature_i = feature_list[i]
            if self.compression:
                feature_i = self.naive_compressor_list[i](feature_i, use_fp16=not self.training)
            x_out, _ = fuse_module(feature_i, psm_single,
                                                record_len, normalized_affine_matrix, req_mask,)

            fused_feature_list.append(x_out)
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})
        
        return output_dict
