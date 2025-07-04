import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import Where2comm
from opencood.utils.transformation_utils import normalize_pairwise_tfm

class PointPillarWhere2comm(nn.Module):
    """
    Where2comm implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = Where2comm(args['where2comm'], dim=args['feat_dim'])
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

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
        
        self.comm_sum = 0.0
        self.comm_n = 0

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

    def forward(self, data_dict, fusion_flag=True):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        
        batch_dict_no_fusion = self.backbone(batch_dict)
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features_2d = batch_dict_no_fusion['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        psm_single = self.cls_head(spatial_features_2d)
        
        comm_sum = 0.0
        comm_now = 0.0
        
        _, CC, HH, WW = spatial_features_2d.shape
        fused_feature, comm_rate = self.fusion_net(spatial_features_2d, psm_single,
                                                    record_len, normalized_affine_matrix)
        comm_now += comm_rate * CC * HH * WW
        comm_sum += CC * HH * WW
        self.comm_n += 1
        self.comm_sum += comm_now / comm_sum

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})
        
        # if self.comm_n == 2170 :
        #     print(f"comm_rate = {self.comm_sum / self.comm_n}")
        
        return output_dict
