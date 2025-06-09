# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json


def getIntermediatelateFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateLateFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False)
            self.confidence_beta = params['postprocess'].get('confidence_beta', None)
            self.confidence_threshold = params['postprocess'].get('confidence_threshold', None)
            print("confidence_beta: ", self.confidence_beta)
            print("confidence_threshold: ", self.confidence_threshold)

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']
                


        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_cav_base : dict
                The dictionary contains the ego CAV's raw information.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )

            selected_cav_processed.update({
                                "single_label_dict": label_dict,
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )


            return selected_cav_processed
        

        # When training, just select one ego to train normally, that is, only intermediate collaboration during training
        # When inference, Ego use intermediate-late collaboration, and other CAVs send detection results without fusion to Ego to reduce the latency
        def get_item_train(self, base_data_dict, idx):
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            processed_features = []
            object_stack = []
            object_id_stack = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                    
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                if self.load_lidar_file:
                    # include voxel_features, voxel_coords, voxel_num_points
                    processed_features.append(
                        selected_cav_processed['processed_features'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])

            # exclude all repetitive objects    
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict

        
        def get_item_test(self, base_data_dict, idx):
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            main_ego_id = -1
            main_ego_content = None
            for ego_id, ego_content in base_data_dict.items():
                if main_ego_id == -1:
                    main_ego_id = ego_id
                    main_ego_content = ego_content
                    main_ego_lidar_pose = main_ego_content['params']['lidar_pose']
                    main_ego_lidar_pose_clean = main_ego_content['params']['lidar_pose_clean']
                ego_id_for_dict = 'ego' if ego_id == main_ego_id else ego_id
                processed_data_dict[ego_id_for_dict] = {}
                
                ego_lidar_pose = ego_content['params']['lidar_pose']
                ego_lidar_pose_clean = ego_content['params']['lidar_pose_clean']
                ego_cav_base = ego_content

                processed_features = []
                object_stack = []
                object_id_stack = []
                too_far = []
                lidar_pose_list = []
                lidar_pose_clean_list = []
                cav_id_list = []

                if self.visualize or self.kd_flag:
                    projected_lidar_stack = []

                # make sure the first one is ego
                base_data_dict_copy = OrderedDict()
                base_data_dict_copy[ego_id] = base_data_dict[ego_id]
                for cav_id, selected_cav_base in base_data_dict.items():
                    if cav_id == ego_id:
                        continue
                    base_data_dict_copy[cav_id] = selected_cav_base
                    
                for cav_id, selected_cav_base in base_data_dict_copy.items():
                    # check if the cav is within the communication range with ego
                    distance = \
                        math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                                ego_lidar_pose[0]) ** 2 + (
                                        selected_cav_base['params'][
                                            'lidar_pose'][1] - ego_lidar_pose[
                                            1]) ** 2)

                    # if distance is too far, we will just skip this agent
                    if distance > self.params['comm_range']:
                        too_far.append(cav_id)
                        continue

                    lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                    lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                    cav_id_list.append(cav_id)   

                for cav_id in too_far:
                    base_data_dict_copy.pop(cav_id)

                pairwise_t_matrix = \
                    get_pairwise_transformation(base_data_dict_copy,
                                                    self.max_cav,
                                                    self.proj_first)

                lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
                lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
                
                # merge preprocessed features from different cavs into the same dict
                cav_num = len(cav_id_list)
                
                for _i, cav_id in enumerate(cav_id_list):
                    selected_cav_base = base_data_dict_copy[cav_id]
                    selected_cav_processed = self.get_item_single_car(
                        selected_cav_base,
                        ego_cav_base)
                    
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']
                    if self.load_lidar_file:
                        # 包含voxel_features, voxel_coords, voxel_num_points
                        processed_features.append(
                            selected_cav_processed['processed_features'])

                    if self.visualize or self.kd_flag:
                        projected_lidar_stack.append(
                            selected_cav_processed['projected_lidar'])

                # exclude all repetitive objects
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

                # make sure bounding boxes across all frames have the same number
                object_bbx_center = \
                    np.zeros((self.params['postprocess']['max_num'], 7))
                mask = np.zeros(self.params['postprocess']['max_num'])
                object_bbx_center[:object_stack.shape[0], :] = object_stack
                mask[:object_stack.shape[0]] = 1
                
                if self.load_lidar_file:
                    merged_feature_dict = merge_features_to_dict(processed_features)
                    processed_data_dict[ego_id_for_dict].update({'processed_lidar': merged_feature_dict})

                # generate targets label
                label_dict = \
                    self.post_processor.generate_label(
                        gt_box_center=object_bbx_center,
                        anchors=self.anchor_box,
                        mask=mask)

                transformation_matrix = x1_to_x2(ego_lidar_pose, main_ego_lidar_pose)
                transformation_matrix_clean = x1_to_x2(ego_lidar_pose_clean, main_ego_lidar_pose_clean)
                
                processed_data_dict[ego_id_for_dict].update(
                    {'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': mask,
                    'object_ids': [object_id_stack[i] for i in unique_indices],
                    'anchor_box': self.anchor_box,
                    'label_dict': label_dict,
                    'cav_num': cav_num,
                    'pairwise_t_matrix': pairwise_t_matrix,
                    'lidar_poses_clean': lidar_poses_clean,
                    'lidar_poses': lidar_poses,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean,
                    })

                if self.visualize:
                    processed_data_dict[ego_id_for_dict].update({'origin_lidar':
                        np.vstack(
                            projected_lidar_stack)})

                processed_data_dict[ego_id_for_dict].update({'sample_idx': idx,
                                                    'cav_id_list': cav_id_list})

            return processed_data_dict

        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])

                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            return output_dict

        
        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            if self.train:
                return self.get_item_train(base_data_dict, idx)
            else:
                return self.get_item_test(base_data_dict, idx)
        
        
        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            # Intermediate fusion is different the other two
            output_dict = {}
            for ego_id, _ in batch[0].items():
                output_dict[ego_id] = {}
                object_bbx_center = []
                object_bbx_mask = []
                object_ids = []
                processed_lidar_list = []
                # used to record different scenario
                record_len = []
                label_dict_list = []
                lidar_pose_list = []
                origin_lidar = []
                lidar_pose_clean_list = []
                # pairwise transformation matrix
                pairwise_t_matrix_list = []

                for i in range(len(batch)):
                    ego_dict = batch[i][ego_id]
                    object_bbx_center.append(ego_dict['object_bbx_center'])
                    object_bbx_mask.append(ego_dict['object_bbx_mask'])
                    object_ids.append(ego_dict['object_ids'])
                    lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                    lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                    if self.load_lidar_file:
                        processed_lidar_list.append(ego_dict['processed_lidar'])

                    record_len.append(ego_dict['cav_num'])
                    label_dict_list.append(ego_dict['label_dict'])
                    pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                    if self.visualize:
                        origin_lidar.append(ego_dict['origin_lidar'])

                # convert to numpy, (B, max_num, 7)
                object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
                object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

                if self.load_lidar_file:
                    merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(merged_feature_dict)
                    output_dict[ego_id].update({'processed_lidar': processed_lidar_torch_dict})

                record_len = torch.from_numpy(np.array(record_len, dtype=int))
                lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
                lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
                label_torch_dict = \
                    self.post_processor.collate_batch(label_dict_list)

                # for centerpoint
                label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask})

                # (B, max_cav)
                pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

                # add pairwise_t_matrix to label dict
                label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
                label_torch_dict['record_len'] = record_len
                
                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(
                        np.array(batch[0][ego_id]['transformation_matrix'])).float()
                # clean transformation matrix
                transformation_matrix_clean_torch = \
                    torch.from_numpy(
                        np.array(batch[0][ego_id]['transformation_matrix_clean'])).float()
                
                # object id is only used during inference, where batch size is 1.
                # so here we only get the first element.
                output_dict[ego_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'record_len': record_len,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids[0],
                                        'pairwise_t_matrix': pairwise_t_matrix,
                                        'lidar_pose_clean': lidar_pose_clean,
                                        'lidar_pose': lidar_pose,
                                        'anchor_box': self.anchor_box_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[ego_id].update({'origin_lidar': origin_lidar})
                
                # transformation matrix is used to transform the bbox
                # pairwise_t_matrix is used to transform the features
                output_dict[ego_id].update({'transformation_matrix':
                                            transformation_matrix_torch,
                                            'transformation_matrix_clean':
                                            transformation_matrix_clean_torch,})

                output_dict[ego_id].update({
                    "sample_idx": batch[0][ego_id]['sample_idx'],
                    "cav_id_list": batch[0][ego_id]['cav_id_list']
                })

            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            data_dict_box = OrderedDict()
            for cav_id, cav_content in data_dict.items():
                distance = \
                    math.sqrt(cav_content['transformation_matrix'][0][3] ** 2 +
                              cav_content['transformation_matrix'][1][3] ** 2)
                if distance <= self.params['comm_range']:
                    data_dict_box[cav_id] = cav_content
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_box, output_dict, confidence_beta=self.confidence_beta,
                                                confidence_threshold=self.confidence_threshold)
            
            data_dict_gt = OrderedDict()
            data_dict_gt['ego'] = data_dict['ego']
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict_gt, project_to_ego=False)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateLateFusionDataset



