import argparse
import os
import statistics
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--backbone_dir', default='', help='which backbone to use')
    parser.add_argument('--backbone_epoch', required=True, type=int, help='which backbone to use')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
    
    ###################################### load backbone ######################################
    pretrain_yaml = yaml_utils.load_yaml(f"opencood/logs/{opt.backbone_dir}/config.yaml")
    pretrain_model = train_utils.create_model(pretrain_yaml)
    _, pretrain_model = train_utils.load_saved_model(
        f"opencood/logs/{opt.backbone_dir}", pretrain_model, epoch=opt.backbone_epoch)

    backbone_state_dict = pretrain_model.backbone.state_dict()
    model.backbone.load_state_dict(backbone_state_dict)
    
    pillar_vfe_state_dict = pretrain_model.pillar_vfe.state_dict()
    model.pillar_vfe.load_state_dict(pillar_vfe_state_dict)
    
    scatter_state_dict = pretrain_model.scatter.state_dict()
    model.scatter.load_state_dict(scatter_state_dict)
    
    if 'shrink_header' in hypes['model']['args']:
        shrink_conv_state_dict = pretrain_model.shrink_conv.state_dict()
        model.shrink_conv.load_state_dict(shrink_conv_state_dict)
    
    cls_head_state_dict = pretrain_model.cls_head.state_dict()
    model.cls_head.load_state_dict(cls_head_state_dict)
    
    reg_head_state_dict = pretrain_model.reg_head.state_dict()
    model.reg_head.load_state_dict(reg_head_state_dict)
    
    if 'dir_args' in hypes['model']['args']:
        dir_head_state_dict = pretrain_model.dir_head.state_dict()
        model.dir_head.load_state_dict(dir_head_state_dict)
    ###########################################################################################

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    from torchinfo import summary
    summary(model)
    
    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()


        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step()
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()

