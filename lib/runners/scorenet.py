import os
import sys
import logging
import time
import glob
import json
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pickle
import torch
import numpy as np
import shutil
import os.path as osp
from tqdm import tqdm
import open3d as o3d
import torch.utils.data as data
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record

from utils.filter_hub import *
from utils.pose_utils import *
from utils.diff_utils import *
from utils.draw import render_mesh, render_side_mesh, render_side_mesh_, render_mesh_fast
from utils.function import get_optimizer, get_model, get_model_score, get_dataloader, process_pred
from utils.relation import *
from utils.cache import Cache
from models.layers.smpl.SMPL import SMPL_layer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

executor = ThreadPoolExecutor(max_workers=8)  # 可根据CPU核数调大
def fast_save(path, img):
    executor.submit(cv2.imwrite, path, img)

def copy_with_structure(src_file: Path, root_dir: Path, dest_root: Path):
    """
    把 src_file 从 root_dir 相对路径的结构复制到 dest_root 下。
    """
    # 计算相对路径
    rel_path = src_file.relative_to(root_dir)
    dest_file = dest_root / rel_path
    
    # 创建目标文件夹
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    shutil.copy2(src_file, dest_file)
    print(f"Copied: {src_file} -> {dest_file}")


def write_obj(path,point,face):
    with open(path,'w') as fid:
        for p in point:
            fid.write('v {} {} {}\n'.format(str(p[0]),str(p[1]),str(p[2])))
        for f in face:
            fid.write('f {} {} {}\n'.format(str(f[0]+1),str(f[1]+1),str(f[2]+1)))


class DeltaDepth(nn.Module):
    def __init__(self,bs):
        super(DeltaDepth, self).__init__()
        self.delta_d = nn.Parameter(torch.zeros(bs,1),requires_grad=True)
    def forward(self,depth):
        return depth+self.delta_d

# def get_estimates(labels, output, smpl):
#     new_label = {}
#     for k in labels.keys():
#         try:
#             new_label[k] = labels[k].detach().clone()
#         except Exception:
#             pass
#     new_output = {}
#     for k in output.keys():
#         new_output[k] = output[k].detach().clone()
#     output_final = process_output(new_output, new_label, smpl, process=True)
#     return output_final

def get_estimates(config, labels, output, smpl):

    with torch.enable_grad():
        init_depth = labels['joint_root'][:,2].clone().view(-1, config.sampling.multihypo_n+2).detach()
        n = init_depth.shape[0]
        depth_model = DeltaDepth(n).cuda(init_depth.device)
        depth_model.train()
        optim_list = [{"params":depth_model.delta_d,"lr":100}]
        optimizer = torch.optim.Adam(optim_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40, gamma=0.2)
        for i in range(200):
            optimizer.zero_grad()
            new_label = {}
            for k in labels.keys():
                try:
                    new_label[k] = labels[k].detach().clone()
                except Exception:
                    pass
            new_output = {}
            for k in output.keys():
                new_output[k] = output[k].detach().clone()
            new_label['joint_root'][:,2] = depth_model(init_depth).view(-1)
            output_final = process_output(new_output, new_label, smpl, process=True)
            output_final['loss'].backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
    return output_final


class ScorenetTrainer(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.num_joints = config.hyponet.num_joints
        self.num_twists = config.hyponet.num_twists
        self.num_item = self.num_joints*3+self.num_twists*2
        self.image_size = np.array(config.hrnet.image_size)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.multihypo_n = config.sampling.multihypo_n
        self.topk = config.training.scorenet.topk

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alpha = alphas_cumprod
        h36m_jregressor = np.load('data/smpl/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            'data/smpl/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        ).to(self.device)
        

    def train(self):
        args, config = self.args, self.config
        rank = torch.distributed.get_rank()
        
        num_models = len(config.training.scorenet.gen_path)
        model, model_cond = [], []
        for idx in range(num_models):
            model_i, model_cond_i, __, __, __, __, __, __, __, __, __ = get_model(config, is_train=False, resume = True, resume_path = config.training.scorenet.gen_path[idx])
            model.append(model_i)
            model_cond.append(model_cond_i)
        
        model_score, model_score_cond, ema_score, ema_score_cond, optimizer_score, optimizer_score_cond, start_epoch, step, loss_score = get_model_score(config, is_train=True, resume = config.training.resume_training, resume_path = self.config.training.resume_ckpt)
            
        model_score = nn.parallel.DistributedDataParallel(model_score, device_ids=[args.device], output_device=args.local_rank,find_unused_parameters=True)
        model_score_cond = nn.parallel.DistributedDataParallel(model_score_cond, device_ids=[args.device], output_device=args.local_rank,find_unused_parameters=True)
            
        for idx in range(num_models):
            for param in model_cond[idx].parameters():
                param.requires_grad = False
            for param in model[idx].parameters():
                param.requires_grad = False
        
    
        train_loaders, train_datasets, train_samplers = get_dataloader(config, is_train = True)
        train_loader = train_loaders['mix']
        train_dataset = train_datasets['mix']
        train_sampler = train_samplers['mix']

        scheduler_model = torch.optim.lr_scheduler.MultiStepLR(optimizer_score,config.optim.lr_step_model, config.optim.lr_factor_model,last_epoch= start_epoch-1)
        scheduler_model_cond = torch.optim.lr_scheduler.MultiStepLR(optimizer_score_cond, config.optim.lr_step_hrnet, config.optim.lr_factor_hrnet,last_epoch= start_epoch-1)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            print('lr lcn:{} lr hrnet:{}'.format(optimizer_score.state_dict()['param_groups'][0]['lr'],optimizer_score_cond.state_dict()['param_groups'][0]['lr']))
            loss_total= []
            loss_pve =[]
            loss_mcam = []
            loss_2d = []
            train_sampler.set_epoch(epoch)
            train_loader1 = tqdm(train_loader, dynamic_ncols=True)
            for i,  (inps, labels, img_ids, bboxes) in enumerate(train_loader1):
                optimizer_score.zero_grad()
                optimizer_score_cond.zero_grad()
                
                step += 1
                multi_n = np.array(config.training.scorenet.cases).sum()
                
                n = inps.size(0)
                
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                input = inps.float().to(self.device)
                joints = labels['joints_uvd_29'][:,:self.num_joints].float().to(self.device)
                twist =labels['target_twist'].float().to(self.device).view(n,-1)
                scale = torch.tensor([self.image_size[0],self.image_size[1],train_dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n,1).view(-1,3)
                
                ''' generation process '''
                output = {}
                score_input_list = {'joint':[], 'twist':[]}
                gen_output_list = {'pred_joints':[], 'pred_twist':[]}
                shuffle_idx = torch.randperm(multi_n)
                for k in range(num_models):
                    state = {
                        'model_cond': model_cond[k],
                        'model': model[k],
                        'input': input.clone(),
                        'scale': scale.clone(),
                    }

                    gen_output, score_input = self.gen_mesh(state, config.training.scorenet.cases[k])
                    for key in gen_output_list.keys():
                        gen_output_list[key].append(gen_output[key])
                    for key in score_input_list.keys():
                        score_input_list[key].append(score_input[key])
                for key in gen_output_list:
                    gen_output_list[key] = torch.cat(gen_output_list[key],dim=1)
                    gen_output_list[key] = gen_output_list[key][:, shuffle_idx]
                for key in score_input_list:
                    score_input_list[key] = torch.cat(score_input_list[key],dim=1)
                    score_input_list[key] = score_input_list[key][:, shuffle_idx]
                    score_input_list[key] = score_input_list[key].view(n*multi_n, -1)

                output['pred_twist'] = gen_output_list['pred_twist'].view(-1,self.num_twists,2)
                output['pred_joints'] = gen_output_list['pred_joints'].view(-1,self.num_joints,3)
                output['pred_shape'] = labels['target_beta'].unsqueeze(1).repeat(1,multi_n,1).view(-1,10)
                output_final = process_output(output.copy(), labels.copy(), self.smpl)
                
                cond_feature, output_final['pred_2d'] = model_score_cond(input.clone())
                joint_2d = labels['joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device)
                scale_2d = torch.tensor([self.image_size[0],self.image_size[1]]).float().to(self.device)
                joint_2d = normalize_pose_cuda(pose = joint_2d.clone(), which='scale_t',scale=scale_2d).to(self.device)
                
                # BUG remove twist
                score_input = torch.cat([score_input_list['joint'], score_input_list['twist']],dim=1).to(self.device)
                # score_input = torch.cat([score_input_list['joint']],dim=1).to(self.device)
                score = model_score(score_input, cond_feature)
                
                
                gt_betas = labels['target_beta']
                gt_thetas = labels['target_theta']
                if False:                # [BUG]
                    gt_thetas[:, :4] = torch.Tensor([1, 0, 0, 0])
                    output_final['theta'][:, 0] = torch.Tensor([1, 0, 0, 0])
                    pred_out = self.smpl(
                        pose_axis_angle=output_final['theta'],
                        betas=output_final['pred_shape'],
                        global_orient=None,
                        return_verts=True
                    )
                    output_final['pred_vertices_'] = pred_out.vertices
                    
                gt_output = self.smpl(
                    pose_axis_angle=gt_thetas,
                    betas=gt_betas,
                    global_orient=None,
                    return_verts=True
                )
                gt_mesh = gt_output.vertices.float()-gt_output.joints_from_verts.float()[:,0].unsqueeze(1)
                gt_mesh = gt_mesh.reshape(n, -1, 3)
                gt = {'mesh': gt_mesh,              # [bs, 6890, 3]
                      'joint_cam': labels['joints_xyz_17'].clone().float().to(self.device),
                      'mask_2d':labels['joints_vis_29'][:,:,:2].clone().float().to(self.device),
                      'pred_2d':joint_2d.clone()}
                
                mask = labels['prelate_weight'].view(-1).to(score.device)
                p_pve, p_mpjpe_cam, loss_2d_cur = loss_score(score, output_final, gt, mask)
                loss = p_pve + p_mpjpe_cam + loss_2d_cur

                loss.backward()
                
                try:
                    torch.nn.utils.clip_grad_norm_(model_score.parameters(), config.optim.grad_clip)
                    torch.nn.utils.clip_grad_norm_(model_score_cond.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                
                
                optimizer_score_cond.step()
                optimizer_score.step()

                loss_total.append(loss.item())
                loss_pve.append(p_pve.item())
                loss_mcam.append(p_mpjpe_cam.item())
                loss_2d.append(loss_2d_cur.item())

                
                if step % self.config.training.loss_freq == 0 or step == 1:
                    if rank ==0:
                        logging.info(f"epoch:{epoch},step: {step}, loss: {loss.item()}, batch size: {n}")
                        logging.info(f"epoch:{epoch},step: {step}, loss pve: {p_pve.item()},  loss mpjpe_cam: {p_mpjpe_cam.item()}, loss 2d:{loss_2d_cur.item()}, batch size: {n}")


                ema_score.update(model_score.parameters())
                ema_score_cond.update(model_score_cond.parameters())


            states = {
                'model_score': model_score.module.state_dict(),
                'optimizer_score': optimizer_score.state_dict(),
                'model_score_cond': model_score_cond.module.state_dict(),
                'optimizer_score_cond': optimizer_score_cond.state_dict(),
                'epoch': epoch,
                'step': step,
                'ema_score': ema_score.state_dict(),
                'ema_score_cond': ema_score_cond.state_dict()
            }
            if rank==0:
                torch.save(states, os.path.join(self.config.log_path, 'ckpt_epoch_{}.pth'.format(epoch)))
                torch.save(states, os.path.join(self.config.log_path, 'ckpt.pth'.format(epoch)))
                loss_total = np.array(loss_total).mean()
                loss_pve = np.array(loss_pve).mean()
                loss_mcam = np.array(loss_mcam).mean()
                loss_2d = np.array(loss_2d).mean()
                logger.info(f"epoch:{epoch},train_loss: {loss_total}, pve:{loss_pve} mcam:{loss_mcam} joint_2d:{loss_2d}")
            train_loader1.close()
            scheduler_model.step()
            scheduler_model_cond.step()
            
            if epoch % self.config.training.validation_freq == 0:
                torch.distributed.barrier()
                ema_score.store(model_score.parameters())
                ema_score.copy_to(model_score.parameters())
                ema_score_cond.store(model_score_cond.parameters())
                ema_score_cond.copy_to(model_score_cond.parameters())
                model_score.eval()
                model_score_cond.eval()
                states = {
                    'model_score': model_score,
                    'model_score_cond': model_score_cond,
                    'epoch': epoch
                }
                error_dict = self.validate(states)
                ema_score.restore(model_score.parameters())
                ema_score_cond.restore(model_score_cond.parameters())
                model_score.train()
                model_score_cond.train()
            torch.distributed.barrier()

    def validate(self ,state=None):
        args, config = self.args, self.config
        
        rank = torch.distributed.get_rank()
        model, model_cond, __, __, __, __, __, __, __, __, __ = get_model(config, is_train=False, resume = True, resume_path = config.training.scorenet.test_path)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)
        model_cond = nn.parallel.DistributedDataParallel(model_cond, device_ids=[args.device], output_device=args.local_rank)


        if state==None:
            if os.path.exists(getattr(self.config.sampling, "ckpt", None)):
                resume_path = self.config.sampling.ckpt
            elif getattr(self.config.sampling, "ckpt", None) is None:
                resume_path = os.path.join(self.config.log_path, "ckpt.pth")
            else:
                resume_path = os.path.join(self.config.log_path, self.config.sampling.ckpt)
            
            model_score, model_score_cond, __, __, __, __, epoch, step, __ = get_model_score(config, is_train=False, resume = True, resume_path = resume_path)
            model_score = nn.parallel.DistributedDataParallel(model_score, device_ids=[args.device], output_device=args.local_rank)
            model_score_cond = nn.parallel.DistributedDataParallel(model_score_cond, device_ids=[args.device], output_device=args.local_rank)
            state = dict(model_score=model_score,epoch=epoch,model_score_cond=model_score_cond)
        
        epoch = state['epoch']
        state['model'] = model
        state['model_cond'] = model_cond
        
        valid_loaders, valid_datasets, __ = get_dataloader(config, is_train = False)
        
        pred = {}
        for test_dataset in self.config.dataset.test_dataset:
            state['dataset'] = valid_datasets[test_dataset]
            state['dataloader'] =  valid_loaders[test_dataset]
            state['dataset_type'] = test_dataset
            pred[test_dataset] = self.sample(state, args.construct_dpo)
            torch.distributed.barrier()
        return pred
    

    def sample(self, state, construct_dpo=False):
        args, config = self.args, self.config
        
        model = state['model']
        model_cond = state['model_cond']
        model_score = state['model_score']
        model_score_cond = state['model_score_cond']
        dataset = state['dataset']
        dataloader = state['dataloader']
        epoch = state['epoch']
        dataset_type = state['dataset_type']
            
        rank = torch.distributed.get_rank()
        
        multi_n = self.multihypo_n
        with torch.no_grad():
            p = {}
            grpo_pair_list = []
            save_all_render_img_dir_list = []
            for i in range(multi_n+2):
                p[i] = {}
            for i, (inps, labels, img_ids, bboxes, img_path) in enumerate(tqdm(dataloader, ncols=100)):
                n = inps.size(0)
                output = {}
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                input = inps.float().to(self.device)
                save_trans_inv = labels['trans_inv'].cpu().numpy().astype(np.float32)
                save_trans = labels['trans'].cpu().numpy().astype(np.float32)
                scale = torch.tensor([self.image_size[0],self.image_size[1],dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n+2,1).view(-1,3)

                state['input'] = input.clone()
                state['scale'] = scale.clone()
                output, score_input = self.gen_mesh(state, multi_n)                     # get denoised twist and joints

                if construct_dpo:
                    # save_pred_joints = output['pred_joints'].cpu().numpy().astype(np.float32)
                    # save_pred_twist = output['pred_twist'].cpu().numpy().astype(np.float32)
                    # save_score_joint = score_input['joint'].cpu().numpy().astype(np.float32)
                    # save_score_twist = score_input['twist'].cpu().numpy().astype(np.float32)
                    # save_joints_uvd_29 = labels['joints_uvd_29'].cpu().numpy().astype(np.float32)
                    # save_w_twist = labels['target_twist'].cpu().numpy().astype(np.float32)
                    # save_trans = labels['trans'].cpu().numpy().astype(np.float32)
                    # ========== For Scorer ==========
                    save_score_joint = score_input['joint'].cpu().numpy().astype(np.float32).reshape(n, multi_n, self.num_joints, 3)
                    save_score_twist = score_input['twist'].cpu().numpy().astype(np.float32)
                    # ==============================

                    # for idx in range(n):
                    #     # BUG Save image
                    #     # img = inps[idx].clone()
                    #     # img[0].mul_(0.225).add_(0.406)  # 还原 R 通道
                    #     # img[1].mul_(0.224).add_(0.457)  # 还原 G 通道
                    #     # img[2].mul_(0.229).add_(0.480)  # 还原 B 通道
                    #     # img = cv2.cvtColor((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    #     # img_save_dir = f'data/dpo/images/{dataset.__class__.__name__}'
                    #     # os.makedirs(img_save_dir, exist_ok=True)
                    #     # cv2.imwrite(f'{img_save_dir}/{img_ids[idx].item():08d}.png', img)

                    #     dpo_pair = {
                    #         'img_path': img_path[idx],
                    #         # 'db_idx': labels['db_idx'][idx].item(),
                    #         'img_idx': img_ids[idx].item(),                     # absolute img_ids 
                    #         # 'img_256': inps[idx].clone().float().detach().tolist(),
                    #         'output_joints': save_pred_joints[idx],
                    #         'output_twists': save_pred_twist[idx],
                    #         'score_joints': save_score_joint[idx],
                    #         'score_twists': save_score_twist[idx],
                    #         'gt_joints': save_joints_uvd_29[idx],
                    #         'gt_twists': save_w_twist[idx],
                    #         'trans': save_trans[idx],
                    #         'trans_inv': save_trans_inv[idx],
                    #     }
                    #     dpo_pair_list.append(dpo_pair)
                    # continue


                output['pred_shape'] = output['pred_shape'].unsqueeze(1).repeat(1, multi_n+2, 1).view(-1,10)
                output['pred_joints'] = output['pred_joints'].view(-1,self.num_joints,3)        # 1600, 29, 3
                output['pred_twist'] = output['pred_twist'].view(-1,self.num_twists,2)          # 1600, 23, 2

                
                if False:           # vis joints
                    image = (input[0].permute(1, 2, 0).detach().cpu().numpy())  # 转换为 (256, 256, 3)
                    image_ = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)

                    print(image_.shape)
                    keypoints = score_input['joint'].view(n, multi_n, 29, 3)

                    # 获取关键点坐标并将其从 [-1, 1] 映射到 [0, 255]
                    keypoints = keypoints[0, 0, :, :2].cpu().numpy()  # 取前两维 (29, 2)
                    keypoints = ((keypoints + 1) / 2 * 255).astype(int)  # 转换为像素坐标

                    for (x, y) in keypoints:
                        print(x,y)
                        cv2.circle(image_, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # 红色圆点
                    cv2.imwrite('debug.png', image_)


                # Scorer selection
                cond_feature,__ = model_score_cond(input.clone())

                score_input = torch.cat([score_input['joint'].view(n*multi_n,-1),score_input['twist'].view(n*multi_n,-1)],dim=1).to(self.device)
                score = model_score(score_input, cond_feature)

                
                score = score.view(n, multi_n)
                joints = output['pred_joints'].view(n,multi_n,self.num_joints,3)
                twist = output['pred_twist'].view(n,multi_n,self.num_twists,2)
                idx_score = score.contiguous().argsort(dim=1)[:, -self.topk:].contiguous().view(-1)
                idx_bs = torch.arange(n).repeat(self.topk).sort()[0]
                
                select_score = score[idx_bs, idx_score].view(n, self.topk)
                select_score = torch.nn.functional.softmax(select_score,dim=-1).unsqueeze(-1)

                select_joints = joints[idx_bs, idx_score].view(n,self.topk,self.num_joints,3).mean(dim=1)           # take the average of topk predictions 160, 29, 3
                #select_joints = (select_joints*select_score.unsqueeze(-1)).sum(dim=1)

                select_twist = twist[idx_bs,idx_score].view(n,self.topk,self.num_twists,2)
                select_twist = select_twist/(torch.norm(select_twist, dim=-1, keepdim=True) + 1e-8)
                select_angle = torch.arctan(select_twist[:,:,:,1]/select_twist[:,:,:,0]) 
                flag = (torch.cos(select_angle)*select_twist[:,:,:,0])<0
                select_angle = select_angle + flag*np.pi
                assert ((torch.cos(select_angle)-select_twist[:,:,:,0]).abs()>1e-6).sum() ==0
                assert ((torch.sin(select_angle)-select_twist[:,:,:,1]).abs()>1e-6).sum() ==0
                select_angle = select_angle.mean(dim=1)                                                             # take the average of topk predictions
                #select_angle = (select_angle*select_score).sum(dim=1)
                select_twist = select_twist.mean(dim=1)
                select_twist = torch.cat([torch.cos(select_angle).unsqueeze(-1), torch.sin(select_angle).unsqueeze(-1)], dim=-1)

                output['pred_joints'] = output['pred_joints'].view(n, multi_n, self.num_joints, 3)
                mean_joint = output['pred_joints'].mean(dim=1).unsqueeze(1)
                output['pred_joints'] = torch.cat([select_joints.unsqueeze(1), output['pred_joints'], mean_joint], dim=1).view(-1, self.num_joints, 3)      # [bs x (topk_mean + multi_n + all hypos' mean), 29, 3]
                output['pred_twist'] = output['pred_twist'].view(n, multi_n, self.num_twists, 2)
                mean_twist = output['pred_twist'].mean(dim=1).unsqueeze(1)
                output['pred_twist'] = torch.cat([select_twist.unsqueeze(1),output['pred_twist'], mean_twist],dim=1).view(-1,self.num_twists,2)      # [bs x (topk_mean + multi_n + all hypos' mean), 23, 2]

                score = torch.cat([torch.zeros(n,1).to(score.device) + score.max()+50, score.view(n, multi_n), torch.zeros(n,1).to(score.device)],dim=-1)
                score = score.view(n, multi_n+2).cpu().data.numpy()
            
                pred_shape = output['pred_shape'].clone().cpu().data.numpy()
                pred_uvd_jts_29 = trans_back(output['pred_joints'].clone(), labels['trans_inv'].clone()).cpu().data.numpy()
                pred_twist = output['pred_twist'].clone().cpu().data.numpy()
                
                pred_shape = pred_shape.reshape(n, multi_n+2, 10)
                pred_twist = pred_twist.reshape(n, multi_n+2, 23, 2)
                pred_uvd_jts_29 = pred_uvd_jts_29.reshape(n, multi_n+2, 29, 3)

                if construct_dpo:           # NOTE beacuse after process_output, the "output" will change
                    save_pred_joints = output['pred_joints'].reshape(n, multi_n+2, self.num_joints, 3).cpu().numpy().astype(np.float32)
                    
                if construct_dpo and True:             # Render SMPL images for VLM scoring
                    draw_num = multi_n + 2

                    output_final = get_estimates(self.config, labels, output, self.smpl)

                    pred_joints = output_final['pred_joints'].clone().cpu().data.numpy() * 2
                    pred_joints = pred_joints.reshape(n,multi_n+2,29,3)
                    pred_uvd_jts_24 = output_final['uvd_24'].cpu().data.numpy().reshape(n,multi_n+2,-1,2)

                    # vertice
                    pred_mesh = output_final['pred_vertices'].reshape(n,multi_n+2,-1,3)
                    # draw_mesh = pred_mesh.view(n,multi_n+2,-1,3).cpu().data.numpy()
                    pred_mesh = pred_mesh.cpu().data.numpy()
                    pred_xyz_jts_17 = output_final['pred_xyz_jts_17'].reshape(n,multi_n+2, 17, 3).cpu().data.numpy()
                    
                    f = labels['f'].cpu().data.numpy()
                    c = labels['c'].cpu().data.numpy()
                    root_img = pred_uvd_jts_29[:,0,0,:2]
                    results = {'pred_mesh': pred_mesh[:, :draw_num],
                                'focal_l':f,
                                'center_pt':c,
                                'pred_root_xy_img': root_img,
                                'img_name': list((ip.split('/')[-2] + '_' + ip.split('/')[-1]) for ip in img_path),             # BUG ensure uniqueness
                                'img_path': img_path,}

                    pred_mesh = results['pred_mesh'] # (N*T, V, 3)
                    focal_l = results['focal_l']
                    center_pt = results['center_pt']
                    img_path_list = results['img_path']
                    img_name = results['img_name']
                    for mesh_idx in tqdm(range(pred_mesh.shape[0])):
                        img = cv2.imread(img_path_list[mesh_idx])
                        ori_img_height, ori_img_width = img.shape[:2]
                        _bbox = (labels['bbox_origin'].cpu().numpy().astype(np.int32)[mesh_idx])
                        dirpath = osp.join(self.config.grpo_render_img_save_path, dataset.__class__.__name__, img_name[mesh_idx])
                        if os.path.exists(dirpath) and os.path.isdir(dirpath):
                            save_all_render_img_dir_list.append(str(dirpath))
                            continue
                            # shutil.rmtree(dirpath)
                        else:
                            os.makedirs(dirpath, exist_ok=True)

                        # # ==== 放大 bbox 并裁剪 ====
                        # bx1, by1, bx2, by2 = _bbox  # 原始 bbox

                        # # bbox 中心点与宽高
                        # bw = bx2 - bx1
                        # bh = by2 - by1
                        # cx = (bx1 + bx2) / 2
                        # cy = (by1 + by2) / 2

                        # scale = 1.2  # 放大比例

                        # # 放大 bbox
                        # new_bw = bw * scale
                        # new_bh = bh * scale

                        # new_bx1 = int(max(0, min(ori_img_width - 1, round(cx - new_bw / 2))))
                        # new_by1 = int(max(0, min(ori_img_height - 1, round(cy - new_bh / 2))))
                        # new_bx2 = int(max(0, min(ori_img_width - 1, round(cx + new_bw / 2))))
                        # new_by2 = int(max(0, min(ori_img_height - 1, round(cy + new_bh / 2))))

                        # thickness = 4
                        # cropped_origin_img = img[new_by1:new_by2, new_bx1:new_bx2]
                        
                        # if cropped_origin_img.size == 0:
                        #     print(f"[WARN] Empty crop for bbox {dirpath}, skipping.")
                        #     cropped_origin_img = img  # fallback

                        # # ==== Resize：长边 = 512 ====
                        # _h, _w = cropped_origin_img.shape[:2]
                        # long_side = max(_h, _w)
                        # _scale_factor = 512 / long_side
                        # new_w = int(round(_w * _scale_factor))
                        # new_h = int(round(_h * _scale_factor))
                        # cropped_origin_img = cv2.resize(cropped_origin_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        # fast_save(osp.join(dirpath,  f"cropped_origin.jpg"), cropped_origin_img)
                        fast_save(osp.join(dirpath,  f"origin.jpg"), img)

                        for i in range(draw_num):
                            pred_mesh_T = pred_mesh[mesh_idx, i]
                            rgb, depth = render_mesh_fast(ori_img_height, ori_img_width, [pred_mesh_T/1000], self.smpl.faces, {'focal': focal_l[mesh_idx], 'princpt': center_pt[mesh_idx]})
                            # side_rgb, side_depth = render_side_mesh_(ori_img_height, ori_img_width, [pred_mesh_T/1000], self.smpl.faces, {'focal': focal_l[mesh_idx], 'princpt': center_pt[mesh_idx]})
                            
                            valid_mask = (depth > 0)[:,:,None] 
                            rendered_img = rgb * valid_mask + img[:,:,::-1] * (1-valid_mask)
                            rendered_img = np.ascontiguousarray(rendered_img.astype(np.uint8)[...,::-1])
                            # rendered_img = cv2.resize(rendered_img, (int(ori_img_width * _s), int(ori_img_height * _s)), interpolation=cv2.INTER_AREA)


                            # Draw bbox
                            # bx1, by1, bx2, by2 = _bbox
                            # # 裁剪并取整
                            # bx1 = int(max(0, min(ori_img_width-1, round(bx1))))
                            # by1 = int(max(0, min(ori_img_height-1, round(by1))))
                            # bx2 = int(max(0, min(ori_img_width-1, round(bx2))))
                            # by2 = int(max(0, min(ori_img_height-1, round(by2))))
                            # cv2.rectangle(rendered_img, (bx1, by1), (bx2, by2), (0, 0, 255), thickness)
                            # cv2.rectangle(rendered_img, (new_bx1, new_by1), (new_bx2, new_by2), (0, 0, 255), thickness)

                            # 裁剪区域
                            # cropped_img = rendered_img[new_by1:new_by2, new_bx1:new_bx2]
                            # if cropped_img.size == 0:
                            #     # print(f"[WARN] Empty crop for bbox {i}, skipping.")
                            #     cropped_img = rendered_img  # fallback
                            # rendered_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


                            save_path_img = osp.join(dirpath,  f"hypo{i:02d}.jpg")
                            os.makedirs(dirpath, exist_ok=True)
                            # cv2.imwrite(save_path_img, rendered_img)
                            fast_save(save_path_img, rendered_img)

                        save_all_render_img_dir_list.append(str(dirpath))
                        print(save_all_render_img_dir_list[-1])

                output_final = process_output(output, labels, self.smpl)
                pred_joints = output_final['pred_joints'].clone().cpu().data.numpy() * 2        # root-relative 3D joints
                pred_joints = pred_joints.reshape(n, multi_n+2, 29, 3)


                gt_betas = labels['target_beta'].view(-1, 10)
                gt_thetas = labels['target_theta'].view(-1, 96)
                gt_output = self.smpl(
                        pose_axis_angle=gt_thetas,
                        betas=gt_betas,
                        global_orient=None,
                        return_verts=True
                    )
                # vertice
                pred_mesh = output_final['pred_vertices'].reshape(n, multi_n+2, -1, 3)
                draw_mesh = pred_mesh.view(n,multi_n+2,-1,3).cpu().data.numpy()
                gt_mesh = gt_output.vertices.float()-gt_output.joints_from_verts.float()[:,0].unsqueeze(1)
                gt_mesh = gt_mesh.reshape(n,1,-1,3)
                pred_mesh = pred_mesh.cpu().data.numpy()
                gt_mesh = gt_mesh.cpu().data.numpy()
                pred_xyz_jts_17 = output_final['pred_xyz_jts_17'].reshape(n, multi_n+2, 17, 3).cpu().data.numpy()   # joints cam [n, multin, 17, 3]
                pred_theta = output_final['theta'].reshape(n, multi_n+2 ,24, 4).cpu().data.numpy()
                
                pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, axis=-1))
                pve = np.mean(pve,axis=-1) * 1000
                pve = pve.reshape(n, multi_n+2)

                if False:           # vis two point clouds
                    # path_s = os.path.join(config.save_path,str(self.multihypo_n),'mesh',state['dataset_type'])
                    # for k in range(multi_n//2+1):
                    #     for j in range(pred_xyz_jts_17.shape[0]):
                    #         path_i = os.path.join(path_s,str(int(img_ids[j])))
                    #         os.makedirs(path_i,exist_ok=True)
                    #         path_i = os.path.join(path_i,str(k)+'.obj')
                    #         write_obj(path_i,draw_mesh[j][k],self.smpl.faces)
                    def save_two_pointclouds_to_one_ply(v1: np.ndarray, v2: np.ndarray, out_path="combined.ply"):
                        pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v1))
                        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v2))
                        pcd1.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
                        pcd2.paint_uniform_color([0.2, 0.5, 1.0])  # 蓝色
                        combined = pcd1 + pcd2
                        o3d.io.write_point_cloud(out_path, combined)

                    save_two_pointclouds_to_one_ply(pred_mesh[0,0], gt_mesh[0,0], "combined_mesh.ply")
                    gt_jts_17 = gt_output.joints_from_verts.float() / 2
                    gt_jts_17 = (gt_jts_17 - gt_jts_17[:, 0:1]).cpu().numpy()
                    pred_xyz_jts_17_vis = pred_xyz_jts_17 - pred_xyz_jts_17[:, :, 0:1, :]
                    save_two_pointclouds_to_one_ply(pred_xyz_jts_17_vis[0,0], gt_jts_17[0], "combined_jts.ply")
                

                if construct_dpo:
                    gt_jts_17 = gt_output.joints_from_verts.float() / 2
                    gt_jts_17 = (gt_jts_17 - gt_jts_17[:, 0:1]).cpu().numpy()
                    pred_xyz_jts_17_vis = pred_xyz_jts_17 - pred_xyz_jts_17[:, :, 0:1, :]
                    mpjpe = np.linalg.norm(pred_xyz_jts_17_vis - gt_jts_17[:, None], axis=-1).mean(-1) * 1000

                    pampjpe = np.empty((n, multi_n+2), dtype=np.float32)

                    for nn in range(n):
                        gt = gt_jts_17[nn]  # (17,3)
                        for mm in range(multi_n+2):
                            p = pred_xyz_jts_17_vis[nn, mm]  # (17,3)
                            pred_aligned = rigid_align(p, gt)  # 你的对齐函数
                            err = np.linalg.norm(pred_aligned - gt, axis=-1).mean() * 1000.0
                            pampjpe[nn, mm] = err
                    
                    # construct with GT as winner
                    save_pred_twist = pred_twist.astype(np.float32)
                    save_pred_joints_uvd_29 = pred_uvd_jts_29.astype(np.float32)
                    

                    save_joints_uvd_29 = labels['joints_uvd_29'].cpu().numpy().astype(np.float32)
                    save_w_twist = labels['target_twist'].cpu().numpy().astype(np.float32)
                    save_trans = labels['trans'].cpu().numpy().astype(np.float32)
                    save_bbox = labels['bbox_origin'].cpu().numpy().astype(np.float32)

                    for idx in range(n):
                        # for idx_hypo in range(multi_n):
                        grpo_pair = {
                            'img_path': img_path[idx],
                            # 'db_idx': labels['db_idx'][idx].item(),
                            'img_idx': img_ids[idx].item(),                     # absolute img_ids
                            'pred_joints': save_pred_joints[idx],
                            'pred_twist': save_pred_twist[idx],
                            'pred_joints_uvd_29': save_pred_joints_uvd_29[idx],
                            'gt_joints': save_joints_uvd_29[idx],
                            'gt_twist': save_w_twist[idx],
                            'score_joints': save_score_joint[idx],
                            'score_twists': save_score_twist[idx],              # multin, 29, 3
                            'trans': save_trans[idx],
                            'trans_inv': save_trans_inv[idx],
                            'pve': pve[idx],
                            'mpjpe': mpjpe[idx],
                            'pampjpe': pampjpe[idx],
                            'theta': pred_theta[idx],
                            'multi_n': multi_n+2,
                        }
                        # copy_with_structure(Path(img_path[idx]), Path('data/pw3d/imageFiles'), Path('data/backup/pw3d'))          # BUG manual change backup path

                        if len(save_all_render_img_dir_list) != 0:
                            grpo_pair['render_smpl_dir_list'] = save_all_render_img_dir_list[idx]
                            grpo_pair['bbox_origin'] = save_bbox[idx]           # bx1, by1, bx2, by2

                        grpo_pair_list.append(grpo_pair)
                    if len(save_all_render_img_dir_list) != 0:
                        assert len(save_all_render_img_dir_list) == len(set(save_all_render_img_dir_list)), "Save List contains duplicate elements!"
                    continue


                for k in range(multi_n+2):
                    for j in range(pred_xyz_jts_17.shape[0]):
                        item_i = {'xyz_17': pred_xyz_jts_17[j][k],
                                'score': score[j][k],
                                'uvd_29': pred_uvd_jts_29[j][k],
                                'shape': pred_shape[j][k],
                                'twist': pred_twist[j][k],
                                'pred': pred_joints[j][k],
                                'pve': pve[j][k],
                                'theta': pred_theta[j][k]
                                }
                        p[k][int(img_ids[j])] = item_i
        
        if construct_dpo:
            # save_path = f'data/dpo/annotations/new_{dataset.__class__.__name__}_train_interval_{config.sampling.interval}_multihypo_{config.sampling.multihypo_n}.npz'
            # if os.path.exists(save_path):
                # print(f"Overwrite {save_path} !!!")
            # Cache.save(save_path, dpo_pair_list)
            # save_path = f'data/grpo/annotations/gt_sorted_{dataset.__class__.__name__}_train_interval_{config.sampling.interval}_multihypo_{config.sampling.multihypo_n}.npz'
            save_path = f'data/grpo/annotations/rendersmpl_sorted_{dataset.__class__.__name__}_train_interval_{config.sampling.interval}_multihypo_{config.sampling.multihypo_n}.npz'
            if os.path.exists(save_path):
                print(f"Overwrite {save_path} !!!")
            np.savez(save_path, grpo_pair_list=grpo_pair_list)
            print(f'[{dataset.__class__.__name__}] Caching datalist to {save_path}...')
            print(f"Saving GRPO data pairs to {save_path}")
            sys.exit(0)

        name = 'validate' + str(epoch)
        save_path=os.path.join(self.config.save_path, name)
        path = save_path+'_'+str(rank)+'.pkl'
        with open(path, 'wb') as fid:
            pickle.dump(p, fid, pickle.HIGHEST_PROTOCOL)
        print('dump the file', path)
        torch.distributed.barrier()
        name = 'validate' + str(epoch)
        save_path = os.path.join(self.config.save_path, name)
        if rank==0:
            print('gpu num: {}'.format(args.world_size))
            p = {}
            for i in range(multi_n+2):
                p[i] = {}
            for r in range(self.args.world_size):
                path = save_path+'_'+str(r)+'.pkl'
                with open(path, 'rb') as fid:
                    pred_i = pickle.load(fid)
                    print('load the file',path)
                for midx in range(multi_n+2):
                    p[midx].update(pred_i[midx])
                os.remove(path)
            save_path= self.config.save_path
            p_relate = process_pred(p, dataset, multi_n+2, type=dataset_type, save_path=save_path, use_score=True)
        else:
            p_relate = {}
        torch.distributed.barrier()
        return p_relate
        

    def sample_pose(self, xj, xt, model,content=None, last=True,gen_multi=True):
        
        args, config = self.args, self.config
        if self.config.diffusion.skip_type == "uniform":
            skip = self.num_timesteps // self.config.diffusion.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.config.diffusion.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.config.diffusion.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        if self.num_timesteps-1 not in list(seq):
            seq = list(seq)+[self.num_timesteps-1]

        xs = generalized_steps(xj, xt, seq, model, self.betas.to(self.device), eta=self.config.diffusion.eta,ctx =content, gen_multi=gen_multi)
        x = xs
        if last:                                    # get the predicted x_0, abandon intermediate results
            x = [x[0][0][-1],x[0][1][-1]]
        return x

    def gen_mesh(self, state, multi_n):
        model = state['model']
        model_cond = state['model_cond']
        scale = state['scale']
        input = state['input']

        gen_output = {}
        score_input = {}

        ctx , pred_shape, __ = model_cond(input)
        gen_output['pred_shape'] = pred_shape                                   # bs, 10

        n = input.shape[0]
        if self.config.sampling.zero and self.config.sampling.multihypo_n==1:
            xj = torch.zeros(n*multi_n, self.num_joints*3).to(self.device)
            xt = torch.zeros(n*multi_n, self.num_twists*2).to(self.device)
        else:
            xj = torch.randn(n,multi_n, self.num_joints*3).to(self.device)      # swing represenataion in joint positions [bs, multi_n. n_joints x 3]
            xt = torch.randn(n,multi_n, self.num_twists*2).to(self.device)      # twist representation
            xj[:,0] = 0
            xj = xj.view(n*multi_n, self.num_joints*3).to(self.device)
            xt[:,0] = 0
            xt = xt.view(n*multi_n, self.num_twists*2).to(self.device)
        
        pred_j, pred_t= self.sample_pose(xj=xj, xt=xt, model=model, content=ctx, gen_multi=True)
        gen_output['pred_joints'] = denormalize_pose_cuda(pose = pred_j, which=self.config.hyponet.norm, scale=scale, \
                                                          mean_and_std=self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints,3)       # bs, multi_n, 29, 3 img space
        gen_output['pred_twist'] = pred_t.view(n, multi_n,self.num_twists,2) / self.config.diffusion.scale                                                  # bs, multi_n, 23, 2

        score_input['joint'] = normalize_pose_cuda(pose = gen_output['pred_joints'].clone().view(-1, self.num_joints, 3), \
                                                   which='scale_t', scale=scale*self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints*3)
        score_input['twist'] = gen_output['pred_twist'].clone()

        return gen_output, score_input