import os
import os.path as osp
import numpy as np
import torch
import cv2
from .humandata import HumanDataset
import torchvision.transforms as transforms
from tqdm import tqdm

from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset
from utils.inference import output2original_scale

class InstaVariety(HumanDataset):
    def __init__(self, cfg, transform, data_split):
        super(InstaVariety, self).__init__(cfg, transform, data_split)

        self._cfg = cfg

        self.datalist = []

        pre_prc_file = 'insta_variety_neural_annot_train.npz'
        cache_filename = 'insta_variety_neural_annot_train_interval_100.npz'
        if self.data_split == 'train':
            filename = getattr(self._cfg, 'filename', pre_prc_file)
        else:
            # raise ValueError('InstaVariety test set is not support')
            filename = getattr(self._cfg, 'filename', pre_prc_file)

        self.img_dir = osp.join(self._cfg.data_dir, 'InstaVariety')
        self.annot_path = osp.join(self._cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(self._cfg.data_dir, 'cache', cache_filename)
        self.use_cache = getattr(self._cfg, 'use_cache', False)
        print(self.annot_path_cache, self.use_cache)
        self.img_shape = (224, 224)  # (h, w)
        self.cam_param = {}
        
        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_train_sample_interval', 1),
                test_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_test_sample_interval', 10))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)

        # # ========== prepare detection results ==========
        # rank = torch.distributed.get_rank()
        # det_device =  torch.device("cuda")
        
        # virtualpose_name = 'VirtualPose' 
        # det_update_config(f'{virtualpose_name}/configs/images/images_inference.yaml')
        
        # cur_path = ""
        # img_paths_list = [self.datalist[idx]['img_path'] for idx in range(len(self.datalist))]

        # det_model = eval('det_models.multi_person_posenet.get_multi_person_pose_net')(det_cfg, is_train=False)
        # with torch.no_grad():
        #     det_model = torch.nn.DataParallel(det_model,device_ids=[rank])

        # pretrained_file = osp.join(cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED)
        # state_dict = torch.load(pretrained_file)
        # new_state_dict = {k:v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
        # det_model.module.load_state_dict(new_state_dict, strict = False)
        # pretrained_file = osp.join(cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED_BACKBONE)
        # det_model = load_backbone_validate(det_model, pretrained_file)

        # # prepare detection dataset
        # infer_dataset = det_dataset.images_custom(
        #     det_cfg, img_paths_list, focal_length=1700, 
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]),
        #     ]))
        # infer_loader = torch.utils.data.DataLoader(
        #     infer_dataset,
        #     batch_size = 160,
        #     shuffle=False,
        #     num_workers = 12,
        #     pin_memory=True,
        #     drop_last=False,)
        
        # det_model.eval()

        # max_person = 0
        # detection_all = []
        # valid_frame_idx_all = []
        # img_list_all = []
        # with torch.no_grad():
        #     f_start = -1
        #     for _, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(infer_loader, dynamic_ncols=True)):
        #         for k in meta.keys():
        #             try:
        #                 meta[k] = meta[k].to(det_device)
        #             except Exception:
        #                 pass
        #             inputs = inputs.to(det_device)
        #             targets_2d =targets_2d.to(det_device)
        #             targets_3d =targets_3d.to(det_device)
        #             weights_2d = weights_2d.to(det_device)
        #             input_AGR = input_AGR.to(det_device)
        #         _, _, output, _, _ = det_model(views=inputs, meta=meta, targets_2d=targets_2d,
        #                                                         weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
        #         det_results, n_person, valid_frame_idx,img_list,f_start = output2original_scale(meta, output, start=f_start+1)
        #         detection_all += det_results
        #         valid_frame_idx_all += valid_frame_idx
        #         img_list_all += img_list
        #         max_person = max(n_person, max_person)
        
        # # list to array
        # detection_all = np.array(detection_all)
        # img_list_all = img_list_all


    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        # get image id
        img_path = self.datalist[idx]['img_path']
        img_id = self.datalist[idx]['img_id']

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.datalist[idx].keys():
            label[k] = self.datalist[idx][k].copy()
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox, img_path
