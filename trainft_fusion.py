import time
from options.train_options import TrainOptions
from models import create_model
from models import find_model_using_name
from models.networks import get_scheduler
#from models.base_model import load_networks
from models.train_model import TrainModel

from models.networks import SIGGRAPHGenerator
from util.visualizer import Visualizer
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

from fusion_dataset import *
from util import util
import os

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def display_results(visuals, epoch,opt):
    web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
    img_dir = os.path.join(web_dir, 'images')
    for label, image in visuals.items():
        image_numpy = util.tensor2im(image)
        img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
        torchvision.utils.save_image(image, img_path)
    print('saved')

if __name__ == '__main__':
    opt = TrainOptions().parse()

    if opt.stage == 'full':
        dataset = Training_Full_Dataset(opt)
    elif opt.stage == 'instance':
        dataset = Training_Instance_Dataset(opt)
    elif opt.stage == 'fusion':
        dataset = Training_Fusion_Dataset(opt)
    else:
        print('Error! Wrong stage selection!')
        exit()
    #Change numbers based on size of dataset split into training and validation
    train_set, val_set = torch.utils.data.random_split(dataset,[27000,3284])
    dataset_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

    dataset_size = len(train_set)
    feature_extracting = False
    print('#training images = %d' % dataset_size)
    path = 'checkpoints/coco_mask'
    path_fusion = path + '/24_net_GF.pth'
    model_fusion = create_model(opt) #init
    model_fusion.setup(opt)
    state_dict = torch.load(path_fusion)
    model_fusion_1 = model_fusion.netGF

    path_full = path + '/latest_net_GComp.pth'
    #model_full = create_model(opt) #init
    #model_full.setup(opt)
    state_dict_full = torch.load(path_full)

    path_instance = path + '/latest_net_G.pth'
    #model_instance = create_model(opt) #init
    #model_instance.setup(opt)
    state_dict_instance = torch.load(path_instance)

    # Loading old pretrained models
    model_fusion_1.load_state_dict(state_dict,strict=False)
    model_fusion.netGComp.load_state_dict(state_dict_full,strict=False) 
    model_fusion.netG.load_state_dict(state_dict_instance,strict=False) 
    
    #set_parameter_requires_grad(model_fusion_1,feature_extracting)
    #model_fusion_1.module.model_class[0] = nn.Conv2d(256, 365, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_fusion_1 = model_fusion_1.to(device)
    model_fusion.netGComp = model_fusion.netGComp.to(device)
    model_fusion.netG = model_fusion.netG.to(device)
    params_to_update = model_fusion_1.parameters()

    """Use this if only finetuning certain layers
    if feature_extracting:
        params_to_update = []
        for name,param in model_fusion_1.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_fusion_1.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    """
    #Optimisation function
    optim_ft = torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))

    opt.display_port = 8098
    visualizer = Visualizer(opt)
    total_steps = 0

    if opt.stage == 'full' or opt.stage == 'instance':
        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0
            t0 = time.time()
            count = 0
            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                data_raw['rgb_img'] = [data_raw['rgb_img']]
                data_raw['gray_img'] = [data_raw['gray_img']]

                input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
                gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=10.0)
                if gt_data is None:
                    continue
                if(gt_data['B'].shape[0] < opt.batch_size):
                    continue
                input_data['B'] = gt_data['B']
                input_data['hint_B'] = gt_data['hint_B']
                input_data['mask_B'] = gt_data['mask_B']

                visualizer.reset()
                model_full.set_input(input_data)
                model_full.optimize_parameters_fus()
                """
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    display_results(model_fusion.get_current_visuals(), epoch, save_result,opt)
                """
                if total_steps % opt.print_freq == 0:
                    losses = model_fusion.get_current_losses()
                    if opt.display_id > 0:
                        #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
                        t1 = time.time()
                        elasped = t1 - t0
                        t0 = t1
                        count += elasped 
                        visualizer.print_current_losses(epoch,epoch_iter,losses,elasped,count)


            if epoch % opt.save_epoch_freq == 0:
                model_full.save_networks('latest')
                model_full.save_networks(epoch)
            model_full.update_learning_rate()
            
    elif opt.stage == 'fusion':
        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            epoch_iter = 0
            t0 = time.time()
            count = 0
            model_fusion.netGF.train()
            model_fusion.netG.train()
            model_fusion.netGComp.train()
            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                box_info = data_raw['box_info'][0]
                box_info_2x = data_raw['box_info_2x'][0]
                box_info_4x = data_raw['box_info_4x'][0]
                box_info_8x = data_raw['box_info_8x'][0]
                cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
                cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
                full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
                full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
                if cropped_gt_data is None or full_gt_data is None:
                    continue
                cropped_input_data['B'] = cropped_gt_data['B']
                full_input_data['B'] = full_gt_data['B']
                visualizer.reset()
                model_fusion.set_input(cropped_input_data)
                model_fusion.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])

                model_fusion.optimize_parameters(optim_ft)
                """
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    display_results(model_fusion.get_current_visuals(), epoch, save_result,opt)
                """
                if total_steps % opt.print_freq == 0:
                    losses = model_fusion.get_current_losses()
                    if opt.display_id > 0:
                        t1 = time.time()
                        elasped = t1 - t0
                        t0 = t1
                        count += elasped 
                        visualizer.print_current_losses(epoch+25,epoch_iter,losses,elasped,count)

            #with torch.no_grad():
            #    val_loss = model_fusion.validatefus(val_loader, optim_ft, epoch, opt)

            if epoch % opt.save_epoch_freq == 0:
                model_fusion.save_fusion_epoch(epoch+25)
                display_results(model_fusion.get_current_visuals(), epoch+25,opt)
            model_fusion.update_learning_rate()
    else:
        print('Error! Wrong stage selection!')
        exit()
