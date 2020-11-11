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
import matplotlib.pyplot as plt

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            print(param.data)

def display_results(visuals, epoch,opt):
    web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
    img_dir = os.path.join(web_dir, 'images')
    for label, image in visuals.items():
        image_numpy = util.tensor2im(image)
        img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
        torchvision.utils.save_image(image, img_path)
    print('saved')

def my_collate(batch):
    rgb = [item['rgb_img'] for item in batch]
    gray = [item['gray_img'] for item in batch]
    rgb = torch.LongTensor(rgb)
    gray = torch.LongTensor(gray)
    return [rgb, gray]

if __name__ == '__main__':
    opt = TrainOptions().parse()

    if opt.stage == 'full':
        dataset = Training_Full_Dataset(opt)
        dataset.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),transforms.ToTensor()])
    elif opt.stage == 'instance':
        dataset = Training_Instance_Dataset(opt)
    elif opt.stage == 'fusion':
        dataset = Training_Fusion_Dataset(opt)
    else:
        print('Error! Wrong stage selection!')
        exit()
    
    #Change numbers based on size of dataset split into training and validation
    train_set, val_set = torch.utils.data.random_split(dataset,[27000,3284])
    #train_set, val_set = torch.utils.data.random_split(dataset,[38000,4023])
    dataset_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataset_size = len(train_set)
    feature_extracting = False
    print('#training images = %d' % dataset_size)
    path = 'checkpoints/coco_full'
    path_full = path + '/latest_net_G.pth'
    model_full = create_model(opt) #init
    model_full.setup(opt)
    model_full_1 = model_full.netG
    state_dict = torch.load(path_full)
    

    # Loading old pretrained model 
    model_full.netG.load_state_dict(state_dict, strict=False) 
    #Use these lines if only finetuning certain layers
    #set_parameter_requires_grad(model_full.netG,feature_extracting)
    #model_full.netG.module.model_class[0] = nn.Conv2d(256, 313, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_full.netG = model_full.netG.to(device)
    params_to_update = model_full.netG.parameters()
    """ Use this if only finetuning certain layers
    if feature_extracting:
        params_to_update = []
        for name,param in model_full.netG.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_full.netG.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    """
    # Optimistaion function
    optim_ft = torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))

    opt.display_port = 8098
    visualizer = Visualizer(opt)
    total_steps = 0

    if opt.stage == 'full' or opt.stage == 'instance':
        for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
            display_count = 0
            epoch_iter = 0
            t0 = time.time()
            count = 0
            model_full.netG.train()
            
            for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                display_count += 1
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
                B = input_data['B']
                input_data['B'] = gt_data['B']
                input_data['hint_B'] = gt_data['hint_B']
                input_data['mask_B'] = gt_data['mask_B']
        
                visualizer.reset()
                model_full.set_input(input_data)
                model_full.optimize_parameters(optim_ft)
                """
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    display_results(model_full.get_current_visuals(), epoch,epoch_iter, save_result,opt)
                """
                if display_count % 200 == 0:
                    losses = model_full.get_current_losses()
                    if opt.display_id > 0:
                        #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
                        t1 = time.time()
                        elasped = t1 - t0
                        t0 = t1
                        count += elasped 
                        visualizer.print_current_losses(epoch,epoch_iter,losses,elasped,count)
                
            # Validation loss at end of epoch
            with torch.no_grad():
                val_loss = model_full.validate(val_loader, optim_ft, epoch, opt)

            
            if epoch % opt.save_epoch_freq == 0:
                model_full.save_networks('latest')
                model_full.save_networks(epoch)
                display_results(model_full.get_current_visuals(), epoch,opt)
            model_full.update_learning_rate()
            
    else:
        print('Error! Wrong stage selection!')
        exit()
