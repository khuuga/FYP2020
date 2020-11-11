import os
import time
import torch
#from torch.autograd import Variable
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import io
from skimage import img_as_ubyte

import matplotlib.pyplot as plt
import math
from matplotlib import colors
from tqdm import trange, tqdm

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

class TrainModel(BaseModel):
    def name(self):
        return 'TrainModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G', 'L1']
        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1
        self.optimizers = []
        if opt.stage == 'full' or opt.stage == 'instance':
            self.model_names = ['G']
            self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=opt.classification)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        elif opt.stage == 'fusion':
            self.model_names = ['G', 'GF', 'GComp']
            self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'instance', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=False)
            self.netG.eval()
            
            self.netGF = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'fusion', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=False)
            self.netGF.eval()

            self.netGComp = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=opt.classification)
            self.netGComp.eval()
            self.optimizer_G = torch.optim.Adam(list(self.netGF.module.weight_layer.parameters()) +
                                                list(self.netGF.module.weight_layer2.parameters()) +
                                                list(self.netGF.module.weight_layer3.parameters()) +
                                                list(self.netGF.module.weight_layer4.parameters()) +
                                                list(self.netGF.module.weight_layer5.parameters()) +
                                                list(self.netGF.module.weight_layer6.parameters()) +
                                                list(self.netGF.module.weight_layer7.parameters()) +
                                                list(self.netGF.module.weight_layer8_1.parameters()) +
                                                list(self.netGF.module.weight_layer8_2.parameters()) +
                                                list(self.netGF.module.weight_layer9_1.parameters()) +
                                                list(self.netGF.module.weight_layer9_2.parameters()) +
                                                list(self.netGF.module.weight_layer10_1.parameters()) +
                                                list(self.netGF.module.weight_layer10_2.parameters()) +
                                                list(self.netGF.module.model10.parameters()) +
                                                list(self.netGF.module.model_out.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        else:
            print('Error Stage!')
            exit()
        self.criterionL1 = networks.HuberLoss(delta=1. / opt.ab_norm)
        self.criterionCE = torch.nn.CrossEntropyLoss()
        #self.criterionL1 = networks.L1Loss()
        #self.criterionL1 = torch.nn.CrossEntropyLoss()
        #initialize average loss values
        self.avg_losses = OrderedDict()
        self.avg_loss_alpha = opt.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0
        
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)
        
        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.opt.mask_cent

        self.real_B_enc = util.encode_ab_ind(self.real_B[:, :, ::4, ::4], self.opt)
    
    def set_fusion_input(self, input, box_info):
        AtoB = self.opt.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)

        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)

        self.full_mask_B_nc = self.full_mask_B + self.opt.mask_cent
        self.full_real_B_enc = util.encode_ab_ind(self.full_real_B[:, :, ::4, ::4], self.opt)
        self.box_info_list = box_info

    def forward(self):
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            (self.fake_B_class, self.fake_B_reg) = self.netG(self.real_A, self.hint_B, self.mask_B)
        elif self.opt.stage == 'fusion':
            (_, self.comp_B_reg) = self.netGComp(self.full_real_A, self.full_hint_B, self.full_mask_B)
            (_, self.feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
            self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B, self.full_mask_B, self.feature_map, self.box_info_list)
        else:
            print('Error! Wrong stage selection!')
            exit()

    def optimize_parameters(self, optimize):
        self.forward()
        optimize.zero_grad()
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            #self.loss_G_CE = self.criterionL1(self.fake_B_class.type(torch.cuda.FloatTensor),
            #                                    self.real_B_enc[:, 0, :, :].type(torch.cuda.LongTensor))  
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
            #self.loss_G = self.loss_G_CE * self.opt.lambda_A + self.loss_G_L1_reg
            #self.loss_G = Variable(self.loss_G, requires_grad=True)
        elif self.opt.stage == 'fusion':
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
            #self.loss_G = Variable(self.loss_G, requires_grad=True)
        else:
            print('Error! Wrong stage selection!')
            exit()
        self.loss_G.backward()
        optimize.step()

    def optimize_parameters_fus(self):
        self.forward()
        self.optimizer_G.zero_grad()
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
            #self.loss_G = Variable(self.loss_G, requires_grad=True)
        elif self.opt.stage == 'fusion':
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
            #self.loss_G = Variable(self.loss_G, requires_grad=True)
        else:
            print('Error! Wrong stage selection!')
            exit()
        self.loss_G.backward()
        
        self.optimizer_G.step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            visual_ret['gray'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), torch.zeros_like(self.real_B).type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

            visual_ret['hint'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.hint_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real_ab'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            
        elif self.opt.stage == 'fusion':
            visual_ret['gray'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), torch.zeros_like(self.full_real_B).type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.full_real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['comp_reg'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.comp_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

            self.instance_mask = torch.nn.functional.interpolate(torch.zeros([1, 1, 176, 176]), size=visual_ret['gray'].shape[2:], mode='bilinear').type(torch.cuda.FloatTensor)
            visual_ret['box_mask'] = torch.cat((self.instance_mask, self.instance_mask, self.instance_mask), 1)
            visual_ret['real_ab'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.full_real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['comp_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.comp_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        else:
            print('Error! Wrong stage selection!')
            exit()
        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        self.error_cnt += 1
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                self.avg_losses[name] = float(getattr(self, 'loss_' + name)) + self.avg_loss_alpha * self.avg_losses[name]
                errors_ret[name] = (1 - self.avg_loss_alpha) / (1 - self.avg_loss_alpha**self.error_cnt) * self.avg_losses[name]
        return errors_ret

    def save_fusion_epoch(self, epoch):
        path = '{0}/{1}_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name), epoch)
        latest_path = '{0}/latest_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name))
        torch.save(self.netGF.state_dict(), path)
        torch.save(self.netGF.state_dict(), latest_path)
    
    def validate(self, val_loader, save_images, epoch,opt):
        self.netG.eval()
        log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Validation Loss (%s) ================\n' % now)
        # Prepare value counters and timers
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        end = time.time()
        already_saved_images = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        for dataval in tqdm(val_loader, desc='validation', dynamic_ncols=True, leave=False):
            count += opt.batch_size
            data_time.update(time.time() - end)
            dataval['rgb_img'] = [dataval['rgb_img']]
            dataval['gray_img'] = [dataval['gray_img']]
            input_data = util.get_colorization_data(dataval['gray_img'], opt, p=1.0, ab_thresh=0)
            gt_data = util.get_colorization_data(dataval['rgb_img'], opt, p=1.0, ab_thresh=10.0)
            if gt_data is None:
                continue
            if(gt_data['B'].shape[0] < opt.batch_size):
                continue
            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']

            self.set_input(input_data)
            # Run model and record loss
            self.forward() # throw away class predictions
            #output_ab.cuda()

            loss = torch.mean(self.criterionL1(self.fake_B_reg, input_data['B'].to(device)))
            
            #losses.update(loss.item(), input_gray.size(0))
            """
            # Save images to file
            if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)): # save at most 5 images
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
            """
            # Record time to do forward passes and save images
            #batch_time.update(time.time() - end)
            #end = time.time()

            # Print model accuracy -- in the code below, val refers to both value and validation
            if count % 1600 == 0:
                t1 = time.time()
                elasped = t1 - t0
                t0 = t1
                message = '(epoch: %d, iters: %d, time: %.3f), loss: %.4f ' % (epoch, count, elasped, loss.item())
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
                """
                print('Validate: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))
                """

        print('Finished validation.')
        return losses.avg
    def validatefus(self, val_loader, save_images, epoch,opt):
        self.netG.eval()
        log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Validation Loss (%s) ================\n' % now)
        # Prepare value counters and timers
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        end = time.time()
        already_saved_images = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        for dataval in tqdm(val_loader, desc='validation', dynamic_ncols=True, leave=False):
            count += opt.batch_size
            data_time.update(time.time() - end)
            dataval['full_rgb'] = [dataval['full_rgb']]
            dataval['full_gray'] = [dataval['full_gray']]
            input_data = util.get_colorization_data(dataval['full_gray'], opt, p=1.0, ab_thresh=0)
            gt_data = util.get_colorization_data(dataval['full_rgb'], opt, p=1.0, ab_thresh=10.0)
            if gt_data is None:
                continue
            if(gt_data['B'].shape[0] < opt.batch_size):
                continue
            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']

            self.set_input(input_data)
            # Run model and record loss
            self.forward() # throw away class predictions
            #output_ab.cuda()

            loss = torch.mean(self.criterionL1(self.fake_B_reg, input_data['B'].to(device)))
            
            #losses.update(loss.item(), input_gray.size(0))
            """
            # Save images to file
            if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)): # save at most 5 images
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
            """
            # Record time to do forward passes and save images
            #batch_time.update(time.time() - end)
            #end = time.time()

            # Print model accuracy -- in the code below, val refers to both value and validation
            if count % 1600 == 0:
                t1 = time.time()
                elasped = t1 - t0
                t0 = t1
                message = '(epoch: %d, iters: %d, time: %.3f), loss: %.4f ' % (epoch, count, elasped, loss.item())
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
                """
                print('Validate: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))
                """

        print('Finished validation.')
        return losses.avg
    """
    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                print(n)
                print(p.requires_grad)
                print(type(p.grad))
                
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig('grad_plot.png')
        """
