import torch
from modules.PRVSNet import PRVSNet, VGG16FeatureExtractor
from modules.Losses import AdversarialLoss
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torch.utils.data import DataLoader
from dataset import Dataset
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.nn.functional as F
from modules.Discriminator import Discriminator
import torch.nn as nn
import os
import time

class PRVSNetFull():
    def __init__(self, opt):
        self.opt = opt
        self.G = PRVSNet(layer_size = opt.layer_size)
        self.lossNet = VGG16FeatureExtractor()
        
        if opt.finetune:
            self.lr = opt.finetune_lr
            self.G.freeze_enc_bn = True
        else:
            self.lr = opt.train_lr
        self.start_iter = 0  
        self.optm_G = optim.Adam(self.G.parameters(), lr = self.lr)
        
        if opt.resume or opt.finetune:
            start_iter = load_ckpt(opt.save_dir + "/ckpt/g_{:d}.pth".format(opt.start_iter), [('generator', self.G)], [('optimizer_G', self.optm_G)])
            self.optm_G = optim.Adam(self.G.parameters(), lr = self.lr)
            for param_group in self.optm_G.param_groups:
                param_group['lr'] = self.lr
            print('Starting from iter ', start_iter)
            self.start_iter = start_iter
        
        self.edge_D = Discriminator(2)
        self.optm_ED = optim.Adam(self.edge_D.parameters(), lr = self.lr * 0.1)
        self.edge_loss = self.edge_net_loss
        self.have_D = True
        self.calculate_adversarial_loss = AdversarialLoss()
        if opt.resume or opt.finetune:
            start_iter = load_ckpt(opt.save_dir + "/ckpt/d_{:d}.pth".format(opt.start_iter), [('edge_D', self.edge_D)], [('optimizer_ED', self.optm_ED)])
            self.optm_ED = optim.Adam(self.edge_D.parameters(), lr = self.lr * 0.1)
            for param_group in self.optm_ED.param_groups:
                param_group['lr'] = self.lr
        
        
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.edge_generated = None
        self.edge_comp = None
        self.edge_gt = None
        self.edge_list = None
        self.gray_image = None
        self.l1_loss = 0.0
        self.D_loss = 0.0
        self.E_loss = 0.0

        
        if torch.cuda.is_available():
            self.device = torch.device(opt.device)
            if opt.device == "cuda":
                if torch.cuda.device_count() > 1:
                    self.G = nn.DataParallel(self.G)
                    self.lossNet = nn.DataParallel(self.lossNet)
                    self.calculate_adversarial_loss = nn.DataParallel(self.calculate_adversarial_loss)
                    self.edge_D = nn.DataParallel(self.edge_D)
                self.G.cuda()
                self.lossNet.cuda()
                if self.have_D:
                    self.calculate_adversarial_loss.cuda()
                    self.edge_D.cuda()
        else:
            self.device = torch.device("cpu")
        
        if self.opt.mode == 2:
            self.test_dataset = Dataset(opt, opt.test_root, opt.test_edge_root, opt.test_mask_root, augment=False, training=False, mask_reverse = True)
        else:
            self.train_dataset = Dataset(opt, opt.train_root, opt.train_edge_root, opt.train_mask_root, augment=True, mask_reverse = True)
            self.val_dataset = Dataset(opt, opt.val_root, opt.val_edge_root, opt.val_mask_root, augment=False, training=True, mask_reverse = True)
            self.sample_iterator = self.val_dataset.create_iterator(opt.batch_size)
        
    def train(self):
        writer = SummaryWriter(log_dir="log_info")
        self.G.train()
        if self.opt.finetune:
            print("here")
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = self.lr)
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_threads,
            drop_last=True,
            shuffle=True
        )
        keep_training = True
        epoch = 0
        i = self.start_iter
        print("starting training")
        s_time = time.time()
        while keep_training:
            epoch += 1
            print("epoch: {:d}".format(epoch))
            for items in train_loader:
                i += self.opt.batch_size
                gt_images, gray_image, gt_edges, masks = self.cuda(*items)
              #  masks = torch.cat([masks]*3, dim = 1)
                self.gray_image = gray_image
                
                masked_images = gt_images * masks
                masked_edges = gt_edges * masks[:,0:1,:,:]
                
                self.forward(masked_images, masks, masked_edges, gt_images, gt_edges)
                self.update_parameters()

                if i % self.opt.log_interval == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("epoch:{:d}, iteration:{:d}".format(epoch, i), ", l1_loss:", self.l1_loss*self.opt.batch_size/self.opt.log_interval, ", time_taken:", int_time)
                    writer.add_scalars("loss_val", {"l1_loss":self.l1_loss*self.opt.batch_size/self.opt.log_interval, "D_loss":self.D_loss/self.opt.log_interval,"E_loss":self.E_loss*self.opt.batch_size/self.opt.log_interval}, i)
                    masked_images = masked_images.cpu()
                    fake_images = self.fake_B.cpu()
                    fake_edges = self.edge_fake[1].cpu()
                    fake_edges = torch.cat([fake_edges]*3, dim = 1)
                    images = torch.cat([masked_images[0:3], fake_images[0:3], fake_edges[0:3]], dim = 0)
                    writer.add_images("imgs", images, i)
                    s_time = time.time()
                    self.l1_loss = 0.0
                    self.D_loss = 0.0
                    self.E_loss = 0.0
                    
                if i % self.opt.save_interval == 0:
                    save_ckpt('{:s}/ckpt/g_{:d}.pth'.format(self.opt.save_dir, i ), [('generator', self.G)], [('optimizer_G', self.optm_G)], i )
                    if self.have_D:
                        save_ckpt('{:s}/ckpt/d_{:d}.pth'.format(self.opt.save_dir, i ), [('edge_D', self.edge_D)], [('optimizer_ED', self.optm_ED)], i )
                    
        writer.close()
        
        
    def test(self):
        test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=1
                )
        self.G.eval()
        count = 0
        for items in test_loader:
        
            gt_images, _, masked_edges, masks = self.cuda(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks]*3, dim = 1)
            fake_B, _, _, edges = self.G(masked_images, masks, masked_edges)
            comp_B = fake_B * (1 - masks[:,0:1,:,:]) + gt_images * masks[:,0:1,:,:]
            comp_edges = edges * (1 - masks[:,0:1,:,:])  + masked_edges
            masked_images = masked_images + (1 - masks[:,0:1,:,:])
            comp_edges = 1 - comp_edges
            if not os.path.exists('{:s}/images/result_final'.format(self.opt.save_dir)):
                os.makedirs('{:s}/images/result_final'.format(self.opt.save_dir))
            for k in range(comp_B.size(0)):
                count += 1
                grid = make_grid(comp_B[k:k+1])
                file_path = '{:s}/images/result_final/img_{:d}.png'.format(self.opt.save_dir, count)
                save_image(grid, file_path)
                
                grid = make_grid(comp_edges[k:k+1])
                file_path = '{:s}/images/result_final/edge_{:d}.png'.format(self.opt.save_dir, count)
                save_image(grid, file_path)
                
                grid = make_grid(masked_images[k:k+1])
                file_path = '{:s}/images/result_final/masked_img_{:d}.png'.format(self.opt.save_dir, count)
                save_image(grid, file_path)
                
                grid = make_grid(gt_images[k:k+1])
                file_path = '{:s}/images/result_final/gt_img_{:d}.png'.format(self.opt.save_dir, count)
                save_image(grid, file_path)
    
    
    def forward(self, masked_image, mask, masked_edge, gt_image, gt_edge):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        self.edge_gt = gt_edge
        
        fake_B, _, edge_small, edge_big = self.G(masked_image, mask, masked_edge)
        self.fake_B = fake_B
        ##mask list should correspond to edge_list, in both length and size of element
        self.edge_fake = [edge_small, edge_big]
        
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.updateG()
        self.updateD()
    
    def updateG(self):
        self.optm_G.zero_grad()
        ##calculate the loss of G
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        real_edge = self.edge_gt
        fake_edge = self.edge_fake
        
        tv_loss = self.calculate_TV_loss(comp_B * (1 - self.mask))
        style_loss = self.calculate_style_loss(real_B_feats, fake_B_feats) + self.calculate_style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.calculate_preceptual_loss(real_B_feats, fake_B_feats) + self.calculate_preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = F.l1_loss(real_B * self.mask , fake_B * self.mask)
        hole_loss = F.l1_loss(real_B * (1 - self.mask), fake_B * (1 - self.mask))
        

        adv_loss_0, edge_loss_0, _ = self.calclulate_edge_loss(fake_edge[1], real_edge, contain_l1 = False)
        adv_loss_1, edge_loss_1, _ = self.calclulate_edge_loss(fake_edge[0], F.interpolate(real_edge, scale_factor = 0.5), contain_l1 = False)
        
        edge_loss = edge_loss_0 + edge_loss_1
        adv_loss = adv_loss_0 + adv_loss_1
        
        loss_G = (  tv_loss * self.opt.lambda_tv
                  + style_loss * self.opt.lambda_style
                  + preceptual_loss * self.opt.lambda_preceptual
                  + valid_loss * self.opt.lambda_valid
                  + hole_loss * self.opt.lambda_hole) + self.opt.lambda_adversarial * adv_loss
        self.l1_loss += (hole_loss + valid_loss).cpu().detach().numpy()
        self.E_loss += edge_loss.cpu().detach().numpy()
        loss_G.sum().backward()
        self.optm_G.step()
        
    
    def updateD(self):
        self.optm_ED.zero_grad()


        real_edges = self.edge_gt
        fake_edges = self.edge_fake
        loss_D = 0

        
        for i in range(2):
            fake_edge = fake_edges[i]
            real_edge = F.interpolate(real_edges, size = fake_edge.size()[2:])
            real_edge = torch.clamp(real_edge*4, 0, 1)
            if self.opt.image_D:
                real_image = F.interpolate(self.real_B, size = fake_edge.size()[2:])
                fake_image = F.interpolate(self.fake_B.detach(), size = fake_edge.size()[2:])
            else:
                real_image = F.interpolate(self.gray_image, size = fake_edge.size()[2:])
                fake_image = real_image
            real_edge = real_edge.detach()
            fake_edge = fake_edge.detach()
            real_edge = torch.cat([real_edge, real_image], dim = 1)
            fake_edge = torch.cat([fake_edge, fake_image], dim = 1)
    
            pred_real, _ = self.edge_D(real_edge)
            pred_fake, _ = self.edge_D(fake_edge)
            
            loss_D += (self.calculate_adversarial_loss(pred_real, True, True)  + self.calculate_adversarial_loss(pred_fake, False, True))/2

        loss_D.sum().backward()
        self.optm_ED.step()
        self.D_loss += loss_D.cpu().detach().numpy()
    
    def edge_net_loss(self, fake_edge, real_edge):
        losses = 0.0
        if self.opt.image_D:
            gray_image = self.fake_B
        else:
            gray_image = self.gray_image
        gray_image = F.interpolate(gray_image, size = fake_edge.size()[2:])
        fake_edge = torch.cat([fake_edge, gray_image], dim = 1)
        real_edge = torch.clamp(real_edge*4, 0, 1)
        real_edge = torch.cat([real_edge, gray_image], dim = 1)
        pred_fake, features_edge1 = self.edge_D(fake_edge)
        
        
        _, features_edge2 = self.edge_D(real_edge)
        for feature_edge1, feature_edge2 in zip(features_edge1, features_edge2):
            losses += F.l1_loss(feature_edge1, feature_edge2)
        return pred_fake, losses
    
    def calculate_style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += F.l1_loss(A_style/(c * w * h), B_style/(c * w * h))
        return loss_value
    
    
    def calclulate_edge_loss(self, fake_edge, real_edge, contain_l1 = True):
        if contain_l1:
            mask = self.mask
            mask = F.interpolate(mask, size = fake_edge.size()[2:])
            temp_real = torch.clamp(real_edge*4, 0, 1)
            l1_loss = F.l1_loss(fake_edge * mask[:,0:1,:,:], temp_real * mask[:,0:1,:,:])
        else:
            l1_loss = 0.0
        pred_fake, FM_loss = self.edge_loss(fake_edge, real_edge)
        return self.calculate_adversarial_loss(pred_fake, True, False), FM_loss, l1_loss
    
    def calculate_TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def calculate_preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += F.l1_loss(A_feat, B_feat)
        return loss_value
            
    def cuda(self, *args):
        return (item.to(self.device) for item in args)
            