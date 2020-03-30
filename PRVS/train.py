import argparse
import os
from model import PRVSNetFull

def train():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--train_root', type=str, default='')
#    parser.add_argument('--train_mask_root', type=str, default='')
#    parser.add_argument('--train_edge_root', type=str, default='')
    parser.add_argument('--train_root', type=str, default='../../Dataset/Pairs_street_view/paris_arged')
    parser.add_argument('--train_mask_root', type=str, default='../../Dataset/irregular_mask/testing_mask_dataset_arged')
    parser.add_argument('--train_edge_root', type=str, default='../Dataset/Pairs_street_view/paris_train_edge')    
    parser.add_argument('--test_root', type=str, default='')
    parser.add_argument('--test_mask_root', type=str, default='')
    parser.add_argument('--test_edge_root', type=str, default='')
    
    parser.add_argument('--val_root', type=str, default='')
    parser.add_argument('--val_mask_root', type=str, default='')
    parser.add_argument('--val_edge_root', type=str, default='')
    
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--nms', type=int, default=1)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--edge', type=int, default=1)
    parser.add_argument('--image_D', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./snapshots/default')
    parser.add_argument('--log_dir', type=str, default='./logs/default')
    parser.add_argument('--train_lr', type=float, default=2e-4)
    parser.add_argument('--finetune_lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_threads', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=120000)
    parser.add_argument('--vis_interval', type=int, default=6000000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--start_iter', type=int, default=60000)
    
    parser.add_argument('--layer_size', type=int, default=8)   
    parser.add_argument('--lambda_tv', type=float, default=0.1)
    parser.add_argument('--lambda_style', type=float, default=150)
    parser.add_argument('--lambda_preceptual', type=float, default=0.05)
    parser.add_argument('--lambda_valid', type=float, default=50)
    parser.add_argument('--lambda_hole', type=float, default=50)
    parser.add_argument('--lambda_adversarial', type=float, default=0.1)
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs('{:s}/images'.format(args.save_dir))
        os.makedirs('{:s}/ckpt'.format(args.save_dir))
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = PRVSNetFull(args)
    model.train()

if __name__ == '__main__':
    train()