import os
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import yaml
import torch
import torch.optim as optim
from data.dataset import get_dataloaders
from models.encoder import EncoderWrapper
from models.ranker import Ranker
from models.losses import SoftSFDLoss
from utils import AverageMeter

def train_epoch(encoder, ranker, loader, optimizer, criterion, device, tau=1.0):
    encoder.train()
    ranker.train()
    losses = AverageMeter()
    
    for bag_imgs, bag_labels in loader:
        bag_imgs = bag_imgs.to(device)
        B, K, C, H, W = bag_imgs.shape
        
        x = bag_imgs.view(B * K, C, H, W)
        features = encoder(x)
        features = features.view(B, K, -1)
        
        scores, p_hat = ranker(features, tau=tau)
        
        target_ranks = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        loss = criterion(p_hat, target_ranks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), B)
        
    return losses.avg

def main():
    # 1. 预解析 --config 参数
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--config', type=str, default='', help='Path to YAML config file')
    partial_args, remaining_argv = conf_parser.parse_known_args()
    
    # 2. 定义完整的参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default='/home/duomeitinrfx/data/HistoricalColor-ECCV2012/data/imgs/decade_database/')
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # 3. 如果指定了配置文件，先加载配置文件并作为默认值
    if partial_args.config:
        if os.path.exists(partial_args.config):
            print(f"Loading config from {partial_args.config}...")
            with open(partial_args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            if config_dict:
                parser.set_defaults(**config_dict)
        else:
            print(f"Warning: Config file {partial_args.config} not found. Using command line defaults.")
            
    args = parser.parse_args(remaining_argv)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, _, train_bag_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    encoder = EncoderWrapper(name=args.encoder, pretrained=True).to(device)
    ranker = Ranker(feature_dim=encoder.feature_dim).to(device)
    
    criterion = SoftSFDLoss()
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': args.lr},
        {'params': ranker.parameters(), 'lr': args.lr * 5}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        tau = max(0.05, 0.5 - (0.45 * epoch / args.epochs))
        loss = train_epoch(encoder, ranker, train_bag_loader, optimizer, criterion, device, tau)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Tau: {tau:.4f}")
        
    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'ranker': ranker.state_dict()
    }, f"results/checkpoints/stage1_{args.encoder}.pt")
    print("Stage 1 saved.")

if __name__ == '__main__':
    main()
