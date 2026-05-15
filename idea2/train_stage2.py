import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloaders
from models.encoder import EncoderWrapper
from models.classifier import Classifier
from models.ranker import Ranker
from models.losses import SoftSFDLoss, CORALLoss
from utils import AverageMeter, compute_metrics

def train_epoch(encoder, classifier, ranker, loader, bag_loader, optimizer, criterion_cls, criterion_rank, device, mode, lam=0.0, tau=0.05):
    encoder.train()
    classifier.train()
    if ranker is not None:
        ranker.train()
    losses = AverageMeter()
    
    bag_iter = iter(bag_loader) if mode == 's2.2' else None
    
    for imgs, labels in loader:
        optimizer.zero_grad()
        
        # Pass 1: Classification
        imgs, labels = imgs.to(device), labels.to(device)
        features = encoder(imgs)
        logits = classifier(features)
        loss_cls = criterion_cls(logits, labels)
        
        # If S2.2, we backward classification first to free its activations
        if mode == 's2.2':
            loss_cls.backward()
            
            # Pass 2: Ranking
            try:
                bag_imgs, _ = next(bag_iter)
            except StopIteration:
                bag_iter = iter(bag_loader)
                bag_imgs, _ = next(bag_iter)
                
            bag_imgs = bag_imgs.to(device)
            B, K, C, H, W = bag_imgs.shape
            x = bag_imgs.view(B * K, C, H, W)
            
            # Forward bag in chunks if necessary, but here we try 80 at once
            bag_feats = encoder(x).view(B, K, -1)
            scores, p_hat = ranker(bag_feats, tau=tau)
            target_ranks = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
            loss_rank = lam * criterion_rank(p_hat, target_ranks)
            
            loss_rank.backward()
            total_loss_val = loss_cls.item() + loss_rank.item()
        else:
            loss_cls.backward()
            total_loss_val = loss_cls.item()
            
        optimizer.step()
        losses.update(total_loss_val, imgs.size(0))
        
    return losses.avg

def evaluate(encoder, classifier, loader, device, is_coral=False):
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            features = encoder(imgs)
            logits = classifier(features)
            if is_coral:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
            else:
                preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return compute_metrics(all_preds, all_targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/duomeitinrfx/data/HistoricalColor-ECCV2012/data/imgs/decade_database/')
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--stage1_ckpt', type=str, default='')
    parser.add_argument('--mode', type=str, default='s2.1', choices=['s2.1', 's2.2', 's2.3'])
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'coral'])
    parser.add_argument('--exp_name', type=str, default='stage2_exp')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, train_bag_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    encoder = EncoderWrapper(name=args.encoder, pretrained=True).to(device)
    ranker = None
    
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        ckpt = torch.load(args.stage1_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"Loaded Stage 1 Encoder weights from {args.stage1_ckpt}")
        if args.mode == 's2.2':
            ranker = Ranker(feature_dim=encoder.feature_dim).to(device)
            ranker.load_state_dict(ckpt['ranker'])
            print(f"Loaded Stage 1 Ranker weights from {args.stage1_ckpt}")
            
    if args.mode == 's2.3':
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for S2.3")
        
    num_classes = 5
    classifier_out = num_classes - 1 if args.loss == 'coral' else num_classes
    classifier = Classifier(feature_dim=encoder.feature_dim, num_classes=classifier_out).to(device)
    
    criterion_cls = CORALLoss(num_classes=num_classes) if args.loss == 'coral' else nn.CrossEntropyLoss()
    criterion_rank = SoftSFDLoss() if args.mode == 's2.2' else None
    
    param_groups = [{'params': classifier.parameters(), 'lr': args.lr}]
    if args.mode != 's2.3':
        param_groups.append({'params': encoder.parameters(), 'lr': args.lr * 0.01})
    if args.mode == 's2.2':
        param_groups.append({'params': ranker.parameters(), 'lr': args.lr * 0.05})
        
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_mae = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.epochs):
        lam = max(0.0, 0.5 * (1.0 - epoch / (args.epochs * 0.7))) if args.mode == 's2.2' else 0.0
        
        loss = train_epoch(encoder, classifier, ranker, train_loader, train_bag_loader, optimizer, criterion_cls, criterion_rank, device, args.mode, lam=lam)
        scheduler.step()
        
        val_metrics = evaluate(encoder, classifier, val_loader, device, is_coral=(args.loss=='coral'))
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val ACC: {val_metrics['acc']:.4f}")
        
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            patience_counter = 0
            os.makedirs('results/checkpoints', exist_ok=True)
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }, f"results/checkpoints/{args.exp_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
    print(f"Testing Best Model for {args.exp_name}...")
    ckpt = torch.load(f"results/checkpoints/{args.exp_name}_best.pt", map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])
    
    test_metrics = evaluate(encoder, classifier, test_loader, device, is_coral=(args.loss=='coral'))
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
