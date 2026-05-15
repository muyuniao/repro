import argparse
import torch
from data.dataset import get_dataloaders
from models.encoder import EncoderWrapper
from models.classifier import Classifier
from train_stage2 import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/duomeitinrfx/data/HistoricalColor-ECCV2012/data/imgs/decade_database/')
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, _ = get_dataloaders(args.data_dir, batch_size=32)
    loader = test_loader if args.split == 'test' else val_loader
    
    encoder = EncoderWrapper(name=args.encoder, pretrained=False).to(device)
    classifier = Classifier(feature_dim=encoder.feature_dim, num_classes=5).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])
    
    metrics = evaluate(encoder, classifier, loader, device)
    print(f"Metrics on {args.split} set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
