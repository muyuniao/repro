import timm
import torch.nn as nn

def get_encoder(name='resnet50', pretrained=True):
    if 'resnet' in name:
        model = timm.create_model('resnet50.a1_in1k', pretrained=pretrained, num_classes=0, global_pool='')
        feature_dim = 2048
    elif 'vit' in name:
        model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, num_classes=0)
        feature_dim = 768
    else:
        raise ValueError(f"Unknown encoder {name}")
        
    return model, feature_dim

class EncoderWrapper(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        self.encoder, self.feature_dim = get_encoder(name, pretrained)
        if 'resnet' in name:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = None
            
    def forward(self, x):
        features = self.encoder(x)
        if self.pool is not None:
            features = self.pool(features).flatten(1)
        return features
