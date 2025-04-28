import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPProjector, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        print(x.shape)
        return self.net(x)

class UnmixCLIP(nn.Module):
    def __init__(self, clip_model, image_projector, text_projector):
        super(UnmixCLIP, self).__init__()
        self.clip_model = clip_model
        self.image_projector = image_projector
        self.text_projector = text_projector
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images) 
            image_features = image_features[:, 0, :]
        img_proj = self.image_projector(image_features)          
        img_proj = F.normalize(img_proj, dim=-1)
        return img_proj

    def forward_text(self, text_tokens):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        text_proj = self.text_projector(text_features)
        text_proj = F.normalize(text_proj, dim=-1)
        return text_proj
