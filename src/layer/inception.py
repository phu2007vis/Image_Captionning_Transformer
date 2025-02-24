import timm
model = timm.create_model('inception_resnet_v2', pretrained=True)
import torch
    
if __name__ == '__main__':
    device = 'cpu'
    # model = Inception_ResNetv2().to(device=device)
    x = torch.randn(2,3,500,500).to(device= device)
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total parameters: {total_params}')
    print(model(x).shape)