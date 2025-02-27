
import timm
import torch.nn as nn 


class Efficient(nn.Module):
    def __init__(self,linear_proj = 256,**kwargs):
        super(Efficient, self).__init__()
        self.main_cnn  = timm.create_model('efficientnet_b4.ra2_in1k', features_only=True, pretrained=True)
        self.linear_proj = nn.Linear(self.main_cnn.blocks[-1][-1].conv_pwl.out_channels, linear_proj)
    def forward(self, x):
        """
        Shape:
            - x: (N, T, H, W)
            - output: (T, N, D)
        """
        # N,D,T
        features =  self.main_cnn(x)[-1].flatten(2)
        
        #T,N,D
        features = features.permute(2,0,1)
        #projection for match the dimention of the decoder input
        return self.linear_proj(features) 
    
# model = Efficient()
# total_params = 0 
# for param in model.parameters():
#     total_params += param.numel()
# print("Total params: %d" % total_params)


# import torch
# x = torch.randn(2,3,600,600)
# with torch.no_grad():
#     features = model(x)

# import pdb;pdb.set_trace()