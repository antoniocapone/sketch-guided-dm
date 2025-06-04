import torch
from torch import nn
import math
import numpy as np
from einops import rearrange

class LGP(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(LGP, self).__init__()
        self.num_encodings = num_encodings

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, t):

        # x: (batch_size, channels, height / 8, width / 8)      latent_image
        # t: (batch_size, channels, height / 8, width / 8)      predicted_noise

        # lista di tensori 4-dimensionali
        pos_elem = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]

        print(np.array(pos_elem).shape)

        # ([2, 36, 64, 64]): batch_size, channels, height / 8, width / 8
        pos_encoding = torch.cat(pos_elem, dim=1)

        print(pos_encoding.shape)
        
        # concateno immagine latente, rumore predetto ed encoding posizionale lungo la dimensione dei canali
        x = torch.cat((x, t, pos_encoding), dim=1)

        # permuto le dimensioni in modo da poter isolare il numero di canali e preparare i dati per il layer lineare

        x = x.permute(0, 2, 3, 1)

        x = x.flatten(start_dim = 0, end_dim = 2)
        
        return self.layers(x[:, :7080])
    
