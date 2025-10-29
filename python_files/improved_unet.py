import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),                                   
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): 
        return self.conv(x)

class U_net2(nn.Module):                                                        # We want to improve the U-net by adding regularization 
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]): 
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)                                          # Added dropout for regualarization 

        # Encoder part  

        for feature in features: 
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck part 
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        # Decoder part 
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        reversed_features = features[::-1]
        for feature in reversed_features: 
            self.upconvs.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(feature*2, feature))

        # Final convolution 
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()                                            # Added sigmoid to ensure output is between 0 and 1  but this might make it too constrained and gives bad results

    def forward(self, x): 
        skip_connections = []
        for encode in self.encoder: 
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)): 
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape: 
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True) # We resize x to have the same shape as skip_connection as this is more common in concatenation skip connections
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](concat_skip)
        return self.final_conv(x) 
        #return self.sigmoid(self.final_conv(x))


