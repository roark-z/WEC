import torch
import torch.nn as nn
import torch.nn.functional as F

# first feature is input features
features = [3, 8, 16, 32]

# Encoder for input frames. Assume that frame is a multiple of 2**len(features)-1 for now
# TODO introduce padding

class FrameEncoder(nn.Module):
    
    '''
    (int, int)  frame_size      : resolution of input frame
    int         out_features    : length of output feature vector 
    '''

    def __init__(self, frame_dims, out_features):
        super(FrameEncoder, self).__init__()

        # Define layers
        self.layers = []
        n_layers = len(features)-1

        for ii in range(0, n_layers):
            size = features[ii]
            next_size = features[ii+1]

            # kernel size = 3
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(size, next_size, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )

        total_input = (frame_dims[0]>>n_layers) * (frame_dims[1]>>n_layers)

        # Define final layer
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(total_input, out_features)
        )


    def forward(self, x):
        x_prime = x
        for layer in self.layers:
            x_prime = layer(x_prime)
            print(x_prime.shape)


        y = self.linear(x_prime)

        return y