import math
import torch
from torch import nn
from torchsummary import summary
import numpy as np


# Define a common interface for CNN fusers
class BaseConvFuser(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclasses")


class ConvFuser1(BaseConvFuser):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]

        sqrt_T = int(math.sqrt(self.T))

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=T // sqrt_T,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(T // sqrt_T)

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )

        self.conv2 = nn.Conv2d(
            in_channels=T // sqrt_T,
            out_channels=T,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 2)
        )
        self.bn2 = nn.BatchNorm2d(T)

        self.gelu = nn.GELU()

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # Add a channel dimension for Conv2d
        x = x.unsqueeze(1)  # Shape becomes: [batch_size, 1, S, T]

        x = self.conv1(x)  # Shape: [batch_size, T//sqrt(T), S, sqrt(T)]
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.pool1(x)  # Shape: [batch_size, T//sqrt(T), S/2, sqrt(T)/2]

        x = self.conv2(x)  # Shape: [batch_size, T, S/2, 1]
        x = self.bn2(x)
        x = self.gelu(x)

        x = self.global_max_pool(x)  # Shape: [batch_size, T, 1, 1]
        return x


class ConvFuser2(BaseConvFuser):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]

    # Some CNN architecture



class MultivariateTSAD(nn.Module):
    """
    Firs we apply a CNN architecture to fuse sensors in a latent space:
        Given a multivariate time-series S X T, where T is the time-series length and S is the
        number of sensors, MultivariateUnivariateFuser is a CNN that learns the hidden patterns
        as well as flattening the input to a first order tensor, then passing it through a transformer
        architecture.

        For Input Matrix (height and width) --> [batch_size, channels, height, width]
        Model structure assumes: Input shape = (S, T) = (sensors, time-series length)
        Output: [batch_size, T, 1, 1]

    Next we pass the CNN's output to a transformer architecture:
        ...
    """

    def __init__(self, conv_fuser: BaseConvFuser):
        super().__init__()

        # Variables Fuse
        self.conv_fuser = conv_fuser

        # todo: A placeholder. Replace with a transformer.
        self.dnn = nn.Sequential(
            nn.Linear(self.conv_fuser.T, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        # Pass through the CNN fuser
        x = self.conv_fuser(x)

        x = x.squeeze(-1).squeeze(-1)  # Ensure shape is [batch_size, T]

        # Pass through the DNN
        x = self.dnn(x)
        return x


def build_model(input_shape, fuser_name: str = 'ConvFuser1'):

    # Choose a fuser dynamically
    if fuser_name == 'ConvFuser1':
        fuser = ConvFuser1(input_shape)
    else:
        fuser = ConvFuser2(input_shape)

    return MultivariateTSAD(conv_fuser=fuser)



if __name__ == '__main__':

    # Define input dimensions
    S, T = 5, 16000
    input_mat = np.random.randn(S, T)

    # Convert input to PyTorch tensor
    mv_input = torch.tensor(input_mat, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    ad_model = build_model(mv_input[0].shape)

    # Print the model summary
    summary(ad_model, input_size=input_mat.shape)

    obj = ad_model
    print(obj.__class__.__name__)

    exit()