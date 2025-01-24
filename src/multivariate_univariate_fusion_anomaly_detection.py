import math
from torch import nn


class ConvFuser1(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]

        sqrt_T = int(math.sqrt(T))

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


class ConvFuser2(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]



class MultivariateTSAD(nn.Module):
    """
    Given a multivariate time-series S X T, where T is the time-series length and S is the
    number of sensors, MultivariateUnivariateFuser is a CNN that learns the hidden patterns
    as well as flattening the input to a first order tensor, then passing it through a transformer
    architecture.

    For Input Matrix (height and width) --> [batch_size, channels, height, width]
    Model structure assumes: Input shape = (S, T) = (sensors, time-series length)
    """

    def __init__(self, conv_fuser: str = ''):
        super().__init__()



if __name__ == '__main__':
    from torchsummary import summary
    import numpy as np

    # Define input dimensions
    S, T = 64, 640
    input_mat = np.random.randn(S, T)
    model = ConvFuser1(input_mat.shape)

    # Print the model summary
    summary(model, input_size=input_mat.shape)


    class MyClass:

        def __init__(self, x):
            self.x = x

        def my_methode(self, y):
            return self.x + y


    obj = MyClass(10)
    print(obj.__class__.__name__)  # Output: MyClass

    print(type(obj).__name__)  # Output: MyClass

    exit()