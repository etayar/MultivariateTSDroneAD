from torch import nn


class MultivariateUnivariateFuserAD(nn.Module):
    """
    Given a multivariate time-series (T X S), where T is the time-series length and S is the
    number of sensors, MultivariateUnivariateFuser is a CNN that learns the hidden patterns
    as well as flattening the input to a first order tensor, then passing it through a transformers
    architecture.

    For Input Matrix (height and width) --> [batch_size, channels, height, width]
    """

    def __init__(self, input_shape, batch_size):
        super().__init__()
        self.T = input_shape[0]
        self.S = input_shape[1]
        self.batch_size = batch_size

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,             # Input channel (1 for single-channel input)
            out_channels=T // int(T),  # Output feature maps = T / int(T)
            kernel_size=(3, 3),        # Kernel size (example: 3x3, adjust if needed)
            stride=1,
            padding=1                  # Maintain spatial dimensions
        )

        # First Pooling Layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),  # Pooling size 2x2
            stride=(2, 2)        # Stride 2x2 for downsampling
        )

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=T // int(T),  # Input channels from previous layer
            out_channels=1,           # Output feature maps = 1
            kernel_size=(3, 3),       # Kernel size (example: 3x3, adjust if needed)
            stride=1,
            padding=1                 # Maintain spatial dimensions
        )

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d((T, 1))  # Global pooling to [T, 1]
