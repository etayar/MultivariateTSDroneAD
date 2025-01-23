from torch import nn


class MultivariateUnivariateFuserAD(nn.Module):
    """
    Given a multivariate time-series (T X S), where T is the time-series length and S is the
    number of sensors, MultivariateUnivariateFuser is a CNN that learns the hidden patterns
    as well as flattening the input to a first order tensor, then passing it through a transformers
    architecture.

    For Input Matrix (height and width) --> [batch_size, channels, height, width]
    Model structure assumes: Input shape = (T X S) = (time-series length, sensors)
    """

    def __init__(self, input_shape):
        super().__init__()
        self.T = T
        self.S = S

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,             # Single input channel (for S x T matrix)
            out_channels=T // int(T),  # T/int(T) feature maps
            kernel_size=(3, 3),        # Kernel size (can be adjusted)
            stride=1,
            padding=1                  # Padding to maintain spatial dimensions
        )

        # First Pooling Layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),  # Pooling size
            stride=(2, 2)        # Stride reduces spatial dimensions
        )

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=T // int(T),  # Input channels from the previous layer
            out_channels=1,           # Output feature map with 1 channel
            kernel_size=(3, 3),       # Kernel size
            stride=1,
            padding=1                 # Padding to maintain spatial dimensions
        )

        # Global Max Pooling Layer
        self.global_pool = nn.AdaptiveMaxPool2d((T, 1))  # Global pooling to [T, 1]
