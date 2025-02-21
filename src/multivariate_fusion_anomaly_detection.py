import math
import torch
from torch import nn
from torchinfo import summary
from transformers import LongformerModel, LongformerConfig
from performer_pytorch import Performer
from linformer import Linformer


# Define a common interface for CNN fusers
class BaseConvFuser(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclasses")


class ConvFuser1(BaseConvFuser):

    def __init__(self, input_shape, time_scaler=1):
        super().__init__()
        self.S = input_shape[0]  # Number of sensors
        self.T = input_shape[1]  # Time-series length
        self.time_scaler = time_scaler

        sqrt_T = int(math.sqrt(self.T))

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.T // sqrt_T,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(self.T // sqrt_T)

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )

        output_T = int(self.time_scaler * self.T)

        self.conv2 = nn.Conv2d(
            in_channels=self.T // sqrt_T,
            out_channels=output_T,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 2)
        )
        self.bn2 = nn.BatchNorm2d(output_T)

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


class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        super(ResNetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Proper down sampling layer
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class ResNet2D(BaseConvFuser):
    def __init__(self, input_shape, blocks=(2, 2, 2), hidden_dim=32, time_scaler=1):
        super().__init__()

        # Monitor the input tensor
        self.S = input_shape[0]  # Number of sensors
        self.T = input_shape[1]  # Time-series length
        print(f"Number of sensors: {self.S}")
        print(f"Time-series length: {self.T}")

        self.in_channels = hidden_dim  # Initial input channels

        # **Initial 2D Conv Layer**
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # **ResNet Blocks**
        self.layers = nn.ModuleList()
        in_channels = hidden_dim  # Track in_channels dynamically

        for i in range(len(blocks)):
            stride = 1 if i == 0 else 2  # First layer keeps stride=1, others use stride=2
            out_channels = in_channels * 2  # Double the channels each time
            self.layers.append(self._make_layer(in_channels, out_channels, blocks[i], stride=stride))
            in_channels = out_channels  # Update in_channels after each layer

        # Projection Layer: Ensures Output is `int(time_scaler * self.T)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, int(time_scaler * self.T), kernel_size=1),  # Reduce to int(time_scaler * self.T)-channel time-series
            nn.AdaptiveAvgPool2d((1, 1))  # Ensures time length = int(time_scaler * self.T)
        )

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBlock2D(in_channels, out_channels, stride=stride)]
        layers.extend(ResNetBlock2D(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, S, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # **Apply ResNet Blocks**
        for layer in self.layers:
            x = layer(x)

        # Project to 1D Time-Series of Length `int(time_scaler * self.T)
        x = self.projection(x)  # (batch, int(time_scaler * self.T), 1, 1)
        return x



class ConvFuser3(BaseConvFuser):

    def __init__(self, input_shape):
        super().__init__()
        self.S = input_shape[0]  # Number of sensors
        self.T = input_shape[1]  # Time-series length

    # Some CNN architecture


def get_transformer_variant(
        variant: str,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_len: int,
        dropout: float
):

    if variant == "vanilla":
        '''
        Quadratic memory and time complexity with respect to sequence length (O(T²)), 
        making it inefficient for very long sequences.
        
        Short to medium-length sequences (e.g., < 512 tokens).
        '''
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    elif variant == "longformer":
        '''
        Handles very long sequences (e.g., 4K-8K tokens) efficiently.
        Local attention reduces complexity to O(T × W) where W is the local window size, significantly improving scalability.
        
        Slightly less flexible in learning global relationships compared to full attention.
        '''
        long_former_config = LongformerConfig(attention_window=512, hidden_size=d_model, num_attention_heads=nhead)
        transformer = LongformerModel(long_former_config)

    elif variant == "linformer":
        '''
        Uses a low-rank approximation of the attention matrix, reducing the complexity 
        of self-attention from O(T²) to O(T × k), where k is the rank of the approximation.
        
        Drastically reduces memory requirements for self-attention.
        Retains competitive performance on tasks with longer sequences.
        
        Approximation may lose some fine-grained global details, especially for tasks requiring high precision.
        '''
        transformer = Linformer(
            dim=d_model,
            seq_len=max_len,
            depth=num_layers,
            heads=nhead,
            k=256  # Low-rank approximation dimension
        )

    elif variant == "performer":
        '''
        A transformer that uses kernelized attention to approximate full attention with linear complexity, 
        scaling as O(T). It employs random feature maps for efficient computation.
        
        Approximation of attention may slightly degrade performance on certain tasks compared to full attention.
        
        Very long sequences (e.g., >10K tokens).
        Time-series forecasting, long document processing, or real-time anomaly detection in massive datasets.
        '''
        transformer = Performer(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
            dim_head=d_model // nhead,
            causal=False  # Non-causal for bidirectional attention
        )

    else:
        raise ValueError(f"Unknown transformer variant: {variant}")
    return transformer


# Positional Encoding (Fixed Sinusoidal)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Get positional indices
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_encoding = self.pos_embedding(positions)
        return x + pos_encoding


class AttentionAggregator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)  # Map each time step to a single weight
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        x: [T, batch_size, d_model]
        Returns: [batch_size, d_model]
        """
        # Compute attention weights for each time step
        weights = self.softmax(self.attention_weights(x))  # Shape: [T, batch_size, 1]
        # Weighted sum of sequence (learnable aggregation)
        x = (weights * x).sum(dim=0)  # Shape: [batch_size, d_model]
        return x


class LinearAggregator(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.projection = nn.Linear(seq_len, 1)  # Reduce the T dimension to 1

    def forward(self, x):
        """
        x: [T, batch_size, d_model]
        Returns: [batch_size, d_model]
        """
        x = x.permute(1, 2, 0)  # Reshape to [batch_size, d_model, T]
        x = self.projection(x).squeeze(-1)  # Shape: [batch_size, d_model]
        return x


# class ConvAggregator(nn.Module):
#     def __init__(self, d_model, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv1d(
#             in_channels=d_model,
#             out_channels=d_model,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size // 2
#         )
#         self.global_pool = nn.AdaptiveMaxPool1d(1)  # Aggregate down to 1
#
#     def forward(self, x):
#         """
#         x: [T, batch_size, d_model]
#         Returns: [batch_size, d_model]
#         """
#         x = x.permute(1, 2, 0)  # Reshape to [batch_size, d_model, T]
#         x = self.conv(x)  # Shape: [batch_size, d_model, T]
#         x = self.global_pool(x).squeeze(-1)  # Shape: [batch_size, d_model]
#         return x


class ResNetBlock1D(nn.Module):
    def __init__(self, d_model, kernel_size=3, stride=1, padding=1):
        """
        Standard 1D ResNet block with optional down sampling.
        - No channel expansion, keeps everything within d_model.
        """
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(d_model)

        # If stride > 1, adjust identity connection
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Conv1d(d_model, d_model, kernel_size=1, stride=stride, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # Store input for skip connection

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Adjust skip connection if needed
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class ConvAggregator(nn.Module):
    def __init__(self, d_model, blocks=(2, 2, 2)):
        """
        Args:
            d_model: Feature dimension remains unchanged throughout the network.
            blocks: Tuple defining the number of ResNet blocks per stage.
        """
        super(ConvAggregator, self).__init__()
        self.d_model = d_model

        # Initial 1D convolution, mapping d_model → d_model (no channel expansion)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Creating ResNet blocks with fixed d_model
        self.layers = nn.ModuleList()
        for num_blocks in blocks:
            self.layers.append(self._make_layer(d_model, num_blocks))

        # Global Average Pooling to reduce sequence to a fixed-size representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, d_model)  # Ensuring final output is [batch_size, d_model]

    def _make_layer(self, d_model, num_blocks, stride=1):
        """
        Creates a block of ResNet layers, keeping d_model unchanged.
        """
        layers = []

        # If we need down sampling, apply stride only in the first block
        layers.append(ResNetBlock1D(d_model, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock1D(d_model, stride=1))  # No stride in deeper blocks

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(1, 2, 0)  # Convert [T, batch, d_model] -> [batch, d_model, T]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Apply all ResNet layers
        for layer in self.layers:
            x = layer(x)

        x = self.global_avg_pool(x)  # Shape: [batch, d_model, 1]
        x = torch.flatten(x, 1)  # Shape: [batch, d_model]
        x = self.fc(x)  # Shape: [batch, d_model]

        return x


class ModularActivation(nn.Module):
    def __init__(self, class_neurons_num, multi_label=False, criterion=None):
        super().__init__()
        self.multi_label = multi_label
        self.use_logits = isinstance(criterion, nn.BCEWithLogitsLoss)

        if class_neurons_num == 1 or multi_label:
            self.activation = nn.Identity() if self.use_logits else nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)  # Softmax for multi-class

    def forward(self, x):
        return self.activation(x)


class MultivariateTSAD(nn.Module):
    """
    First, we apply a CNN architecture to fuse sensor data into a latent space:
    Given a multivariate time-series of shape (S, T), where T is the time-series length and
    S is the number of sensors, the MultivariateUnivariateFuser CNN learns hidden patterns
    and flattens the input into a first-order tensor. This tensor is then passed through a
    transformer architecture for further processing.

    Input Shape:
    For the input matrix [batch_size, channels, height, width], the model assumes an input shape
    of (S, T), where S is the number of sensors and T is the time-series length.
    Output Shape:
    After the CNN, the output shape is [batch_size, T, 1, 1].

    Next, the CNN output is fed into a transformer architecture:
    By adapting the CNN's output to the transformer architecture, we leverage the transformer's ability
    to capture long-term dependencies and support parallel computation. For this purpose, we use transformer
    variants like Performer, Linformer, and Longformer, which are optimized to address the quadratic
    complexity of the standard transformer's attention mechanism, particularly for long sequences.

    Finally, the transformer's output passes through a fully connected layer for classification:
    The model ends with a modular classification layer, allowing for both binary and multi-class outputs.
    This modularity ensures the model can be pre-trained on a different task or dataset and fine-tuned for
    UAV anomaly detection.
    """

    def __init__(
            self,
            conv_fuser: BaseConvFuser,
            transformer_variant="vanilla",
            d_model=256,
            nhead=8,
            num_layers=6,
            dropout=0.15,
            use_learnable_pe=True,
            aggregator="attention",
            class_neurons_num=1,  # Binary classification default for anomaly detection.
            multi_label=False,
            time_scaler=1,
            criterion=None
    ):
        super().__init__()

        self.multi_label = multi_label

        # Variables Fuse
        self.conv_fuser = conv_fuser
        self.T = conv_fuser.T  # Sequence length
        self.embedding_dim = d_model

        # Embedding layer for transformer input
        # self.embedding = nn.Linear(self.T, d_model)
        self.embedding = nn.Linear(1, d_model)

        scaled_T = int(time_scaler * self.T)

        # Positional encoding
        if use_learnable_pe:
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=scaled_T)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=scaled_T)

        # Transformer encoder
        self.transformer = get_transformer_variant(
            variant=transformer_variant,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=scaled_T,
            dropout=dropout
        )

        # Aggregator
        if aggregator == "attention":
            self.aggregator = AttentionAggregator(d_model)
        elif aggregator == "linear":
            self.aggregator = LinearAggregator(seq_len=scaled_T, d_model=d_model)
        elif aggregator == "conv":
            self.aggregator = ConvAggregator(d_model)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        # Fully connected layers
        self.fc1 = nn.Linear(d_model, 128)
        self.activation1 = nn.LeakyReLU(0.01)
        self.batch_norm = nn.BatchNorm1d(128)
        self.layer_norm = nn.LayerNorm(128)  # Alternative normalization
        self.fc2 = nn.Linear(128, class_neurons_num)  # Output logits

        # Modular activation function (Sigmoid or Softmax based on loss)
        self.activation2 = ModularActivation(class_neurons_num, multi_label=multi_label, criterion=criterion)

        # Apply weight initialization
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_fuser(x)  # Shape: [batch_size, T, 1, 1]
        x = x.squeeze(-1)  # Squeeze: [batch_size, T, 1]

        # Map to embedding space and add positional encoding
        x = self.embedding(x)  # Shape: [batch_size, T, d_model]
        x = self.pos_encoding(x)

        # Transpose for transformer: [batch_size, T, d_model] -> [T, batch_size, d_model]
        x = x.permute(1, 0, 2)

        # Pass through transformer
        x = self.transformer(x)   # Shape: [T, batch_size, d_model]
        if isinstance(self.transformer, LongformerModel):
            x = x.last_hidden_state  # Shape: [batch_size, T, d_model]
            x = x.permute(1, 0, 2)  # Match shape for aggregator: [T, batch_size, d_model]

        # Aggregate the sequence dimension using the specified aggregator
        x = self.aggregator(x)  # Shape: [batch_size, d_model]

        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.activation1(x)

        # Automatically use LayerNorm if batch size is small
        if x.shape[0] <= 16:  # Adjust this threshold as needed
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)

        x = self.fc2(x)  # Compute logits
        x = x.view(-1, 1) if x.shape[-1] == 1 else x  # Ensure logits shape is correct
        x = self.activation2(x)  # Apply activation dynamically
        return x

    def predict(self, inputs, device="cpu", threshold=0.5):  # Default threshold = 0.5 for binary too
        """
        Perform inference on a batch of inputs.

        Args:
            inputs: Input tensor.
            device: Device for inference ("cpu" or "cuda").
            threshold: Threshold for classification (used in both binary and multi-label).

        Returns:
            predictions: Predicted outputs.
        """
        self.eval()
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = self(inputs)

            if isinstance(self.fc[-1].activation, nn.Sigmoid):
                predictions = (outputs > threshold).float()  # Use threshold dynamically for binary and multi-label
            elif isinstance(self.fc[-1].activation, nn.Softmax):
                predictions = torch.argmax(outputs, dim=1)  # Multi-class classification
            else:
                raise ValueError("Unknown activation function in the final layer.")

        return predictions


def build_model(model_config: dict):

    input_shape = model_config['input_shape']
    time_scaler = model_config['time_scaler']
    fuser_name = model_config['fuser_name']
    transformer_variant = model_config['transformer_variant']
    use_learnable_pe = model_config['use_learnable_pe']
    aggregator = model_config['aggregator']
    class_neurons_num = model_config['class_neurons_num']
    d_model = model_config['d_model']
    nhead = model_config['nhead']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    multi_label = model_config['multi_label']
    criterion = model_config['criterion']

    if time_scaler is None:
        S, T = input_shape
        S_tag = math.log(S)
        traget_T = S_tag * T
        time_scaler = traget_T / T

    # Choose CNN fuser dynamically
    if fuser_name == "ConvFuser1":
        fuser = ConvFuser1(input_shape, time_scaler)
    elif fuser_name == "ConvFuser2":
        fuser = ResNet2D(input_shape, blocks=model_config['blocks'], time_scaler=time_scaler)
    elif fuser_name == "ConvFuser3":
        fuser = ConvFuser3(input_shape)
    else:
        raise ValueError(f"Unknown fuser: {fuser_name}")

    # Build and return the model
    return MultivariateTSAD(
        conv_fuser=fuser,
        transformer_variant=transformer_variant,
        use_learnable_pe=use_learnable_pe,
        aggregator=aggregator,
        class_neurons_num=class_neurons_num,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        multi_label=multi_label,
        time_scaler=time_scaler,
        criterion=criterion
    )


if __name__ == '__main__':

    config = {
        'normal_path': 'normal_path',
        'fault_path': 'fault_path',
        'multilabel_path': 'multilabel_path',
        'multiclass_path': 'multiclass_path',
        'checkpoint_epoch_path': 'checkpoint_path',
        'best_model_path': 'best_model_path',
        'training_res': 'training_res',
        'test_res': 'test_res',
        'multi_class': True, # binary class' is determined by the number of data classes. Multilabel class' is concluded.
        'fuser_name': 'ConvFuser2',
        'blocks': (3, 3, 3, 3, 3, 3, 3),  # The ResNet skip connection blocks
        'transformer_variant': 'vanilla',  # Choose transformer variant
        'use_learnable_pe': True,  # Use learnable positional encoding
        'aggregator': 'conv',  # Use aggregation
        'num_epochs': 50,
        'd_model': 365,
        'nhead': 5,  # # transformer heads
        'num_layers': 8,  # transformer layers
        'batch_size': 16,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'time_scaler': None,  # The portion of T for conv output time-series latent representative
        'prediction_threshold': 0.5,
        'split_rates': (0.2, 0.3),
        'experimental_dataset_name': 'experimental_dataset_name'
    }

    # Define input dimensions. config['input_shape'] is calculated based on concreate data set inside
    # the main function in main.py.
    S, T = 64, 640  # Sensors and sequence length
    input_tens = torch.rand(1, S, T)  # [batch_size, S, T]
    config['input_shape'] = input_tens[0].shape
    config['class_neurons_num'] = 1
    config['multi_label'] = False
    config['criterion'] = torch.nn.BCEWithLogitsLoss()

    # Build the model with specific configurations
    ad_model = build_model(
        model_config=config
    )

    # Test forward pass
    output = ad_model(input_tens)  # Forward pass with input tensor
    print("Output shape:", output.shape)  # Expected: [batch_size, 1]

    # Print the model summary
    summary(ad_model, input_size=(1, S, T))  # Exclude batch dimension

    obj = ad_model
    print(obj.__class__.__name__)

    exit()