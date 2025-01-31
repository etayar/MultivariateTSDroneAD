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
        self.T = input_shape[1]  # Time-series length
        self.S = input_shape[0]  # Number of sensors
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


class ConvFuser2(BaseConvFuser):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]

    # Some CNN architecture


class ConvFuser3(BaseConvFuser):

    def __init__(self, input_shape):
        super().__init__()
        self.T = input_shape[1]
        self.S = input_shape[0]

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
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


class ConvAggregator(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Aggregate down to 1

    def forward(self, x):
        """
        x: [T, batch_size, d_model]
        Returns: [batch_size, d_model]
        """
        x = x.permute(1, 2, 0)  # Reshape to [batch_size, d_model, T]
        x = self.conv(x)  # Shape: [batch_size, d_model, T]
        x = self.global_pool(x).squeeze(-1)  # Shape: [batch_size, d_model]
        return x


class ModularActivation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(x)



class MultivariateTSAD(nn.Module):
    """
    First, we apply a CNN architecture to fuse sensor data into a latent space:
    Given a multivariate time-series of shape (S, T), where 𝑇 is the time-series length and
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
            num_classes=1  # Binary classification default for anomaly detection.
    ):
        super().__init__()

        # Variables Fuse
        self.conv_fuser = conv_fuser
        self.T = conv_fuser.T  # Sequence length
        self.embedding_dim = d_model

        # Embedding layer for transformer input
        # self.embedding = nn.Linear(self.T, d_model)
        self.embedding = nn.Linear(1, d_model)

        # Positional encoding (choose between sinusoidal and learnable)
        if use_learnable_pe:
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=self.T)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=self.T)

        # Transformer encoder
        self.transformer = get_transformer_variant(
            variant=transformer_variant,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=self.T,
            dropout=dropout
        )

        # Aggregator
        if aggregator == "attention":
            self.aggregator = AttentionAggregator(d_model)
        elif aggregator == "linear":
            self.aggregator = LinearAggregator(seq_len=self.T, d_model=d_model)
        elif aggregator == "conv":
            self.aggregator = ConvAggregator(d_model)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer
            ModularActivation(num_classes)  # Modular activation
        )


    def forward(self, x):
        x = self.conv_fuser(x)  # Shape: [batch_size, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # Squeeze: [batch_size, T]
        x = x.unsqueeze(-1)  # Add singleton feature dimension: [batch_size, T, 1]

        # Map to embedding space and add positional encoding
        x = self.embedding(x)  # Shape: [batch_size, T, d_model]
        x = self.pos_encoding(x)

        # Transpose for transformer: [batch_size, T, d_model] -> [T, batch_size, d_model]
        x = x.permute(1, 0, 2)

        # Pass through transformer
        if isinstance(self.transformer, nn.TransformerEncoder):
            x = self.transformer(x)  # Shape: [T, batch_size, d_model]
        elif isinstance(self.transformer, LongformerModel):
            x = self.transformer(x).last_hidden_state  # Shape: [batch_size, T, d_model]
            x = x.permute(1, 0, 2)  # Match shape for aggregator: [T, batch_size, d_model]
        elif isinstance(self.transformer, (Performer, Linformer)):
            x = self.transformer(x)  # Shape: [T, batch_size, d_model]

        # Aggregate the sequence dimension using the specified aggregator
        x = self.aggregator(x)  # Shape: [batch_size, d_model]

        # Pass through fully connected layer
        x = self.fc(x)  # Shape: [batch_size, 1]
        return x

    def predict(self, inputs, device="cpu"):
        """
        Perform inference on a batch of inputs.

        Args:
            inputs: Input tensor.
            device: Device for inference ("cpu" or "cuda").

        Returns:
            predictions: Predicted outputs.
        """
        self.eval()
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = self(inputs)

            # Determine prediction logic based on the number of classes
            if self.fc[-1].activation.__class__.__name__ == "Sigmoid":  # Binary classification
                predictions = (outputs > 0.5).float()
            elif self.fc[-1].activation.__class__.__name__ == "Softmax":  # Multi-class classification
                predictions = torch.argmax(outputs, dim=1)
            else:
                raise ValueError("Unknown activation function in the final layer.")

        return predictions


def build_model(
        model_config: dict
):
    input_shape = model_config['input_shape']
    time_scaler = model_config['time_scaler']
    fuser_name = model_config['fuser_name']
    transformer_variant = model_config['transformer_variant']
    use_learnable_pe = model_config['use_learnable_pe']
    aggregator = model_config['aggregator']
    num_classes = model_config['num_classes']
    d_model = model_config['d_model']
    nhead = model_config['nhead']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']

    # Choose CNN fuser dynamically
    if fuser_name == "ConvFuser1":
        fuser = ConvFuser1(input_shape, time_scaler)
    elif fuser_name == "ConvFuser2":
        fuser = ConvFuser2(input_shape)
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
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )


if __name__ == '__main__':
    # Define input dimensions
    S, T = 64, 640  # Sensors and sequence length
    input_tens = torch.rand(1, S, T)  # [batch_size, S, T]

    config = {
        'input_shape': input_tens[0].shape,
        'time_scaler': 0.7966,  # time_scaler may be smaller than 1 for computational improvement or bigger than 1 for higher representation of sensors temporal patterns.
        'fuser_name': 'ConvFuser1',
        'transformer_variant': 'vanilla',  # Choose transformer variant
        'use_learnable_pe': True,  # Use learnable positional encoding
        'aggregator': 'attention',  # Use attention-based aggregation
        'num_epochs': 50,  # This is used during training; it is included here to demonstrate the configuration structure
        'num_classes': 12,
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dropout': 0.15
    }

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