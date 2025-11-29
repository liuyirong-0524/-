from abc import abstractmethod

import torch
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

import copy
import torch.nn.functional as F

import ot  # pip install POT

class OTAlignModule(nn.Module):
  """
    Functionality:
    Performs Optimal Transport (OT) based alignment between single-cell (SC) gene 
    expression data and spatial gene expression features (h_spot).
    It computes an OT map to project SC features onto the spatial grid, calculates 
    a regularization loss, and fuses the projected features with the original 
    spatial features.
    """
    def __init__(self, in_channels: int, reg: float = 10.0, alpha: float = 1e-3, use_conv: bool = False):
      """
        Inputs:
        :param in_channels: The number of feature channels (C), representing the number of genes.
        :type in_channels: int
        :param reg: The regularization parameter for the Sinkhorn algorithm. (Sinkhorn Regularization)
        :type reg: float
        :param alpha: The weight for fusing the OT-aligned feature map with h_spot. (OT Feature Fusion Weight)
        :type alpha: float
        :param use_conv: If True, uses a 1x1 convolution on the OT feature map before fusion.
        :type use_conv: bool
        """
        super().__init__()
        self.in_channels = in_channels  # C (gene_num)
        self.reg = reg                   # Sinkhorn regularization
        self.alpha = alpha               # OT feature fusion weight
        self.use_conv = use_conv
        self.use_conv = use_conv

        if self.use_conv:
            # 1x1 conv to maintain channels, facilitating direct addition
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, sc: torch.Tensor, h_spot: torch.Tensor):
      """
        Functionality:
        Computes the optimal transport map between single-cell features (source) 
        and spatial features (target) for each batch element. It then projects the 
        single-cell features onto the spatial dimension and fuses them with h_spot.

        Inputs:
        :param sc: Single-cell gene expression data. Expected shape: [n_cells, C_sc].
        :type sc: torch.Tensor
        :param h_spot: Spatial gene expression features. Expected shape: [B, C, H, W].
        :type h_spot: torch.Tensor

        Outputs:
        :return h_spot_aligned: The spatial features aligned with the SC features, after fusion. Shape: [B, C, H, W].
        :rtype h_spot_aligned: torch.Tensor
        :return L_ot: The mean MSE loss calculated between the original spatial features and the projected SC features across the batch.
        :rtype L_ot: torch.Tensor
        :return ot_feat: The projected single-cell feature map (before fusion). Shape: [B, C, H, W].
        :rtype ot_feat: torch.Tensor
        """

        device = h_spot.device
        dtype = h_spot.dtype

        B, C, H, W = h_spot.shape
        assert C == self.in_channels, f"OTAlignModule in_channels={self.in_channels}, but h_spot has C={C}"

        assert sc.ndim == 2
        n_cells, C_sc = sc.shape

        sc_sub = sc[:, :C]  # [n_cells, C]
        sc_sub = sc_sub.to(device=device, dtype=dtype)

        Y_spatial = h_spot.view(B, C, -1).permute(0, 2, 1)  # [B, n_spots, C]

        ot_losses = []
        ot_maps = []

        X_sc_np = sc_sub.detach().cpu().numpy()  # [n_cells, C]

        for b in range(B):
            Y_b = Y_spatial[b]  # [n_spots, C]
            Y_b_np = Y_b.detach().cpu().numpy()

            cost_matrix = ot.dist(X_sc_np, Y_b_np, metric="euclidean") ** 2

            a = np.ones((n_cells,), dtype=np.float64) / n_cells        
            b_vec = np.ones((Y_b_np.shape[0],), dtype=np.float64) / Y_b_np.shape[0]  

            M = ot.sinkhorn(a, b_vec, cost_matrix, self.reg)  # [n_cells, n_spots]
            M_T = M.T  # [n_spots, n_cells]

            X_space_np = M_T @ X_sc_np
            X_space = torch.from_numpy(X_space_np).to(device=device, dtype=dtype)  # [n_spots, C]

            L_b = F.mse_loss(Y_b, X_space)
            ot_losses.append(L_b)

            X_space_map = X_space.permute(1, 0).contiguous().view(C, H, W)  # [C, H, W]
            ot_maps.append(X_space_map)

        L_ot = torch.stack(ot_losses).mean() if len(ot_losses) > 0 else torch.zeros([], device=device, dtype=dtype)
        ot_feat = torch.stack(ot_maps, dim=0)  # [B, C, H, W]

        if self.use_conv:
            ot_feat = self.conv(ot_feat)  # [B, C, H, W]

        h_spot_aligned = h_spot + self.alpha * ot_feat

        return h_spot_aligned, L_ot, ot_feat
    
class LightweightModuleCoordinator(nn.Module):
  """
    Functionality:
    Coordinates and fuses information from multiple feature maps (experts) using 
    a token-based negotiation mechanism. It extracts tokens from input features, 
    updates them via a negotiator, and uses the updated tokens to modulate 
    the original features with residual gates. Optionally fuses features into a 
    latent representation (z).
    """
    def __init__(self, in_channels_list, d_token: int = 128, num_heads: int = 4, token_mode: str = "gap", latent_dim: int | None = None):
      """
        Inputs:
        :param in_channels_list: A list of input channel sizes [C1, C2, C3, C4] for the 4 feature modules.
        :type in_channels_list: list[int]
        :param d_token: The dimension of the feature tokens.
        :type d_token: int
        :param num_heads: The number of attention heads in the TokenNegotiator.
        :type num_heads: int
        :param token_mode: The method used by ExpertTokenizer (e.g., "gap" for Global Average Pooling).
        :type token_mode: str
        :param latent_dim: The channel dimension of the output latent feature (z). If None, no latent fusion is performed.
        :type latent_dim: Optional[int]
        """
        super().__init__()
        assert len(in_channels_list) == 4
        self.num_modules = 4
        self.tokenizers = nn.ModuleList([ExpertTokenizer(C, d_token, mode=token_mode) for C in in_channels_list])
        self.negotiator = TokenNegotiator(d_token=d_token, num_heads=num_heads, num_modules=self.num_modules)
        self.modulators = nn.ModuleList([ChannelModulator(d_token, C) for C in in_channels_list])

        self.residual_gates = nn.ParameterList([nn.Parameter(torch.full((1,), -2.0)) for _ in range(self.num_modules)])

        self.latent_dim = latent_dim
        if latent_dim is not None:
            self.to_latent = nn.ModuleList([nn.Conv2d(C, latent_dim, kernel_size=1) for C in in_channels_list])
            self.latent_gate = nn.Parameter(torch.tensor(-2.0))  

    def forward(self, feats: list[torch.Tensor], z: torch.Tensor | None = None):
      """
        Functionality:
        Passes input features through tokenizers, updates the tokens via negotiation, 
        uses the updated tokens to modulate the input features, and fuses them into 
        a latent feature (z) if provided.

        Inputs:
        :param feats: A list of 4 input feature tensors [F1, F2, F3, F4], where Fi has shape [B, Ci, Hi, Wi].
        :type feats: list[torch.Tensor]
        :param z: An optional input latent feature tensor to be updated. Shape: [B, C_lat, Hz, Wz].
        :type z: Optional[torch.Tensor]

        Outputs:
        :return out_feats: A list of 4 output feature tensors, modulated and updated via residual connection.
        :rtype out_feats: list[torch.Tensor]
        :return out_z: The updated latent feature tensor, or None if z was not provided.
        :rtype out_z: Optional[torch.Tensor]
        """
        # feats: [F1,F2,F3,F4], Fi: [B, Ci, Hi, Wi]
        tokens = [tok(fea) for tok, fea in zip(self.tokenizers, feats)]   # 4 * [B,d]
        tokens = torch.stack(tokens, dim=1)                               # [B,4,d]
        updated = self.negotiator(tokens)                                 # [B,4,d]

        out_feats = []
        for i in range(self.num_modules):
            token_i = updated[:, i, :]            # [B,d]
            feat_i  = feats[i]                    # [B,Ci,Hi,Wi]
            gamma_i, beta_i = self.modulators[i](token_i, feat_i)
            mod_i = gamma_i * feat_i + beta_i
            gate_i = torch.sigmoid(self.residual_gates[i]) * 0.5
            out_i = feat_i + gate_i * mod_i
            out_feats.append(out_i)

        out_z = None
        if (self.latent_dim is not None) and (z is not None):
            B, C_lat, Hz, Wz = z.shape
            fused = 0
            for i in range(self.num_modules):
                ali = self.to_latent[i](out_feats[i])          # [B,latent_dim,Hi,Wi]
                if ali.shape[-2:] != (Hz, Wz):
                    ali = F.interpolate(ali, size=(Hz, Wz), mode="bilinear", align_corners=False)
                fused = fused + ali
            lgate = torch.sigmoid(self.latent_gate) * 0.5
            out_z = z + lgate * fused
        return out_feats, out_z
    def rebuild_weights(self, in_channels_list, latent_dim):
      """
        Functionality:
        Resets and reinitializes the weights of internal modules (specifically nn.Linear 
        layers) while keeping the original Parameter registration intact. This method 
        is typically used when channel dimensions change dynamically.

        Inputs:
        :param in_channels_list: The updated list of input channel sizes.
        :type in_channels_list: list[int]
        :param latent_dim: The updated latent feature dimension.
        :type latent_dim: int
        """
        self.in_channels_list = in_channels_list
        self.latent_dim = latent_dim
        # Recreate internal linear layers, but keep the original Parameter registry intact
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

      
class AttentionPool2d(nn.Module):
  """
    Functionality:
    Performs attention-based global feature pooling on 2D spatial data. 
    It incorporates positional embeddings, computes QKV attention across the 
    flattened spatial dimension, and returns the pooled feature vector 
    corresponding to the class token (the mean vector). This is common in 
    Vision Transformers (ViT) and CLIP architectures.
    """
    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
      """
        Inputs:
        :param spacial_dim: The height/width of the input feature map (H or W), assuming H=W.
        :type spacial_dim: int
        :param embed_dim: The channel dimension (C) of the input features.
        :type embed_dim: int
        :param num_heads_channels: The number of channels per attention head (C_head). Used to calculate num_heads.
        :type num_heads_channels: int
        :param output_dim: The final output dimension after projection. Defaults to embed_dim.
        :type output_dim: Optional[int]
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
      """
        Functionality:
        Flattens the spatial input, prepends the mean vector as a 'class token', 
        adds positional embeddings, applies attention, and returns the result 
        of the projected class token.

        Inputs:
        :param x: The input feature map. Shape: [B, C, H, W], where H=W=spacial_dim.
        :type x: torch.Tensor

        Outputs:
        :return: The global pooled feature vector (class token representation). Shape: [B, output_dim].
        :rtype: torch.Tensor
        """
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
  """
    Functionality:
    An abstract base class for neural network blocks (layers) that must explicitly 
    consume and utilize time step embeddings (emb), typically used in Diffusion Models.
    """


    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
  """
    Functionality:
    A sequential container module (like nn.Sequential) that correctly handles 
    time step embeddings (`emb`). It passes `emb` only to layers that inherit 
    from `TimestepBlock`, and skips `emb` for standard modules.
    """

    def forward(self, x, emb):
      """
        Functionality:
        Passes input `x` sequentially through all layers. If a layer is a 
        `TimestepBlock`, both `x` and `emb` are passed; otherwise, only `x` is passed.

        Inputs:
        :param x: The input data tensor.
        :type x: torch.Tensor
        :param emb: The time step embedding tensor.
        :type emb: torch.Tensor

        Outputs:
        :return: The final output data tensor after sequential processing.
        :rtype: torch.Tensor
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SE_Attention(nn.Module):
  """
    Functionality:
    Implements the Squeeze-and-Excitation (SE) block, a channel attention mechanism.
    It adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels. 
    """
    def __init__(self, channel=512, reduction=16):
      """
        Inputs:
        :param channel: The number of input and output feature channels (C).
        :type channel: int
        :param reduction: The reduction ratio used to determine the size of the 
                          bottleneck in the fully connected layers.
        :type reduction: int

        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def init_weights(self):
      """
        Functionality:
        Custom initialization method for the weights of the SE-Attention block, 
        using Kaiming initialization for Conv2d, constant for BatchNorm2d, and 
        Normal initialization for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
      """
        Functionality:
        Performs the squeeze operation (global average pooling), the excitation 
        operation (two linear layers), and finally scales the input feature map 
        channel-wise.

        Inputs:
        :param x: The input feature map. Shape: [B, C, H, W].
        :type x: torch.Tensor

        Outputs:
        :return: The channel-wise scaled output feature map. Shape: [B, C, H, W].
        :rtype: torch.Tensor
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Upsample(nn.Module):
  """
    Functionality:
    Performs spatial upsampling of the input tensor by a factor of 2. 
    It uses nearest neighbor interpolation and optionally applies a 
    convolutional layer afterward to refine the features.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
      """
        Inputs:
        :param channels: The number of input feature channels.
        :type channels: int
        :param use_conv: If True, apply a 3x3 convolution after upsampling.
        :type use_conv: bool
        :param dims: The number of spatial dimensions (2 for image, 3 for volume/video).
        :type dims: int
        :param out_channels: The number of output channels. Defaults to input channels.
        :type out_channels: Optional[int]
      """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
      """
        Functionality:
        Upsamples the input tensor using nearest neighbor interpolation and optionally 
        processes it with a convolution.
        """
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
  """
    Functionality:
    Performs spatial downsampling of the input tensor by a factor of 2. 
    It uses either a stride-2 convolution or a stride-2 average pooling operation.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
      """
        Functionality:
        Applies the downsampling operation (convolution or pooling) to the input tensor.

        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
  """
    Functionality:
    Implements a residual block (ResBlock), a fundamental building block in UNets and 
    Diffusion Models. It applies two convolution layers and adds the input (skip 
    connection) to the output. It integrates time step embeddings (`emb`) to condition 
    the feature processing. It can optionally perform upsampling or downsampling 
    within the block. 
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
      """
        Functionality:
        Wrapper around `_forward` to handle gradient checkpointing if enabled.
        """

        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
      """
        Functionality:
        Executes the main residual block logic, including spatial update, embedding 
        integration (via addition or scale-shift-norm), and the final residual connection.

        Outputs:
        :return: The output tensor: `skip_connection(x) + h`.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
  """
    Functionality:
    Implements a self-attention block, typically used in UNet architectures (like 
    those in Diffusion Models) to allow the model to capture global dependencies 
    across the spatial dimensions. It uses a residual connection around the attention 
    mechanism.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
      """
        Functionality:
        Wrapper around `_forward` that applies gradient checkpointing for memory optimization.
        """
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
      """
        Functionality:
        Executes the self-attention process: normalization, QKV projection, attention calculation, 
        output projection, and residual addition.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
  """
    Functionality:
    Calculates the approximate floating-point operations (FLOPs) for the attention mechanism 
    (QKVAttention and QKVAttentionLegacy). It assumes the dominant cost comes from the two 
    matrix multiplications (Q*K and Weight*V).

    Inputs:
    :param model: The attention module instance (used for accessing total_ops counter).
    :type model: nn.Module
    :param _x: Ignored input.
    :type _x: Tuple
    :param y: The output of the attention module (used for shape information).
    :type y: Tuple[torch.Tensor]

    Outputs:
    None (Updates model.total_ops counter in-place).
    """

    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
  """
    Functionality:
    Implements multi-head self-attention using a "legacy" ordering: first splits 
    the QKV tensor into heads * 3 * channels and then splits into Q, K, V. 
    It computes attention weights and applies them to V.
    """

    def __init__(self, n_heads):
      """
        Inputs:
        :param n_heads: The number of attention heads.
        :type n_heads: int
        """
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
      """
        Functionality:
        Calculates attention weights via QK interaction, applies softmax, and computes 
        the weighted sum with V.

        Inputs:
        :param qkv: The concatenated Query, Key, and Value tensor. Shape: [B, 3*C, N], 
                    where C is the total channel size and N is the sequence length (H*W).
        :type qkv: torch.Tensor

        Outputs:
        :return: The output attention tensor. Shape: [B, C, N].
        :rtype: torch.Tensor
        """

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
      """
        Functionality:
        Static method to count FLOPs by calling the helper function.
        """
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
  """
    Functionality:
    Implements multi-head self-attention using a "new/standard" ordering: 
    first splits the QKV tensor into Q, K, V chunks, and then reshapes them 
    for multi-head processing.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
      """
        Functionality:
        Calculates attention weights via QK interaction, applies softmax, and computes 
        the weighted sum with V.

        Inputs:
        :param qkv: The concatenated Query, Key, and Value tensor. Shape: [B, 3*C, N].
        :type qkv: torch.Tensor

        Outputs:
        :return: The output attention tensor. Shape: [B, C, N].
        :rtype: torch.Tensor
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
      """
        Functionality:
        Static method to count FLOPs by calling the helper function.
        """
        return count_flops_attn(model, _x, y)

class GMSD_loss(nn.Module):
  """
    Functionality:
    Implements a modified Gradient Magnitude Similarity Deviation (GMSD) image 
    quality metric. It computes GMSD between a distorted image and a reference image, 
    enhancing the feature maps by mapping them to a hypersphere and adding noise 
    before gradient calculation. 
    """
    def __init__(self, c=170, device='cuda', noise_std=0.01,num_channels=128):
      """
        Inputs:
        :param c: A small constant used for numerical stability in the GMSD calculation.
        :type c: int
        :param device: The device where the tensors are located ('cuda' or 'cpu').
        :type device: str
        :param noise_std: The standard deviation of the small noise added during hypersphere mapping.
        :type noise_std: float
        :param num_channels: The number of channels (C) in the input feature maps/images.
        :type num_channels: int

        """
        super(GMSD_loss, self).__init__()
        self.c = c
        self.device = device
        self.noise_std = noise_std
        self.num_channels = num_channels

        self.hx = torch.tensor([[1/3, 0, -1/3]]*3, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.hx = self.hx.expand(num_channels, 1, 3, 3).to(self.device)
        self.hy = self.hx.transpose(2, 3).to(self.device)

        self.ave_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
        self.ave_filter = self.ave_filter.expand(num_channels, 1, 2, 2).to(self.device)

    def forward(self, dis_img, ref_img):
        """
        Functionality:
        Computes the GMSD loss between the distorted and reference images after 
        preprocessing and applying hypersphere enhancement.

        Inputs:
        :param dis_img: Distorted image tensor. Shape: [B, C, H, W].
        :type dis_img: torch.Tensor
        :param ref_img: Reference image tensor. Shape: [B, C, H, W].
        :type ref_img: torch.Tensor

        Outputs:
        :return: The GMSD score, calculated as the standard deviation of the GMS map.
        :rtype: torch.Tensor (scalar)
        """
        if torch.max(dis_img) <= 1:
            dis_img = dis_img * 255
        if torch.max(ref_img) <= 1:
            ref_img = ref_img * 255

        dis_img = dis_img.float()
        ref_img = ref_img.float()

        # Mean filtering
        ave_dis = F.conv2d(dis_img, self.ave_filter, stride=1, groups=self.num_channels)
        ave_ref = F.conv2d(ref_img, self.ave_filter, stride=1, groups=self.num_channels)

        # Downsampling
        down_step = 2
        ave_dis_down = ave_dis[:, :, 0::down_step, 0::down_step]
        ave_ref_down = ave_ref[:, :, 0::down_step, 0::down_step]

        # Hypersphere mapping (enhancement)
        ave_dis_down = self._map_to_hypersphere(ave_dis_down)
        ave_ref_down = self._map_to_hypersphere(ave_ref_down)

        # Gradient magnitude calculation
        mr_sq = F.conv2d(ave_ref_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_ref_down, self.hy, groups=self.num_channels)**2
        md_sq = F.conv2d(ave_dis_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_dis_down, self.hy, groups=self.num_channels)**2
        mr = torch.sqrt(mr_sq)
        md = torch.sqrt(md_sq)

        # GMS (Gradient Magnitude Similarity) calculation
        GMS = (2 * mr * md + self.c) / (mr_sq + md_sq + self.c)
        GMSD = torch.std(GMS.view(-1))
        return GMSD

    def _map_to_hypersphere(self, x):
        """
        Functionality:
        Normalizes the input feature map onto a unit hypersphere (L2-normalization) 
        and adds small noise, followed by re-normalization.

        Inputs:
        :param x: Input feature tensor. Shape: [B, C, H, W].
        :type x: torch.Tensor

        Outputs:
        :return: The normalized and noise-enhanced feature tensor on the hypersphere.
        :rtype: torch.Tensor
        """
        x_normalized = x / x.norm(dim=1, keepdim=True)  # [B, C, H, W]

        noise = torch.randn_like(x_normalized) * self.noise_std
        x_noisy = x_normalized + noise

        x_noisy_normalized = x_noisy / x_noisy.norm(dim=1, keepdim=True)
        return x_noisy_normalized
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
  """
    Functionality:
    Implements the spatial attention block from the CBAM (Convolutional Block 
    Attention Module) architecture. It generates a 2D spatial attention map 
    by aggregating channel features using average and max pooling, applying 
    a convolution, and then multiplying the result with the input features. 
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 7x7 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      """
        Functionality:
        Computes the spatial attention map from the input feature map and applies 
        the weights via element-wise multiplication.

        Inputs:
        :param x: Input feature tensor. Shape: [B, C, H, W].
        :type x: torch.Tensor

        Outputs:
        :return: The input feature map scaled by the spatial attention weights.
        :rtype: torch.Tensor (Shape: [B, C, H, W])
        """
        # Compute spatial attention weights
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, H, W]
        attn = self.conv1(attn)  # [batch, 1, H, W]
        return self.sigmoid(attn) * x  # Apply attention weights to input


class GeneRelationWeighting(nn.Module):
    """
    Module to adjust ST gene features based on a gene relationship matrix.
    """
    def __init__(self, num_genes, gene_relation_matrix=None):
      """
        Inputs:
        :param num_genes: The total number of genes (G) in the feature set.
        :type num_genes: int
        :param gene_relation_matrix: An optional initial matrix representing gene-gene relationships.
        :type gene_relation_matrix: Optional[np.ndarray]
        """
        super(GeneRelationWeighting, self).__init__()
        gene_relation_matrix = torch.tensor(gene_relation_matrix, dtype=torch.float32)

        self.gene_relation_matrix = nn.Parameter(gene_relation_matrix)

    def forward(self, gene_features):
        """
        Functionality:
        Applies the gene relationship matrix to the gene dimension of the spatial 
        features, effectively mixing information across genes for each spatial location.
        Inputs:
        :param gene_features: Input feature tensor containing gene expression at spatial locations. 
                              Shape: [B, num_genes, H, W].
        :type gene_features: torch.Tensor
        
        Outputs:
        :return: The gene features weighted/transformed by the relationship matrix.
        :rtype: torch.Tensor (Shape: [B, num_genes, H, W])
        """
        # Reshape gene features for matrix multiplication
        batch, num_genes, H, W = gene_features.shape
        gene_features = gene_features.view(batch, num_genes, -1)  # [batch, num_genes, H*W]
        # Apply relationship matrix
        weighted_features = torch.matmul(self.gene_relation_matrix, gene_features)  # [batch, num_genes, H*W]
        return weighted_features.view(batch, num_genes, H, W)

class GeneNameFeatureProcessor(nn.Module):
    """
    Functionality:
    Processes gene name embeddings (e.g., from a language model like BERT) 
    and projects them into a 2D spatial feature map. It uses a fully connected layer 
    and sequential upsampling/convolution to transform the 1D gene features into 
    a high-resolution feature map.
    """
    def __init__(self, out_channels=20, spatial_size=64):
        """
        Functionality:
        Initializes the projection and upsampling layers.

        Inputs:
        :param out_channels: The number of output channels, should match the second dimension of the input (20).
        :type out_channels: int
        :param spatial_size: The target height/width of the output feature map (e.g., 64).
        :type spatial_size: int

        Internal Transformations:
        - Input: tensor shape [batch, 20, 768]
        - Output: tensor shape [batch, 20, spatial_size, spatial_size]
        """
        super(GeneNameFeatureProcessor, self).__init__()
        self.out_channels = out_channels  # Should match the second dimension of the input (20)
        self.spatial_size = spatial_size
        # Maps each gene's 768-dim vector to a smaller 256-dim vector for initial spatial arrangement
        self.fc = nn.Linear(768, 256) 
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d1 = nn.Conv2d(20, 20, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d2 = nn.Conv2d(20, 20, 3, padding=1)
        
    def forward(self, gene_name_features):
        """
        Functionality:
        Passes the gene features through the FC layer, reshapes them into a small 
        spatial map, and then upsamples/refines them to the target spatial size.

        Inputs:
        :param gene_name_features: Input tensor containing gene name embeddings. Shape: [batch, 20, 768].
        :type gene_name_features: torch.Tensor

        Outputs:
        :return: The processed spatial feature map. Shape: [batch, 20, spatial_size, spatial_size].
        :rtype: torch.Tensor
        """
        # gene_name_features: [batch, 20, 768]
        gene_name_features = gene_name_features.to(next(self.parameters()).device)
        x = self.fc(gene_name_features)  # shape: [batch, 20, 256]
        
        # Reshape to initial spatial map (e.g., 16x16, since 256 = 16*16)
        x = x.view(-1, self.out_channels, self.spatial_size//4, self.spatial_size//4)  # shape: [batch, 20, 16, 16]
        
        # Upsample 1 (16x16 -> 32x32)
        x = self.upsample1(x)  
        x = self.conv2d1(x)  
        
        # Upsample 2 (32x32 -> 64x64)
        x = self.upsample2(x)
        x = self.conv2d2(x)
        
        return x

class MetadataFeatureProcessor(nn.Module):
    """
    Functionality:
    Processes global metadata features (e.g., sample-level features, also 768-dim) 
    and expands them into a 2D spatial feature map that can be broadcast across 
    spatial locations. This effectively creates a spatial bias or condition for the model.
    """
    def __init__(self, out_channels=20, spatial_size=64):
        """
        Functionality:
        Initializes the projection and upsampling layers for metadata processing.

        Inputs:
        :param out_channels: The number of output channels (e.g., 20).
        :type out_channels: int
        :param spatial_size: The target height/width of the output feature map (e.g., 64).
        :type spatial_size: int

        Outputs:
        None
        """
        super(MetadataFeatureProcessor, self).__init__()
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        # FC layer for initial feature processing
        self.fc = nn.Linear(768, 768)
        
        # Upsampling and refinement layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Note: The channel mapping is hardcoded to 3 -> 20 in conv2d1 based on view layer's output
        self.conv2d1 = nn.Conv2d(3, 20, 3, padding=1) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d2 = nn.Conv2d(20, 20, 3, padding=1)
        
    def forward(self, metadata_feature, batch_size):
        """
        Functionality:
        Processes the metadata vector, reshapes it into a small spatial map (3 channels), 
        and then upsamples/refines it to the target spatial size and channel depth.

        Inputs:
        :param metadata_feature: Input tensor containing metadata features. Shape: [batch, 768] or [768].
        :type metadata_feature: torch.Tensor
        :param batch_size: The current batch size (used when input is a single vector).
        :type batch_size: int

        Outputs:
        :return: The processed spatial feature map derived from metadata. Shape: [batch, 20, spatial_size, spatial_size].
        :rtype: torch.Tensor
        """
        # Add a dimension if it's a 1D vector
        if metadata_feature.dim() == 1:
            metadata_feature = metadata_feature.unsqueeze(0)  # shape: [1, 768]
            
        # Ensure input data is on the correct device
        device = next(self.parameters()).device
        metadata_feature = metadata_feature.to(device)
        
        # Map to [batch, 768] (assuming input/output dimensions are same here)
        x = self.fc(metadata_feature)  
        
        # Reshape to initial spatial map. Assumes 768 = 3 * 16 * 16 (for spatial_size=64)
        x = x.view(-1, 3, self.spatial_size//4, self.spatial_size//4)  # shape: [batch, 3, 16, 16]
        
        # Upsample 1 (16x16 -> 32x32)
        x = self.upsample1(x)  
        x = self.conv2d1(x)  # Channel mapping (3 -> 20)
        
        # Upsample 2 (32x32 -> 64x64)
        x = self.upsample2(x) 
        x = self.conv2d2(x)
        
        return x

class ATransformer(nn.Module):
    """
    Functionality:
    A simplified Vision Transformer (ViT) block used to process and encode 
    2D feature maps. It flattens the HxW spatial dimensions into a sequence of 
    tokens (HW), processes them using a standard Transformer Encoder, and 
    then restores the spatial dimensions.
    """
    def __init__(self, in_ch, out_ch, d=128, depth=2, nheads=4, mlp_ratio=4.0):
        """
        Inputs:
        :param in_ch: Number of input channels (Cin).
        :type in_ch: int
        :param out_ch: Number of output channels (Cout).
        :type out_ch: int
        :param d: The model dimension (d_model) used in the Transformer layers.
        :type d: int
        :param depth: Number of stacked Transformer Encoder layers.
        :type depth: int
        :param nheads: Number of attention heads.
        :type nheads: int
        :param mlp_ratio: Ratio to determine the hidden dimension of the feedforward network (d * mlp_ratio).
        :type mlp_ratio: float

        Outputs:
        None
        """
        super().__init__()
    
        self.in_proj  = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=int(d*mlp_ratio),
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=depth)
        
        self.out_proj = nn.Conv2d(d, out_ch, kernel_size=1, bias=False)

    def forward(self, x):                        # x: [B, Cin, H, W]
        """
        Functionality:
        Processes the input image features through the Transformer encoder.

        Inputs:
        :param x: Input feature map. Shape: [B, Cin, H, W].
        :type x: torch.Tensor

        Outputs:
        :return: Output feature map after Transformer encoding. Shape: [B, Cout, H, W].
        :rtype: torch.Tensor
        """
        B, _, H, W = x.shape
        x = self.in_proj(x)                      # [B, d, H, W]
        tokens = x.flatten(2).transpose(1, 2)    # [B, HW, d]
        tokens = self.encoder(tokens)            # [B, HW, d]
        x = tokens.transpose(1, 2).reshape(B, -1, H, W) # [B, d, H, W]
        x = self.out_proj(x)                     # [B, Cout, H, W]
        return x   
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class UNetModel(nn.Module):
    """
    Functionality:
    A highly specialized, multi-modal UNet-based architecture, likely used 
    in Diffusion Models for Spatial Transcriptomics or multi-modal image generation. 
    It integrates features from gene expression, low-resolution spatial data, 
    Whole Slide Imaging (WSI) at multiple resolutions, single-cell data, and 
    contextual embeddings via various attention, memory, and coordination modules.
    """

    def __init__(
            self,
            gene_num,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False, # Whether to use ResBlocks for up/downsampling.
            use_new_attention_order=False, # Whether to use the new attention order.
            root='',
            args=None,
    ):
        """
        Inputs:
        :param gene_num: Number of genes, used for defining input/output channels.
        :param model_channels: Base number of channels in the model.
        :param num_res_blocks: Number of ResBlocks per down/up sampling level.
        :param attention_resolutions: Resolutions (downsampling rates) where attention mechanisms are applied.
        :param channel_mult: Channel multiplier tuple, controlling channel growth at different levels.
        :param conv_resample: Whether to use convolution for up/downsampling.
        :param resblock_updown: Whether to use residual blocks for upsampling and downsampling.
        :param use_new_attention_order: Whether to use the new attention order mode.
        """
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.gene_num = gene_num # Gene count, used for defining input and output channel counts.
        self.model_channels = model_channels
        # self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions # Specifies at which downsampling rates attention is used.
        self.dropout = dropout
        self.channel_mult = channel_mult # Channel multiplier, controlling channel changes at different layers.
        self.conv_resample = conv_resample # Whether to use convolution for upsampling and downsampling.
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.Constrained_Refinement = Constrained_Refinement(gene_num=gene_num).cuda()

        time_embed_dim = model_channels * 4 # The time embedding layer is used to process time step information, generating a time embedding vector that is subsequently injected into multiple locations in the network.
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.pre = nn.Sequential(
            conv_nd(dims, model_channels, self.gene_num, 3, padding=1),
            nn.SiLU()
        )
        self.post = nn.Sequential(
            conv_nd(dims, self.gene_num, model_channels, 3, padding=1),
            nn.SiLU()
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.gene_num, ch, 3, padding=1))]
        )

        self.input_blocks_WSI5120 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        self.input_blocks_WSI320 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )


        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,time_embed_dim,dropout,out_channels=int(mult * model_channels),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_WSI5120.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                self.input_blocks_WSI5120.append(
                    TimestepEmbedSequential(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,time_embed_dim,dropout,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,time_embed_dim,dropout,out_channels=int(model_channels * mult),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,up=True,)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = self.channel_mult[-1] * self.model_channels

        self.input_blocks_lr = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 2 * conv_ch, conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.fc_modulation_1 = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.fc_modulation_2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.gene_num*2, 3, padding=1)),
        )
        self.pre_he_block = PreHEBlock().cuda()
        self.to_q = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k = nn.Linear(model_channels, model_channels, bias=False)
        self.to_v = nn.Linear(model_channels, model_channels, bias=False)

        self.to_q_con = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)
        self.to_v_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)
        self.replacer = FeatureNoiseReplacer(replacement_prob=0.8)  
        self.ot_align = OTAlignModule(
            in_channels=self.gene_num,
            reg=10.0,      
            alpha=1e-3,     
            use_conv=False  
        )
        # co_expression = np.load('./gene_coexpre.npy')

        self.MemoryNetwork = GeneMemoryNetwork(gene_dim=64, num_genes=self.gene_num,query_dim=self.gene_num)
        #self.optimized_gmsd = OptimizedGMSD(GMSD_loss(), self.gene_num, co_expression_new)
        self.GMSD_loss = GMSD_loss(num_channels=self.model_channels)
        self.conv_layer1 = nn.Sequential(
            conv_nd(dims, 148+self.gene_num, ch, 3, padding=1),
            nn.SiLU()
        )

        #self.conv_map = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Sequential(
            conv_nd(dims, input_ch+self.gene_num, input_ch, 3, padding=1),
            nn.SiLU()
        )
        self.scale_head = ScaleHead(in_ch=ch)
        self.he_attention = HEGuidedAttention(
            scales=[1.0, 2.0, 4.0],
            gamma=2.0,
            kappa=0.5,
            alpha=1.0
        )   
        self.lmc = None
        self.lmc_d_token = 128    
        self.lmc_heads   = 4      
        self.lmc_token_mode = "gap"  
        self.T_st  = ATransformer(in_ch=2*self.gene_num, out_ch=128*2, d=256, depth=2, nheads=4)
        self.T_wsi = ATransformer(in_ch=3, out_ch=128, d=256, depth=2, nheads=4)
        self.inject_alpha = 1e-3  # 0.001

        self.lmc = LightweightModuleCoordinator(
            in_channels_list=[128, 128, 128, 128],
            d_token=self.lmc_d_token,
            num_heads=self.lmc_heads,
            token_mode=self.lmc_token_mode,
            latent_dim=128
        )
        self.reduce_F4 = torch.nn.Conv2d(192, 128, kernel_size=1)
        self.proj_sc = SCGPTReproject()

    def convert_to_fp16(self):
        """
        Functionality:
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Functionality:
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        
    def forward(self, x, timesteps,low_res, WSI_5120,WSI_320, gene_class, Gene_index_map, metadata_feature,scale_gt, co_expression,WSI_mask,sc,scgpt,pre_he,last_HQST):
        """
        Functionality:
        The forward pass involves multi-modal feature integration, UNet encoding/decoding, 
        attention mechanisms, and loss calculation for various components (OT, GMSD, scale prediction).

        Inputs:
        :param x: High-resolution noisy input feature map.
        :param timesteps: Current timestep.
        :param low_res: Low-resolution spatial feature map.
        :param WSI_5120: High-resolution WSI image feature.
        :param WSI_320: Low-resolution WSI image feature.
        :param sc: Single-cell expression data.
        :param co_expression: Gene co-expression matrix.
        # ... (and other contextual inputs)
        
        Outputs:
        :return: A tuple containing intermediate features/losses (com_WSI, com_spot, dist_WSI, dist_spot, GMSD_loss, attention_weights, Constrained_loss, L_ot, scale_pred) and the final denoised output map (self.out(h)).
        """
        ratio = x[0, 0, 0, 0]
        x = x[:, int(x.shape[1] / 2):x.shape[1], ...]

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        WSI_5120=WSI_5120/255
        WSI_320=th.reshape(WSI_320,(-1,WSI_320.shape[2],WSI_320.shape[3],WSI_320.shape[4]))/255  

        h_x = x.type(self.dtype)    
        h_spot = low_res.type(self.dtype)              
        h_5120WSI = WSI_5120.type(self.dtype)#        
        h_320WSI = WSI_320.type(self.dtype)       
        scgpt = scgpt.type(self.dtype)
        scgpt = self.proj_sc(scgpt) 

        if hasattr(self, "ot_align") and (self.ot_align is not None) and (sc is not None):
            if sc.ndim == 3:

                sc_for_ot = sc[0]
            else:
                sc_for_ot = sc

            h_spot, L_ot, ot_feat = self.ot_align(sc_for_ot, h_spot)
        else:
            L_ot = torch.zeros([], device=h_spot.device, dtype=h_spot.dtype)
            ot_feat = None

        pre_he = pre_he.type(self.dtype)
        pre_he = self.pre_he_block(pre_he) 
        
        h_spot = self.replacer.replace_with_noise(h_spot, ratio, dtype=torch.float32)


        for idx in range(len(self.input_blocks)):

            h_x = self.input_blocks[idx](h_x, emb)
            h_spot = self.input_blocks_lr[idx](h_spot, emb) 
            h_5120WSI = self.input_blocks_WSI5120[idx](h_5120WSI, emb) 
            hs.append((1 / 3) * h_x + (1 / 3) * F.interpolate(h_spot,(h_x.shape[2],h_x.shape[3])) + (1 / 3) * h_5120WSI)
        for idx in range(len(self.input_blocks_WSI320)):
            h_320WSI = self.input_blocks_WSI320[idx](h_320WSI, emb)
        h_5120WSI = h_5120WSI+0.00010*pre_he
        h_x = h_x + 0.00010 * scgpt
        
        F1 = h_x
        h_ori=h_temp = h_x
        h_temp=self.pre(h_temp)
        h_temp = self.MemoryNetwork(h_temp,memory_bank = co_expression)
        h_temp = F.relu(h_temp)
        h_temp = self.post(h_temp)
        h_x=h_ori*0.999+h_temp*0.001

        h_x, attention_weights = self.he_attention(
        h_x, 
        WSI_mask, 
        return_weights=True)
        F2 = h_x
        Constrained_loss = self.Constrained_Refinement(h_5120WSI, last_HQST, WSI_mask)

        if ratio == 2.0:
            ratio = 1.0

        h_320WSI = th.reshape(h_320WSI, (h_x.shape[0], -1, h_320WSI.shape[1], h_320WSI.shape[2], h_320WSI.shape[3]))
        h_320WSI = h_320WSI[:, 0:int(h_320WSI.shape[1] * ratio), ...] 
        h_320WSI = th.mean(h_320WSI, dim=1) 
        h_320WSI = F.interpolate(h_320WSI, size=(h_5120WSI.shape[2], h_5120WSI.shape[3])) 
        h_320WSI=th.reshape(h_320WSI,(h_320WSI.shape[0],h_320WSI.shape[1],-1))
        h_320WSI=th.transpose(h_320WSI,1,2) 

        h_5120WSI_pre=th.reshape(h_5120WSI,(h_5120WSI.shape[0],h_5120WSI.shape[1],-1))
        h_5120WSI_pre = th.transpose(h_5120WSI_pre,1,2)    

        q = self.to_q(h_5120WSI_pre)  
        k = self.to_k(h_320WSI) 
        v = self.to_v(h_320WSI) 
        mid_atten=torch.matmul(q,th.transpose(k,1,2))

        scale = q.shape[2] ** -0.5
        mid_atten=mid_atten*scale
        sfmax = nn.Softmax(dim=-1)
        mid_atten=sfmax(mid_atten)
        WSI_atten = torch.matmul(mid_atten, v)  
        WSI_atten=th.transpose(WSI_atten,1,2) 
        WSI_atten = th.reshape(WSI_atten, (WSI_atten.shape[0],WSI_atten.shape[1], h_5120WSI.shape[2], h_5120WSI.shape[3]))# [N x 16 x 64 x64 ]

        WSI_atten=0.9*h_5120WSI+0.1*WSI_atten
        F3 = WSI_atten


        GMSD_loss = self.GMSD_loss(WSI_atten.clone().cuda(), h_spot.clone().cuda())
        
        com_WSI = self.conv_common(WSI_atten)  
        com_spot = self.conv_common(h_spot)
        com_spot =F.interpolate(com_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3]))  

        dist_WSI = self.conv_distinct(WSI_atten)  
        dist_spot = self.conv_distinct(h_spot)
        dist_spot = F.interpolate(dist_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3]))    

        com_h = (1 / 2) * com_WSI + (1 / 2) * com_spot  

        part=2
        part_width=int(dist_WSI.shape[2]/part)
        WSI_part_dist=dist_WSI
        spot_part_dist = dist_spot
        for i in range(part):
            for j in range(part):
                WSI_part=dist_WSI[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width] # [N x 8 x 32 x 32]
                spot_part = dist_spot[..., i * part_width:(i + 1) * part_width, j * part_width:(j + 1) * part_width] # [N x 8 x 32 x 32]
                WSI_part = th.reshape(WSI_part, (WSI_part.shape[0], WSI_part.shape[1], -1)) # [N x 8 x 1024]
                spot_part = th.reshape(spot_part, (spot_part.shape[0], spot_part.shape[1], -1))  # [N x 8 x 1024]
                WSI_part_T = th.transpose(WSI_part, 1, 2)  # [N x 1024 x 8 ]
                spot_part_T = th.transpose(spot_part, 1, 2)  # [N x 1024 x 8 ]

                F_WSItoSpot=th.matmul(spot_part_T,WSI_part)# [N x 1024 x 1024]
                w_WSItoSpot=self.fc_modulation_1(F_WSItoSpot)# [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_WSItoSpot=sfmax_module(w_WSItoSpot)# [N x 1024 x 1024]
                spot_part_out = th.matmul(spot_part, w_WSItoSpot)  # [N x 8 x 1024]
                spot_part_out = th.reshape(spot_part_out, (spot_part_out.shape[0],spot_part_out.shape[1],
                                                         int(math.sqrt(spot_part_out.shape[2])), int(math.sqrt(spot_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                spot_part_dist[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width]=spot_part_out

                F_SpottoWSI = th.matmul(WSI_part_T, spot_part)  # [N x 1024 x 1024]
                w_SpottoWSI = self.fc_modulation_2(F_SpottoWSI)  # [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_SpottoWSI = sfmax_module(w_SpottoWSI)  # [N x 1024 x 1024]
                WSI_part_out = th.matmul(WSI_part, w_SpottoWSI)  # [N x 8 x 1024]
                WSI_part_out = th.reshape(WSI_part_out, (WSI_part_out.shape[0], WSI_part_out.shape[1], int(math.sqrt(WSI_part_out.shape[2])),
                                                         int(math.sqrt(WSI_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                WSI_part_dist[..., i * part_width:(i + 1) * part_width,j * part_width:(j + 1) * part_width] = WSI_part_out
        # weight
        WSI_part_dist = 0.9*dist_WSI+0.1*WSI_part_dist
        spot_part_dist =  0.9*dist_spot+0.1*spot_part_dist
        h_condition = th.cat([com_h, WSI_part_dist,spot_part_dist], dim=1) # [N x 24 x 64 x 64]
        F4 = h_condition
        # cross attention for embedding condition
        h_condition_pre = th.reshape(h_condition, (h_condition.shape[0], h_condition.shape[1], -1))
        h_condition_pre = th.transpose(h_condition_pre, 1, 2)  # [N x 4096 x 24]

        h_x_pre = th.reshape(h_x, (h_x.shape[0], h_x.shape[1], -1))
        h_x_pre = th.transpose(h_x_pre, 1, 2)    

        q = self.to_q_con(h_x_pre)    
        k = self.to_k_con(h_condition_pre)    
        v = self.to_v_con(h_condition_pre)    
        mid_atten = torch.matmul(q, th.transpose(k, 1, 2))

        scale = q.shape[2] ** -0.5
        mid_atten = mid_atten * scale
        sfmax = nn.Softmax(dim=-1)
        mid_atten = sfmax(mid_atten)  # [N x 4096 x 4096]
        Final_merge = torch.matmul(mid_atten, v)    
        Final_merge = th.transpose(Final_merge, 1, 2)  # [N x 16 x 4096 ]
        Final_merge = th.reshape(Final_merge, (Final_merge.shape[0], Final_merge.shape[1], h_x.shape[2], h_x.shape[3]))  # [N x 16 x 64 x64 ]
        
        meta_processor = MetadataFeatureProcessor(out_channels=20, spatial_size=64)
        batch_size = x.shape[0]  
        device = Final_merge.device 
        
        processed_meta = meta_processor(metadata_feature, batch_size)  
        # processed_gene = processed_gene.to(device)
        processed_meta = processed_meta.to(device)# 
        #new1.5#t2
        Gene_index_map = Gene_index_map.float() 
        Gene_index_map1 = torch.nn.functional.interpolate(Gene_index_map, (64, 64)) 
        Gene_index_map1 = Gene_index_map1.to(device)

        Final_merge = th.cat([Final_merge,Gene_index_map1,processed_meta], dim=1)
        
        h = self.conv_layer1(Final_merge)
        scale_pred = self.scale_head(h)

        expected_channels = [F1.shape[1], F2.shape[1], F3.shape[1], F4.shape[1]]
        if hasattr(self.lmc, "in_channels_list") and self.lmc.in_channels_list != expected_channels:
            self.lmc.in_channels_list = expected_channels
            self.lmc.rebuild_weights(expected_channels, latent_dim=h.shape[1])
        F4 = self.reduce_F4(F4)
        _out_feats, h_injected = self.lmc([F1, F2, F3, F4], z=h)
        
        if h_injected is not None:
            h = h+0.0001*h_injected  

        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        scale_pred = scale_pred.squeeze(-1)    

        return com_WSI, com_spot,  dist_WSI, dist_spot, GMSD_loss, attention_weights,Constrained_loss,L_ot,scale_pred, self.out(h)
      
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeneMemoryNetwork(nn.Module):
    """
    Functionality:
    A specialized network that refines input spatial gene expression features 
    by leveraging a pre-computed global gene co-expression matrix (memory bank). 
    It uses attention mechanisms to query the memory and modulate the gene features.
    """
    def __init__(self, query_dim, num_genes, gene_dim):
        """
        Inputs:
        :param query_dim: The number of channels for the single gene query vector (e.g., 3).
        :type query_dim: int
        :param num_genes: The number of genes in the gene memory bank (e.g., 200).
        :type num_genes: int
        :param gene_dim: The feature dimension for each gene (e.g., 32).
        :type gene_dim: int
        """
        super(GeneMemoryNetwork, self).__init__()
        self.query_dim = query_dim
        self.num_genes = num_genes
        self.gene_dim = gene_dim

        # Projection layers for Cross-Attention
        self.WQ = nn.Linear(gene_dim, gene_dim)
        self.WK = nn.Linear(gene_dim, gene_dim)
        self.WV = nn.Linear(gene_dim, gene_dim)

        # External or placeholder modules
        self.meta_updater = MetaMemoryUpdater(gene_dim)
        self.sparse_attention = SparseAttention(sparsity=0.3)

        # Multi-Head Cross-Attention to combine query with memory
        self.cross_attention = nn.MultiheadAttention(embed_dim=gene_dim, num_heads=8)

        # Final projection to restore the spatial feature dimensions (e.g., 32*32 or 64*64)
        self.restore_channels = nn.Linear(gene_dim, 64 * 64)

    def forward(self, single_gene_data, memory_bank):
        """
        Functionality:
        Processes the input gene data by extracting queries, adapting the global memory 
        based on these queries, applying sparse attention, and fusing the results 
        using cross-attention before restoring the original spatial shape.

        Inputs:
        :param single_gene_data: Input spatial gene feature map. Shape: [batch, query_dim, 32, 32].
        :type single_gene_data: torch.Tensor
        :param memory_bank: Global gene co-expression matrix. Shape: [num_genes, num_genes].
        :type memory_bank: Union[torch.Tensor, np.ndarray]

        Outputs:
        :return: The refined spatial gene feature map. Shape: [batch, query_dim, 32, 32].
        :rtype: torch.Tensor
        """
        batch_size, _, height, width = single_gene_data.shape


        queries = self.extract_queries(single_gene_data)  # [batch, query_dim, gene_dim]


        local_memory = self.extract_local_memory(single_gene_data)  # [batch, query_dim]


        global_memory = self.extract_gene_features(memory_bank)  # [num_genes, gene_dim]
        adapted_memory = self.meta_updater.adapt_memory(global_memory, queries)  # [num_genes, gene_dim]


        sparse_memory = self.sparse_attention(queries, adapted_memory)  # [batch, query_dim, gene_dim]


        local_memory = local_memory.unsqueeze(2).expand(-1, -1, sparse_memory.size(-1))
        combined_memory = torch.cat([local_memory, sparse_memory], dim=1)  # [batch, query_dim*2, gene_dim]

        # QKV Projections
        query_proj = self.WQ(queries)
        key_proj = self.WK(combined_memory)
        value_proj = self.WV(combined_memory)

        # Transpose for MultiheadAttention (L, N, E)
        query_proj = query_proj.permute(1, 0, 2)  # [query_dim, batch, gene_dim]
        key_proj = key_proj.permute(1, 0, 2)
        value_proj = value_proj.permute(1, 0, 2)

        # Cross-Attention
        attn_output, _ = self.cross_attention(query_proj, key_proj, value_proj)
        attn_output = attn_output.permute(1, 0, 2)  # [batch, query_dim, gene_dim]

        # Restore to spatial shape
        restored = self.restore_channels(attn_output)
        restored = restored.view(batch_size, self.query_dim, height, width)

        return restored

    def extract_local_memory(self, single_gene_data):
        """
        Functionality: Extracts a mean vector representing local feature context.
        """
        return single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]

    def extract_queries(self, single_gene_data):
        """
        Functionality: Extracts mean feature vectors to serve as queries for the memory bank.
        """
        batch_size, channels, height, width = single_gene_data.shape
        queries = single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]
        # Expand last dimension to match gene_dim
        return queries.unsqueeze(-1).expand(-1, -1, self.gene_dim)  # [batch, query_dim, gene_dim]

    def extract_gene_features(self, memory_bank):
        """
        Functionality:
        Converts the global co-expression matrix into a set of 1D gene features 
        that match the model's feature dimension (gene_dim).

        Inputs:
        :param memory_bank: Co-expression matrix. Shape: [num_genes, num_genes].

        Outputs:
        :return: Gene features ready for memory interaction. Shape: [num_genes, gene_dim].
        :rtype: torch.Tensor
        """
        if not isinstance(memory_bank, torch.Tensor):
            memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
        memory_bank = memory_bank.float().to(next(self.parameters()).device)

        if memory_bank.dim() > 2:
            memory_bank = memory_bank[0]  
        elif memory_bank.dim() == 1:
            memory_bank = memory_bank.unsqueeze(0)


        gene_features = memory_bank.mean(dim=-1)  # [num_genes]
        # Expand feature vector to match the required gene_dim
        return gene_features.unsqueeze(1).expand(-1, self.gene_dim)  # [num_genes, gene_dim]
      

class MetaMemoryUpdater(nn.Module):
    """
    Functionality:
    Adjusts the global memory bank using meta-learning principles based on 
    the current task's query features.
    """
    def __init__(self, gene_dim):
        """
        Inputs:
        :param gene_dim: The feature dimension for each gene.
        """
        super().__init__()
        self.adaptive_layer = nn.Linear(gene_dim, gene_dim)

    def adapt_memory(self, memory_bank, query_features):
        """
        Functionality:
        Dynamically adjusts the memory bank based on the query features.

        Inputs:
        :param memory_bank: Global co-expression memory. Shape: [num_genes, gene_dim].
        :type memory_bank: torch.Tensor
        :param query_features: Gene features from the current batch/task. Shape: [batch, query_dim, gene_dim].
        :type query_features: torch.Tensor

        Outputs:
        :return: The adapted memory bank. Shape: [num_genes, gene_dim].
        :rtype: torch.Tensor
        """
        device = memory_bank.device
        task_adaptation = query_features.mean(dim=0)  # [query_dim, gene_dim]
        # Calculate task adaptation offset (delta)
        delta = self.adaptive_layer(task_adaptation.mean(dim=0)).to(device)
        adapted_memory = memory_bank + delta
        return adapted_memory

class SparseAttention(nn.Module):
    """
    Functionality:
    Implements a sparse attention mechanism where each query only attends to the 
    top-K most relevant entries in the memory bank, enforcing sparsity.
    """
    def __init__(self, sparsity=0.1):
        """
        Inputs:
        :param sparsity: The ratio of memory entries to keep (K/num_genes).
        """
        super().__init__()
        self.sparsity = sparsity

    def forward(self, query, memory):
        """
        Functionality:
        Computes attention scores, selects the top-K memory entries, and returns 
        the weighted sum of these sparse memory entries.

        Inputs:
        :param query: Query features from the current batch. Shape: [batch, query_dim, gene_dim].
        :type query: torch.Tensor
        :param memory: Global co-expression memory. Shape: [num_genes, gene_dim].
        :type memory: torch.Tensor

        Outputs:
        :return: Sparsely attended memory. Shape: [batch, query_dim, gene_dim].
        :rtype: torch.Tensor
        """
        scores = torch.matmul(query, memory.T)  # [batch, query_dim, num_genes]
        topk = max(1, int(memory.size(0) * self.sparsity))
        topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)

        batch_size, query_dim, _ = topk_scores.shape
        gene_dim = memory.size(-1)

        # Gather the top-K memory entries and apply weights
        sparse_memory = memory[topk_indices.view(-1)].view(batch_size, query_dim, topk, gene_dim)
        sparse_weights = torch.softmax(topk_scores, dim=-1)
        weighted_memory = torch.einsum('bqt,bqtd->bqd', sparse_weights, sparse_memory)
        return weighted_memory

class FeatureNoiseReplacer:
    """
    Functionality:
    Replaces a random subset of channels in the feature map with Gaussian noise 
    based on a dynamic probability ratio. This is often used for feature dropout 
    or augmentation in diffusion models.
    """
    def __init__(self, replacement_prob=0.8):
        """
        Inputs:
        :param replacement_prob: Base probability for channel replacement (though dynamically scaled in the method).
        """
        self.replacement_prob = replacement_prob
    
    def replace_with_noise(self, feature_map, ratio, dtype=torch.float32):
        """
        Functionality:
        Selects a dynamic subset of channels (C/2) and replaces their values with 
        random noise for a given sample in the batch based on the calculated `ratio`.
        
        Inputs:
        :param feature_map: The input feature map. Shape: [B, C, H, W].
        :type feature_map: torch.Tensor
        :param ratio: A value used to determine the probability of applying noise.
        :type ratio: float
        :param dtype: The data type of the generated noise.
        :type dtype: torch.dtype
        
        Outputs:
        :return: The feature map with noise injected into selected channels.
        :rtype: torch.Tensor
        """
        if ratio == 2.0:
            ratio = 0.2
        elif ratio <0.5:
            ratio = 0.0
        else:
            ratio = ratio / 4.0

        B, C, H, W = feature_map.shape
        
        replaced_feature = feature_map.clone()

        for b in range(B):
            if torch.rand(1, device=feature_map.device).item() < ratio:
                half_channels = C // 2
                selected_channels = torch.randperm(C, device=feature_map.device)[:half_channels]

                noise = torch.randn((half_channels, H, W),
                                     dtype=dtype,
                                     device=feature_map.device)

                replaced_feature[b, selected_channels] = noise

        return replaced_feature

class ScaleHead(nn.Module):
    """
    Functionality:
    Predicts the patch-level upscaling factor S from the low-resolution feature map.
    Expected input: [B, C, H, W], where C is typically 64~256 and H, W are 32~64.
    """
    def __init__(self, in_ch: int, hidden: int = 128):
        """
        Inputs:
        :param in_ch: Input channel dimension (C).
        :type in_ch: int
        :param hidden: Hidden layer dimension for the MLP.
        :type hidden: int
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(1),              # Flatten: [B, C, 1, 1] -> [B, C]
            nn.BatchNorm1d(in_ch),
            nn.Linear(in_ch, hidden),
            nn.SiLU(),                  # Activation function
            nn.Linear(hidden, 1),
            nn.Softplus(beta=5.0)       # Ensures positive output, suitable for scale prediction
        )
        nn.init.constant_(self.net[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Functionality:
        Applies the pooling and MLP sequence to predict the scale factor.

        Inputs:
        :param x: Input feature map. Shape: [B, C, H, W].
        :type x: torch.Tensor

        Outputs:
        :return: Predicted scale factor. Shape: [B].
        :rtype: torch.Tensor
        """
        return self.net(x).squeeze(-1)  # [B]


from typing import Tuple, List, Optional
from torchvision.transforms import GaussianBlur

class HEGuidedAttention(nn.Module):
    """
    Functionality:
    Modulates input features (e.g., gene expression) using a spatial attention map 
    derived from the structural saliency of a corresponding H&E image. This guides 
    the feature processing to focus on structurally rich tissue regions.
    """
    def __init__(
        self,
        scales: List[float] = [1.0, 2.0, 4.0],
        scale_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        kappa: float = 0.5,
        alpha: float = 1.0,
        epsilon: float = 1e-3
    ):
        """
        Inputs:
        :param scales: Standard deviations (sigma) for the Gaussian blur at different scales.
        :param scale_weights: Weights for combining the saliency maps from different scales.
        :param gamma: Exponent for non-linear attention enhancement ($\text{Attention} \propto \text{Saliency}^\gamma$).
        :param kappa: Scaling factor for the gating operation ($1 + \kappa W$).
        :param alpha: Unused parameter (kept for completeness or future expansion).
        :param epsilon: Small value for numerical stability.
        """
        super().__init__()
        
        assert gamma >= 1.0, "gamma must be >= 1.0"
        assert kappa > 0, "kappa must be > 0"
        assert alpha > 0, "alpha must be > 0"
        
        self.scales = scales
        self.scale_weights = scale_weights if scale_weights else [1.0] * len(scales)
        self.gamma = gamma
        self.kappa = kappa
        self.alpha = alpha
        self.epsilon = epsilon
        
        assert len(self.scales) == len(self.scale_weights), "scales and scale_weights must have the same length"

    def compute_gradient_magnitude(
        self, 
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Functionality:
        Computes the gradient magnitude (edge strength) using Sobel filters.

        Inputs:
        :param image: Input feature map (single channel expected). Shape: [B, C, H, W].

        Outputs:
        :return: Gradient magnitude map. Shape: [B, 1, H, W].
        """
        if image.shape[1] > 1:
            image = image.mean(dim=1, keepdim=True)
        
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return gradient_mag
    
    def compute_structural_saliency(
        self, 
        he_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Functionality:
        Computes a multi-scale structural saliency map from the H&E image 
        by combining normalized gradient magnitudes at various Gaussian scales.

        Inputs:
        :param he_image: Input H&E image. Shape: [B, C, H, W].

        Outputs:
        :return: The aggregated and normalized structural saliency map. Shape: [B, 1, H, W].
        """
        saliency = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            # Apply Gaussian smoothing
            kernel_size = int(6 * scale + 1)  # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            blurred = GaussianBlur(
                kernel_size=kernel_size, 
                sigma=scale
            )(he_image)
            
            # Compute gradient magnitude
            grad_mag = self.compute_gradient_magnitude(blurred)
            
            # Normalize by max
            grad_max = grad_mag.flatten(1).max(dim=1, keepdim=True)[0]
            grad_max = grad_max.view(-1, 1, 1, 1)
            grad_normalized = grad_mag / (grad_max + 1e-8)
            
            # Weighted accumulation
            saliency = saliency + weight * grad_normalized
        
        # Normalize to [0, 1]
        saliency = saliency / sum(self.scale_weights)
        
        return saliency
    
    def apply_spatial_transform(
        self,
        saliency_map: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Functionality:
        Applies a spatial transformation (e.g., affine or elastic) to the saliency map
        using grid sampling, often to align it with the feature map coordinate system.

        Inputs:
        :param saliency_map: The structural saliency map. Shape: [B, 1, H, W].
        :param spatial_transform: The grid for spatial warping. Shape: [B, H, W, 2].

        Outputs:
        :return: The transformed saliency map.
        """
        if spatial_transform is None:
            return saliency_map
        
        # Apply grid sampling
        transformed = F.grid_sample(
            saliency_map,
            spatial_transform,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return transformed
    
    def compute_attention_weights(
        self,
        he_image: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Functionality:
        Calculates the final attention weights by computing saliency, applying
        spatial alignment, normalizing, and non-linearly enhancing the map.

        Inputs:
        :param he_image: Input H&E image. Shape: [B, C, H, W].
        :param spatial_transform: Grid for optional spatial alignment.

        Outputs:
        :return: The final attention weight map. Shape: [B, 1, H, W].
        """
        saliency = self.compute_structural_saliency(he_image)
        
        # Apply spatial transformation
        saliency_transformed = self.apply_spatial_transform(
            saliency, 
            spatial_transform
        )
        
        # Normalize to [0, 1]
        saliency_min = saliency_transformed.flatten(1).min(dim=1, keepdim=True)[0]
        saliency_max = saliency_transformed.flatten(1).max(dim=1, keepdim=True)[0]
        saliency_min = saliency_min.view(-1, 1, 1, 1)
        saliency_max = saliency_max.view(-1, 1, 1, 1)
        
        saliency_norm = (saliency_transformed - saliency_min) / (
            saliency_max - saliency_min + 1e-8
        )
        
        # Nonlinear enhancement
        attention_weights = torch.pow(saliency_norm, self.gamma)
        
        return attention_weights
    
    def apply_feature_gating(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Functionality:
        Applies the attention weights to the input features using a gating mechanism.
        
        Inputs:
        :param features: Input feature map. Shape: [B, C, H, W].
        :param attention_weights: The calculated attention map. Shape: [B, 1, H', W'].

        Outputs:
        :return: The gated output features.
        """
        if attention_weights.shape[-2:] != features.shape[-2:]:
            attention_weights = F.interpolate(
                attention_weights,
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Apply gating: $F^{\text{out}} = F^{\text{in}} \odot (1 + \kappa W)$
        gating_factor = 1.0 + self.kappa * attention_weights
        features_out = features * gating_factor
        
        return features_out
    
    def forward(
        self,
        features: torch.Tensor,
        he_image: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Functionality:
        Executes the H&E guided attention by computing weights and applying gating.

        Inputs:
        :param features: Input features to be modulated.
        :param he_image: H&E image used to derive structural saliency.
        :param spatial_transform: Optional grid for spatial alignment.
        :param return_weights: If True, returns the attention map along with the features.

        Outputs:
        :return: A tuple containing the gated features and the attention weights (if requested).
        """
        attention_weights = self.compute_attention_weights(
            he_image, 
            spatial_transform
        )
        
        # Apply feature gating
        gated_features = self.apply_feature_gating(features, attention_weights)
        
        if return_weights:
            return gated_features, attention_weights
        else:
            return gated_features, None


class SimpleGraphEncoder(nn.Module):
    """
    Functionality:
    A basic Graph Neural Network (GNN) encoder for feature aggregation based on 
    an adjacency matrix. It performs two layers of graph convolution (message passing).
    """
    def __init__(self, in_dim, hidden_dim=64):
        """
        Inputs:
        :param in_dim: Input feature dimension per node.
        :param hidden_dim: Output/hidden feature dimension per node.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, adj):
        """
        Functionality:
        Performs two layers of graph convolution: $H' = \sigma(A X W)$.

        Inputs:
        :param X: Node features. Shape: [n_nodes, in_dim].
        :param adj: Adjacency/normalized affinity matrix. Shape: [n_nodes, n_nodes].

        Outputs:
        :return: Encoded node features. Shape: [n_nodes, hidden_dim].
        """
        h = torch.relu(self.fc1(torch.matmul(adj, X)))
        h = torch.relu(self.fc2(torch.matmul(adj, h)))
        return h


class Constrained_Refinement(nn.Module):
    """
    Functionality:
    Implements a cell-level graph loss to enforce structural consistency between 
    H&E (histological) features and HQST (spatial transcriptomics) features 
    within highly informative regions.
    """
    def __init__(self, gene_num=20, feat_dim=64, radius=3):
        """
        Inputs:
        :param gene_num: Number of genes in the HQST data.
        :param feat_dim: The feature dimension used for the GNN and padding.
        :param radius: Spatial radius parameter for defining graph connectivity (adjacency).
        """
        super().__init__()
        self.gene_num = gene_num
        self.feat_dim = feat_dim
        self.gnn = SimpleGraphEncoder(in_dim=feat_dim, hidden_dim=feat_dim)
        self.radius = radius

    def forward(self, HE, HQST, cell_mask):
        """
        Functionality:
        Calculates a loss by comparing the cell-cell correlation matrices derived 
        from H&E and HQST features after GNN encoding, focusing on a high-entropy patch.

        Inputs:
        :param HE: H&E image features (e.g., WSI features). Shape: [B, C_HE, H, W].
        :param HQST: High-quality spatial transcriptomics features. Shape: [B, gene_num, H, W].
        :param cell_mask: Segmentation masks for individual cells. Shape: [B, N_cells, H', W'].

        Outputs:
        :return: The total structural refinement loss (mean SSIM-like distance over the batch).
        """
        B = HE.shape[0]
        total_loss = 0.0
        eps = 1e-6

        # Select a patch with high structural entropy
        HE_crop, HQST_crop, coords = self.select_high_entropy_region(
            HE, HQST, patch_size=16, return_coords=True
        )

        for b in range(B):
            HE_b = HE_crop[b]  # [3,16,16]
            HQST_b = HQST_crop[b]  # [gene_num,16,16]
            masks_b = cell_mask[b]  # [N,256,256]
            N = masks_b.shape[0]

            # Crop the cell masks to match the selected high-entropy patch
            row, col, patch_size = coords[b]
            masks_b = masks_b[:, row*patch_size:(row+1)*patch_size,
                                   col*patch_size:(col+1)*patch_size]  # [N,16,16]
            E_HE, E_HQ = [], []
            for i in range(N):
                m = masks_b[i].to(HE.device)
                if m.sum() < 1:
                    continue
                m = m / (m.sum() + eps)
                e_he = (HE_b * m).sum(dim=(1, 2))  # Mean H&E feature within cell mask
                e_hq = (HQST_b * m).sum(dim=(1, 2)) # Mean HQST feature within cell mask
                E_HE.append(e_he)
                E_HQ.append(e_hq)
            if len(E_HE) < 2:
                continue

            E_HE = torch.stack(E_HE)  # [n_valid, 3]
            E_HQ = torch.stack(E_HQ)  # [n_valid, gene_num]

            # Calculate cell centers for graph construction
            coords_cell = []
            for i in range(N):
                m = masks_b[i]
                ys, xs = torch.where(m > 0)
                if len(xs) > 0:
                    coords_cell.append(torch.stack([xs.float().mean(), ys.float().mean()]))
            if len(coords_cell) < 2:
                continue

            # Graph Construction
            coords_cell = torch.stack(coords_cell)
            dist = torch.cdist(coords_cell, coords_cell)
            adj = torch.exp(-dist / (self.radius + 1e-6))
            adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)

            # Feature Padding and GNN Encoding
            pad_he = F.pad(E_HE, (0, self.feat_dim - E_HE.shape[1]))
            pad_hq = F.pad(E_HQ, (0, self.feat_dim - E_HQ.shape[1]))
            G_HE = self.gnn(pad_he, adj)
            G_HQ = self.gnn(pad_hq, adj)

            # Feature Fusion (Element-wise concatenation)
            E_HE_fused = torch.cat([pad_he, G_HE], dim=1)
            E_HQ_fused = torch.cat([pad_hq, G_HQ], dim=1)

            # Correlation Matrix Comparison
            CM_HE = self.corr_matrix(E_HE_fused)
            CM_HQ = self.corr_matrix(E_HQ_fused)

            # Loss: (1 - SSIM) of the correlation matrices
            ssim_val = self.ssim_matrix(CM_HE, CM_HQ)
            ssim_val = torch.clamp(ssim_val, 0.0, 1.0)
            loss = 1 - ssim_val
            total_loss += loss

        total_loss = total_loss / max(B, 1)
        return total_loss

    @staticmethod
    def corr_matrix(X, eps=1e-6):
        """
        Functionality:
        Calculates a stable, normalized cell-cell correlation matrix.

        Inputs:
        :param X: Cell features. Shape: [n_cells, feat_dim].

        Outputs:
        :return: Normalized cell-cell correlation matrix.
        """
        # Stable cell–cell correlation matrix: $\text{Corr} = (X' / \text{std}(X')) (X' / \text{std}(X'))^T / (D-1)$
        X = X - X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True) + eps
        X = X / std
        corr = X @ X.T / (X.shape[1] - 1)
        corr = torch.clamp(corr, -1.0, 1.0)
        return corr.detach()

    def shannon_entropy(self, patch):
        """
        Functionality: Computes the Shannon entropy of a 2D patch for texture/complexity measurement.
        """
        patch = patch.detach().cpu()
        patch = patch - patch.min()
        patch = patch / (patch.max() + 1e-8)
        hist = torch.histc(patch, bins=256, min=0.0, max=1.0)
        prob = hist / (hist.sum() + 1e-8)
        prob = prob[prob > 0]
        entropy = -torch.sum(prob * torch.log2(prob))
        return entropy.item()

    def select_high_entropy_region(self, HE, HQST, patch_size=16, return_coords=False):
        """
        Functionality:
        Selects a patch from the HE image that exhibits the maximum Shannon entropy 
        (i.e., the most complex/textured region).
        """
        B, _, H, W = HE.shape
        n_h = H // patch_size
        n_w = W // patch_size

        HE_crops, HQST_crops, coords = [], [], []

        for b in range(B):
            # Convert to grayscale for entropy calculation
            gray = 0.2989 * HE[b, 0] + 0.5870 * HE[b, 1] + 0.1140 * HE[b, 2]
            entropies = torch.zeros((n_h, n_w))
            for i in range(n_h):
                for j in range(n_w):
                    patch = gray[i*patch_size:(i+1)*patch_size,
                                     j*patch_size:(j+1)*patch_size]
                    entropies[i, j] = self.shannon_entropy(patch)

            max_idx = torch.argmax(entropies)
            row, col = divmod(max_idx.item(), n_w)
            coords.append((row, col, patch_size))

            # Crop the H&E and HQST features
            HE_crop = HE[b, :, row*patch_size:(row+1)*patch_size,
                              col*patch_size:(col+1)*patch_size]
            HQST_crop = HQST[b, :, row*patch_size:(row+1)*patch_size,
                                 col*patch_size:(col+1)*patch_size]

            HE_crops.append(HE_crop)
            HQST_crops.append(HQST_crop)

        if return_coords:
            return torch.stack(HE_crops), torch.stack(HQST_crops), coords
        else:
            return torch.stack(HE_crops), torch.stack(HQST_crops)

    def ssim_matrix(self, x, y, C1=1e-4, C2=1e-4):
        """
        Functionality:
        Calculates a Structural Similarity Index (SSIM)-like measure between 
        two matrices (CM_HE and CM_HQ).
        """
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = x.var()
        sigma_y = y.var()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim
      
class TinyBiEncoder(nn.Module):
    """
    Functionality:
    A minimal bi-encoder for two inputs (x, y) with a simple learned gate 
    (weight 'w') to produce a fused output 'g'.
    """
    def __init__(self, G, d):
        """
        Inputs:
        :param G: Input dimension for both x and y.
        :param d: Embedding dimension.
        """
        super().__init__()
        self.proj_x = nn.Linear(G, d, bias=False)
        self.proj_y = nn.Linear(G, d, bias=False)
        self.ln_x = nn.LayerNorm(d)
        self.ln_y = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.zeros(d))  # Learnable gate parameter

    def forward(self, x, y):
        """
        Functionality:
        Encodes x and y, and produces a gated fusion output: $g = \sigma(\alpha) h_x + (1 - \sigma(\alpha)) h_y$.

        Inputs:
        :param x: First input tensor.
        :param y: Second input tensor.

        Outputs:
        :return: Encoded x ($h_x$), Encoded y ($h_y$), and Fused output ($g$).
        """
        hx = self.ln_x(F.gelu(self.proj_x(x)))
        hy = self.ln_y(F.gelu(self.proj_y(y)))
        w = torch.sigmoid(self.alpha)
        g = w * hx + (1 - w) * hy
        return hx, hy, g
      
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    """
    Functionality:
    Performs multi-modal feature fusion and contrastive learning loss calculation 
    (NCE Loss, Alignment Loss) on two single-cell feature representations ($x$ and $y$).
    """
    def __init__(self, gene_dim=1000, embed_dim=64, lambda_mm=1.0):
        """
        Inputs:
        :param gene_dim: Input dimension of single-cell features.
        :param embed_dim: Embedding dimension after projection.
        :param lambda_mm: Weight for the multi-modal loss.
        """
        super().__init__()
        self.proj_x = nn.Linear(gene_dim, embed_dim, bias=False)
        self.proj_y = nn.Linear(gene_dim, embed_dim, bias=False)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.ln_y = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.zeros(embed_dim))
        self.lambda_mm = lambda_mm

    def forward(self, sc_dict):
        """
        Functionality:
        Encodes two single-cell feature views ($x, y$), fuses them, and calculates 
        a combined alignment and contrastive (NCE) loss.

        Inputs:
        :param sc_dict: Dictionary containing "ot_x" and "ot_y" single-cell features.

        Outputs:
        :return: Fused feature vector ($g$) and the total multi-modal loss.
        """
        if not isinstance(sc_dict, dict):
            return None, torch.tensor(0.0, device=list(self.parameters())[0].device)

        sc_ot_x = sc_dict.get("ot_x", None)
        sc_ot_y = sc_dict.get("ot_y", None)

        if sc_ot_x is None or sc_ot_y is None:
            return None, torch.tensor(0.0, device=list(self.parameters())[0].device)

        # Type and device handling
        if not isinstance(sc_ot_x, torch.Tensor):
            sc_ot_x = torch.tensor(sc_ot_x, dtype=torch.float32)
        if not isinstance(sc_ot_y, torch.Tensor):
            sc_ot_y = torch.tensor(sc_ot_y, dtype=torch.float32)

        device = self.alpha.device
        sc_ot_x = sc_ot_x.to(device)
        sc_ot_y = sc_ot_y.to(device)

        # Encoding and Fusion
        hx = self.ln_x(F.gelu(self.proj_x(sc_ot_x)))
        hy = self.ln_y(F.gelu(self.proj_y(sc_ot_y)))
        w = torch.sigmoid(self.alpha)
        g = w * hx + (1 - w) * hy  
        
        # Alignment Loss (Direct Cosine Similarity)
        align_loss = (1 - F.cosine_similarity(hx, hy, dim=1)).mean()

        # NCE Loss (Contrastive Learning)
        hx_n = F.normalize(hx, dim=1)
        hy_n = F.normalize(hy, dim=1)
        logits = hx_n @ hy_n.T / 0.07
        target = torch.arange(hx.shape[0], device=device)
        nce_loss = F.cross_entropy(logits, target)

        # Combined Multi-Modal Loss
        mm_loss = self.lambda_mm * (0.5 * align_loss + 0.5 * nce_loss)

        return g, mm_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertTokenizer(nn.Module):
    """
    Functionality:
    Converts a 2D feature map from an 'expert' module into a compact token vector 
    using either Global Average Pooling (GAP) or Attention-based pooling.
    """
    def __init__(self, C: int, d_token: int, mode: str = "gap"):
        """
        Inputs:
        :param C: Input channel dimension.
        :param d_token: Output token dimension.
        :param mode: Pooling mode, either "gap" (Global Average Pooling) or "attn" (Attention Pooling).
        """
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(C, d_token)
        if mode == "attn":
            self.attn_conv = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Functionality:
        Pools the 2D feature map to a 1D vector and projects it to the token space.

        Inputs:
        :param x: Input feature map. Shape: [B, C, H, W].

        Outputs:
        :return: The extracted token. Shape: [B, d_token].
        """
        B, C, H, W = x.shape
        if self.mode == "gap":
            pooled = x.mean(dim=(2, 3))  # [B, C]
        else:
            A = self.attn_conv(x)  # [B,1,H,W]
            A = torch.softmax(A.view(B, 1, -1), dim=-1).view(B, 1, H, W)
            pooled = (x * A).sum(dim=(2, 3))  # [B, C]
        token = self.proj(pooled)  # [B, d]
        return token

import torch
import torch.nn as nn

class TokenNegotiator(nn.Module):
    """
    Functionality:
    A Transformer-style block that allows tokens from different expert modules 
    to interact and negotiate a consensus representation using Multi-Head Attention (MHA).
    """
    def __init__(self, d_token: int, num_heads: int = 4, dropout: float = 0.0, num_modules: int = 4): 
        """
        Inputs:
        :param d_token: Token dimension.
        :param num_heads: Number of attention heads.
        :param num_modules: Expected number of expert tokens (used for optional learned embedding).
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_token, num_heads=num_heads, dropout=dropout, batch_first=True)  
        self.ln1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),
            nn.Linear(4 * d_token, d_token),
        )
        self.ln2 = nn.LayerNorm(d_token)
        self.module_embed = nn.Parameter(torch.randn(1, num_modules, d_token) * 0.02) # Learned position/module embedding

    def forward(self, x):
        """
        Functionality:
        Performs a standard Transformer Encoder block operation (MHA -> Norm -> FFN -> Norm).

        Inputs:
        :param x: Stacked tokens from expert modules. Shape: [B, N_modules, d_token].

        Outputs:
        :return: The refined and negotiated tokens.
        """
        # Multi-Head Attention
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        # Feed-Forward Network
        x = self.ln2(x + self.ffn(x))
        return x

class ChannelModulator(nn.Module):
    """
    Functionality:
    Generates a spatial-independent, channel-wise modulation (Scale and Shift, $\gamma$ and $\beta$) 
    for a feature map based on the negotiated token.
    """
    def __init__(self, d_token: int, C: int):
        """
        Inputs:
        :param d_token: Token dimension (input to MLP).
        :param C: Target feature map channel dimension (output of MLP is 2*C).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_token, 2 * C),
            nn.GELU(),
            nn.Linear(2 * C, 2 * C)
        )
        # Initialize final layer to zero for stable training start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token: torch.Tensor, feat: torch.Tensor):
        """
        Functionality:
        Generates $\gamma$ and $\beta$ from the token and expands them to match 
        the spatial size of the feature map.

        Inputs:
        :param token: Negotiated token. Shape: [B, d_token].
        :param feat: Target feature map. Shape: [B, C, H, W].

        Outputs:
        :return: Channel-wise scale ($\gamma$) and shift ($\beta$) parameters.
        """
        B, C, H, W = feat.shape
        out = self.mlp(token)  # [B,2C]
        gamma_raw, beta_raw = out.chunk(2, dim=-1)  # [B,C], [B,C]
        
        # Soft clamping/scaling
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta  = 0.0 + 0.1 * torch.tanh(beta_raw)
        
        # Expand to match feature map dimensions
        gamma = gamma.view(B, C, 1, 1).expand(B, C, H, W)
        beta  = beta.view(B, C, 1, 1).expand(B, C, H, W)
        return gamma, beta
      
class PreHEBlock(nn.Module):
    """
    Functionality:
    Pre-processes a linearized/tokenized low-resolution H&E feature into a 2D 
    feature map of fixed size [B, 128, 64, 64] through reshaping, upsampling, 
    and channel expansion.
    """
    def __init__(self):
        super().__init__()

        self.channel_expand = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, pre_he):
        """
        Inputs:
        :param pre_he: Linearized H&E features. Shape: [B, 3, 512].

        Outputs:
        :return: Processed H&E feature map. Shape: [B, 128, 64, 64].
        """
        B, C, L = pre_he.shape
        assert C == 3, "the channel of pre_he must be 3"

        H0, W0 = 16, 32  
        assert H0 * W0 == L, "the length of pre_he must be 512"
        x = pre_he.view(B, C, H0, W0)  # [B, 3, 16, 32]

        # Upsample to target resolution (64x64)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)  # [B, 3, 64, 64]

        # Channel expansion
        x = self.channel_expand(x)  # [B, 128, 64, 64]

        return x

class SCGPTReproject(nn.Module):
    """
    Functionality:
    Reprojects and upsamples features from an external model (e.g., SC-GPT) 
    to match the spatial and channel dimensions of the UNet's input/intermediate layers.
    """
    def __init__(self):
        super().__init__()
        self.channel_reduce = nn.Conv2d(512, 128, kernel_size=1)

        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Functionality:
        Reduces channels and upsamples the input feature map.

        Inputs:
        :param x: Input feature map from SCGPT. Shape: [B, 512, 26, 26].

        Outputs:
        :return: Reprojected feature map. Shape: [B, 128, 64, 64].
        """
        x = self.channel_reduce(x)  # (b, 128, 26, 26)
        x = self.upsample(x)  # (b, 128, 64, 64)
        return x
