import torch as th
from torch import nn
from typing import List
import json
import math
from PartialConv import PartialConv2d

def conv_output_size_same_padding(in_size, pool_size):
    return  math.ceil(in_size / pool_size)

def conv_output_size(in_size, kernel, padding, stride):
    return  math.ceil((in_size - kernel + 2 * padding) / stride + 1)


n_channels_string = "n_channels"
image_nrows_string = "cutted_nrows"
image_ncols_string = "cutted_ncols"

model_cathegory_string: str = "models"
dataset_cathegory_string: str = "dataset"

def initialize_model_and_dataset_kind(params, model_kind: str) -> tuple[nn.Module, str]:
    """Initialize the model and dataset kind from the json file.

    Args:
        params: json file with parameters
        model_kind (str): kind of model to initialize

    Raises:
        ValueError: if the model kind is not recognized

    Returns:
        tuple[nn.Module, str]: model and dataset kind
    """
    
    if model_kind == "simple_conv":
        model = simple_conv(params)
        dataset_kind = "extended"
    elif model_kind == "conv_maxpool":
        model = conv_maxpool(params)
        dataset_kind = "extended"
    elif model_kind == "conv_unet":
        model = conv_unet(params)
        dataset_kind = "extended"
    elif model_kind == "DINCAE_like":
        model = DINCAE_like(params)
        dataset_kind = "extended"
    elif model_kind == "DINCAE_pconvs":
        model = DINCAE_pconvs(params)
        dataset_kind = "minimal"
    else:
        raise ValueError(f"Model kind {model_kind} not recognized")
    
    return model, dataset_kind
class DINCAE_like(nn.Module):
    def __init__(self, params, n_channels: int = None, image_nrows: int = None, image_ncols: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None):
        super(DINCAE_like, self).__init__()
        
        self.model_name: str = "DINCAE_like"
        
        self.n_channels = n_channels
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        
        self._load_configurations(params)
        
        w, h = self._calculate_sizes()
        
        self.conv1 = nn.Conv2d(self.n_channels, self.middle_channels[0], self.kernel_sizes[0], padding = self.kernel_sizes[0] // 2)
        self.pool1 = nn.MaxPool2d(self.pooling_sizes[0], ceil_mode=True)
        
        self.conv2 = nn.Conv2d(self.middle_channels[0], self.middle_channels[1], self.kernel_sizes[1], padding = 'same')
        self.pool2 = nn.MaxPool2d(self.pooling_sizes[1], ceil_mode=True)
        
        self.conv3 = nn.Conv2d(self.middle_channels[1], self.middle_channels[2], self.kernel_sizes[2], padding = 'same')
        self.pool3 = nn.MaxPool2d(self.pooling_sizes[2], ceil_mode=True)
        
        self.conv4 = nn.Conv2d(self.middle_channels[2], self.middle_channels[3], self.kernel_sizes[3], padding = 'same')
        self.pool4 = nn.MaxPool2d(self.pooling_sizes[3], ceil_mode=True)
        
        self.conv5 = nn.Conv2d(self.middle_channels[3], self.middle_channels[4], self.kernel_sizes[4], padding = 'same')
        self.pool5 = nn.MaxPool2d(self.pooling_sizes[4], ceil_mode=True)
        self.activation = nn.ReLU()
        
    
        self.interp1 = nn.Upsample(size=(w[4], h[4]), mode=self.interp_mode)
        self.deconv1 = nn.Conv2d(self.middle_channels[4], self.middle_channels[3], self.kernel_sizes[4], padding='same')
        self.interp2 = nn.Upsample(size=(w[3], h[3]), mode=self.interp_mode)
        self.deconv2 = nn.Conv2d(self.middle_channels[3], self.middle_channels[2], self.kernel_sizes[3], padding='same')
        self.interp3 = nn.Upsample(size=(w[2], h[2]), mode=self.interp_mode)
        self.deconv3 = nn.Conv2d(self.middle_channels[2], self.middle_channels[1], self.kernel_sizes[2], padding='same')
        self.interp4 = nn.Upsample(size=(w[1], h[1]), mode=self.interp_mode)
        self.deconv4 = nn.Conv2d(self.middle_channels[1], self.middle_channels[0], self.kernel_sizes[1], padding='same')
        self.interp5 = nn.Upsample(size=(self.image_nrows, self.image_ncols), mode=self.interp_mode)
        self.deconv5 = nn.Conv2d(self.middle_channels[0], self.n_channels, self.kernel_sizes[0], padding='same')
        
    def _load_configurations(self, params):
        if params is not None:
                
            if self.image_nrows is None:
                self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            if self.image_ncols is None:
                self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            if self.n_channels is None:
                dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
                self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, None)
            if self.middle_channels is None:
                self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", None)
            if self.kernel_sizes is None:
                self.kernel_sizes = params[model_cathegory_string][self.model_name].get("kernel_sizes", None)
            
            if self.pooling_sizes is None:
                self.pooling_sizes = params[model_cathegory_string][self.model_name].get("pooling_sizes", None)
            if self.interp_mode is None:
                self.interp_mode = params[model_cathegory_string][self.model_name].get("interp_mode", None)
            
        
        for var in [self.n_channels, self.image_nrows, self.image_ncols, self.middle_channels, self.kernel_sizes, self.pooling_sizes, self.interp_mode]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")

    def _calculate_sizes(self):
        w = []
        h = []
        w.append(self.image_nrows)
        h.append(self.image_ncols)
        for i in range(1, 5):
            w.append(conv_output_size_same_padding(w[i-1], self.pooling_sizes[i-1]))
            h.append(conv_output_size_same_padding(h[i-1], self.pooling_sizes[i-1]))
        return w,h
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass

        Args:
            x (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), not containing NaNs

        Returns:
            th.Tensor: output image
        """
        enc1 = self.pool1(self.activation(self.conv1(x)))
        enc2 = self.pool2(self.activation(self.conv2(enc1)))
        enc3 = self.pool3(self.activation(self.conv3(enc2)))
        enc4 = self.pool4(self.activation(self.conv4(enc3)))
        enc5 = self.pool5(self.activation(self.conv5(enc4)))
        dec1 = self.activation(self.deconv1(self.interp1(enc5)))
        x = dec1 + enc4
        dec2 = self.activation(self.deconv2(self.interp2(x)))
        x = dec2 + enc3
        dec3 = self.activation(self.deconv3(self.interp3(x)))
        x = dec3 + enc2
        dec4 = self.activation(self.deconv4(self.interp4(x)))
        x = dec4 + enc1
        dec5 = self.activation(self.deconv5(self.interp5(x)))
        
        return dec5

class DINCAE_pconvs(nn.Module):
    def __init__(self, params, n_channels: int = None, image_nrows: int = None, image_ncols: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None):
        super(DINCAE_pconvs, self).__init__()
        
        self.model_name = "DINCAE_pconvs"
        
        self.n_channels = n_channels
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        
        self._load_configurations(params)
        
        self.output_size = self.n_channels
        
        self.n_layers = len(self.middle_channels)
        self.w, self.h = self._calculate_sizes()
        
        self.pconv1 = PartialConv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2)
        self.pool1 = nn.MaxPool2d(self.pooling_sizes[0], ceil_mode=True)
        
        self.pconv2 = PartialConv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_sizes[1], padding='same')
        self.pool2 = nn.MaxPool2d(self.pooling_sizes[1], ceil_mode=True)
        
        self.pconv3 = PartialConv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_sizes[2], padding='same')
        self.pool3 = nn.MaxPool2d(self.pooling_sizes[2], ceil_mode=True)
        
        self.pconv4 = PartialConv2d(self.middle_channels[2], self.middle_channels[3], kernel_size=self.kernel_sizes[3], padding='same')
        self.pool4 = nn.MaxPool2d(self.pooling_sizes[3], ceil_mode=True)
        
        self.pconv5 = PartialConv2d(self.middle_channels[3], self.middle_channels[4], kernel_size=self.kernel_sizes[4], padding='same')
        self.pool5 = nn.MaxPool2d(self.pooling_sizes[4], ceil_mode=True)
        
        self.activation = nn.ReLU()
        
    
        self.interp1 = nn.Upsample(size=(self.w[4], self.h[4]), mode=self.interp_mode)
        self.pdeconv1 = PartialConv2d(self.middle_channels[4], self.middle_channels[3], kernel_size=self.kernel_sizes[4], padding='same')
        
        self.interp2 = nn.Upsample(size=(self.w[3], self.h[3]), mode=self.interp_mode)
        self.pdeconv2 = PartialConv2d(self.middle_channels[3], self.middle_channels[2], kernel_size=self.kernel_sizes[3], padding='same')
        
        self.interp3 = nn.Upsample(size=(self.w[2], self.h[2]), mode=self.interp_mode)
        self.pdeconv3 = PartialConv2d(self.middle_channels[2], self.middle_channels[1], kernel_size=self.kernel_sizes[2], padding='same')
        
        self.interp4 = nn.Upsample(size=(self.w[1], self.h[1]), mode=self.interp_mode)
        self.pdeconv4 = PartialConv2d(self.middle_channels[1], self.middle_channels[0], kernel_size=self.kernel_sizes[1], padding='same')
        
        self.interp5 = nn.Upsample(size=(self.image_nrows, self.image_ncols), mode=self.interp_mode)
        self.pdeconv5 = PartialConv2d(self.middle_channels[0], self.n_channels, kernel_size=self.kernel_sizes[0], padding='same')

    def _load_configurations(self, params):
        if params is not None:
                
            if self.image_nrows is None:
                self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            if self.image_ncols is None:
                self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            if self.n_channels is None:
                dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
                self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, None)
            if self.middle_channels is None:
                self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", None)
            if self.kernel_sizes is None:
                self.kernel_sizes = params[model_cathegory_string][self.model_name].get("kernel_sizes", None)
            
            if self.pooling_sizes is None:
                self.pooling_sizes = params[model_cathegory_string][self.model_name].get("pooling_sizes", None)
            if self.interp_mode is None:
                self.interp_mode = params[model_cathegory_string][self.model_name].get("interp_mode", None)
            
        
        for var in [self.n_channels, self.image_nrows, self.image_ncols, self.middle_channels, self.kernel_sizes, self.pooling_sizes, self.interp_mode]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
        
    def _calculate_sizes(self):
        """Calculate the output sizes of the convolutions and the downsampling and upsampling layers."""
        w = []
        h = []
        w.append(self.image_nrows)
        h.append(self.image_ncols)
        for i in range(1, self.n_layers):
            w.append(conv_output_size_same_padding(w[i - 1], self.pooling_sizes[i - 1]))
            h.append(conv_output_size_same_padding(h[i - 1], self.pooling_sizes[i - 1]))
        return w, h
    
    def forward(self, x: th.Tensor, mask: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass

        Args:
            x (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), can contain NaNs
            mask (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), 1 where x is valid, 0 where x is masked

        Returns:
            th.Tensor: output image and mask
        """
        x1, mask1 = self.pconv1(x, mask)
        x1 = self.activation(x1)
        x1 = self.pool1(x1)
        mask1 = self.pool1(mask1)
        
        x2, mask2 = self.pconv2(x1, mask1)
        x2 = self.activation(x2)
        x2 = self.pool2(x2)
        mask2 = self.pool2(mask2)
        
        x3, mask3 = self.pconv3(x2, mask2)
        x3 = self.activation(x3)
        x3 = self.pool3(x3)
        mask3 = self.pool3(mask3)
        
        x4, mask4 = self.pconv4(x3, mask3)
        x4 = self.activation(x4)
        x4 = self.pool4(x4)
        mask4 = self.pool4(mask4)
        
        x5, mask5 = self.pconv5(x4, mask4)
        x5 = self.activation(x5)
        x5 = self.pool5(x5)
        mask5 = self.pool5(mask5)
        
        dec1, dmask1= self.interp1(x5), self.interp1(mask5)
        dec1, dmask1 = self.pdeconv1(dec1, dmask1)
        dec1 = self.activation(dec1)
        x = dec1 + x4
        mask = dmask1 + mask4
        
        dec2, dmask2 = self.interp2(x), self.interp2(mask)
        dec2, dmask2 = self.pdeconv2(dec2, dmask2)
        dec2 = self.activation(dec2)
        x = dec2 + x3
        mask = dmask2 + mask3
        
        dec3, dmask3 = self.interp3(x), self.interp3(mask)
        dec3, dmask3 = self.pdeconv3(dec3, dmask3)
        dec3 = self.activation(dec3)
        x = dec3 + x2
        mask = dmask3 + mask2
        
        dec4, dmask4 = self.interp4(x), self.interp4(mask)
        dec4, dmask4 = self.pdeconv4(dec4, dmask4)
        dec4 = self.activation(dec4)
        x = dec4 + x1
        mask = dmask4 + mask1
        
        dec5, dmask5 = self.interp5(x), self.interp5(mask)
        dec5, dmask5 = self.pdeconv5(dec5, dmask5)
        dec5 = self.activation(dec5)
        
        self.output_mask = dmask5
        
        return dec5

class simple_conv(nn.Module):
    def __init__(self, params, n_channels: int = None, middle_channels: List[int] = None, kernel_size: List[int] = None, stride: List[int] = None, padding: List[int] = None, output_padding: List[int] = None):
        super(simple_conv, self).__init__()
        
        self.model_name: str = "simple_conv"
        self.n_channels = n_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        self._load_configurations(params)
        
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[0]),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1]),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[2]),  # 16x16 -> 8x8
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.middle_channels[2], self.middle_channels[1], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[0], output_padding=self.output_padding[0]),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(self.middle_channels[1], self.middle_channels[0], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1], output_padding=self.output_padding[1]),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(self.middle_channels[0], self.n_channels, kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[2], output_padding=self.output_padding[2]),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def _load_configurations(self, params):
        if params is not None:
    
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", "temperature")
            self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, 3)
            
            self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", [10, 10, 10])
            self.kernel_size = params[model_cathegory_string][self.model_name].get("kernel_size", [2, 2, 2])
            self.stride = params[model_cathegory_string][self.model_name].get("stride", [2, 2, 2])
            self.padding = params[model_cathegory_string][self.model_name].get("padding", [1, 1, 1])
            self.output_padding = params[model_cathegory_string][self.model_name].get("output_padding", [1, 1, 1])
        
        for var in [self.n_channels, self.middle_channels, self.kernel_size, self.stride, self.padding, self.output_padding]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")

    def forward(self, x: th.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class conv_unet(nn.Module):
    def __init__(self, params, n_channels: int = None, middle_channels: List[int] = None, kernel_size: List[int] = None, stride: List[int] = None, padding: List[int] = None, output_padding: List[int] = None):
        super(conv_unet, self).__init__()
        
        self.model_name: str = "conv_unet"
        self.n_channels = n_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        self._load_configurations(params)
        
        self.encoder1 = nn.Conv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[0])  # 64x64 -> 32x32
        self.encoder2 = nn.Conv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1])  # 32x32 -> 16x16
        self.encoder3 = nn.Conv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[2])  # 16x16 -> 8x8
        
        self.decoder1 = nn.ConvTranspose2d(self.middle_channels[2], self.middle_channels[1], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[0], output_padding=self.output_padding[0])  # 8x8 -> 16x16
        self.decoder2 = nn.ConvTranspose2d(self.middle_channels[1], self.middle_channels[0], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1], output_padding=self.output_padding[1])
        self.decoder3 = nn.ConvTranspose2d(self.middle_channels[0], self.n_channels, kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[2], output_padding=self.output_padding[2])
        
        self.relu = nn.ReLU()

    def _load_configurations(self, params):
        if params is not None:
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", "temperature")
            self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, 3)
            
            self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", [10, 10, 10])
            self.kernel_size = params[model_cathegory_string][self.model_name].get("kernel_size", [7, 7, 7])
            self.stride = params[model_cathegory_string][self.model_name].get("stride", [11, 11, 11])
            self.padding = params[model_cathegory_string][self.model_name].get("padding", [3, 3, 3])
            self.output_padding = params[model_cathegory_string][self.model_name].get("output_padding", [4, 4, 4])
        
        for var in [self.n_channels, self.middle_channels, self.kernel_size, self.stride, self.padding, self.output_padding]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")

    def forward(self, x: th.Tensor):
        enc1 = self.relu(self.encoder1(x))
        enc2 = self.relu(self.encoder2(enc1))
        enc3 = self.relu(self.encoder3(enc2))
        dec1 = self.relu(self.decoder1(enc3))
        
        dec2 = self.relu(self.decoder2(dec1 + enc2))
        dec3 = nn.Sigmoid()(self.decoder3(dec2 + enc1))
        
        return dec3

class conv_maxpool(nn.Module):
    def __init__(self, params, n_channels: int = None, middle_channels: list = None, kernel_size: int = None, stride: int = None, pool_size: int = None, up_kernel: int = None, up_stride: int = None, print_sizes: bool = None):
        super(conv_maxpool, self).__init__()
        
        self.model_name: str = "conv_maxpool"
        self.n_channels = n_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.up_kernel = up_kernel
        self.up_stride = up_stride
        self.print_sizes = print_sizes
        
        self._load_configurations(params)
        
        assert self.kernel_size % 2 == 1, "Kernel size must be an odd number"
        
        # Parameters
        activation = nn.ReLU()
        
        # Define encoder
        self.encoder_blocks = nn.ModuleList([
            self._create_conv_block(self.n_channels, self.middle_channels[0], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[0], self.middle_channels[1], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[1], self.middle_channels[2], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[2], self.middle_channels[3], self.kernel_size, self.stride, activation)
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(self.pool_size),
            nn.MaxPool2d(self.pool_size),
            nn.MaxPool2d(self.pool_size),
            nn.MaxPool2d(self.pool_size)
        ])
        
        # Define decoder
        self.decoder_blocks = nn.ModuleList([
            self._create_conv_block(self.middle_channels[3], self.middle_channels[4], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[4], self.middle_channels[3], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[3], self.middle_channels[2], self.kernel_size, self.stride, activation),
            self._create_conv_block(self.middle_channels[2], self.middle_channels[1], self.kernel_size, self.stride, activation)
        ])
        
        padding = (self.up_kernel - 1) // 2
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(self.middle_channels[4], self.middle_channels[3], self.up_kernel, self.up_stride, padding=padding),
            nn.ConvTranspose2d(self.middle_channels[3], self.middle_channels[2], self.up_kernel, self.up_stride, padding=padding),
            nn.ConvTranspose2d(self.middle_channels[2], self.middle_channels[1], self.up_kernel, self.up_stride, padding=padding),
            nn.ConvTranspose2d(self.middle_channels[1], self.middle_channels[0], self.up_kernel, self.up_stride, padding=padding)
        ])
        
        # Output layer
        self.output_conv = nn.Conv2d(self.middle_channels[1], self.n_channels, self.kernel_size, self.stride, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def _load_configurations(self, params):
        if params is not None:
            
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", "temperature")
            self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, 3)
            
            self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", [12, 12, 12, 12, 12])
            self.kernel_size = params[model_cathegory_string][self.model_name].get("kernel_size", 5)
            self.stride = params[model_cathegory_string][self.model_name].get("stride", 5)
            self.pool_size = params[model_cathegory_string][self.model_name].get("pool_size", 5)
            self.up_kernel = params[model_cathegory_string][self.model_name].get("up_kernel", 5)
            self.up_stride = params[model_cathegory_string][self.model_name].get("up_stride", 5)
            self.print_sizes = params[model_cathegory_string][self.model_name].get("print_sizes", False)
        
        for var in [self.n_channels, self.middle_channels, self.kernel_size, self.stride, self.pool_size, self.up_kernel, self.up_stride]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
    
    def _create_conv_block(self, n_channels, out_channels, kernel_size, stride, activation):
        """Helper method to create a convolutional block."""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation,
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        
        encodings = []
        
        if self.print_sizes:
            print(f"Input: {x.shape}", flush=True)
        
        # Encoder
        for conv, pool in zip(self.encoder_blocks, self.pools):
            x = conv(x)
            encodings.append(x)
            x = pool(x)
            if self.print_sizes:
                print(f"Encoder block: {x.shape}", flush=True)
        
        # Decoder
        for i, (deconv, upconv) in enumerate(zip(self.decoder_blocks, self.upconvs)):
            x = deconv(x)
            x = upconv(x)
            x = th.cat([x, encodings[-(i + 1)]], dim=1)
            if self.print_sizes:
                print(f"Decoder block {i + 1}: {x.shape}", flush=True)
        
        # Output
        x = self.output_conv(x)
        if self.print_sizes:
            print(f"Output: {x.shape}", flush=True)
        
        return self.sigmoid(x)