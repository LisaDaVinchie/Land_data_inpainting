import torch as th
from torch import nn
from pathlib import Path
from typing import List
from utils.import_params_json import load_config
import math



def conv_output_size(in_size, pool_size):
    return  math.ceil(in_size / pool_size)
class DINCAE_like(nn.Module):
    def __init__(self, params_path: Path, n_channels: int = None, image_width: int = None, image_height: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None, output_size: int = None):
        super(DINCAE_like, self).__init__()
        
        model_params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 3)
        self.image_width = image_width if image_width is not None else model_params.get("image_width", 64)
        self.image_height = image_height if image_height is not None else model_params.get("image_height", 64)
        
        model_params = load_config(params_path, ["DINCAE_like"]).get("DINCAE_like", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", [10, 10, 10, 10, 10])
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else model_params.get("kernel_sizes", [2, 2, 2, 2, 2])
        self.pooling_sizes = pooling_sizes if pooling_sizes is not None else model_params.get("pooling_sizes", [7, 7, 7, 7, 7])
        self.interp_mode = interp_mode if interp_mode is not None else model_params.get("interp_mode", "bilinear")
        self.output_size = output_size if output_size is not None else model_params.get("output_size", 2)
        
        w = []
        h = []
        w.append(self.image_width)
        h.append(self.image_height)
        for i in range(1, 5):
            w.append(conv_output_size(w[i-1], self.pooling_sizes[i-1]))
            h.append(conv_output_size(h[i-1], self.pooling_sizes[i-1]))
        
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
        
    
        self.interp1 = nn.Upsample(size=(w[4], h[4]), mode='bilinear')
        self.deconv1 = nn.Conv2d(self.middle_channels[4], self.middle_channels[3], self.kernel_sizes[4], padding='same')
        self.interp2 = nn.Upsample(size=(w[3], h[3]), mode='bilinear')
        self.deconv2 = nn.Conv2d(self.middle_channels[3], self.middle_channels[2], self.kernel_sizes[3], padding='same')
        self.interp3 = nn.Upsample(size=(w[2], h[2]), mode='bilinear')
        self.deconv3 = nn.Conv2d(self.middle_channels[2], self.middle_channels[1], self.kernel_sizes[2], padding='same')
        self.interp4 = nn.Upsample(size=(w[1], h[1]), mode='bilinear')
        self.deconv4 = nn.Conv2d(self.middle_channels[1], self.middle_channels[0], self.kernel_sizes[1], padding='same')
        self.interp5 = nn.Upsample(size=(self.image_width, self.image_height), mode='bilinear')
        self.deconv5 = nn.Conv2d(self.middle_channels[0], self.output_size, self.kernel_sizes[0], padding='same')
        
    def forward(self, x: th.tensor) -> th.tensor:
        enc1 = self.pool1(self.activation(self.conv1(x)))
        enc2 = self.pool2(self.activation(self.conv2(enc1)))
        enc3 = self.pool3(self.activation(self.conv3(enc2)))
        enc4 = self.pool4(self.activation(self.conv4(enc3)))
        enc5 = self.pool5(self.activation(self.conv5(enc4)))
        dec1 = self.activation(self.deconv1(self.interp1(enc5)))
        dec1 += enc4
        dec2 = self.activation(self.deconv2(self.interp2(dec1)))
        dec2 += enc3
        dec3 = self.activation(self.deconv3(self.interp3(dec2)))
        dec3 += enc2
        dec4 = self.activation(self.deconv4(self.interp4(dec3)))
        dec4 += enc1
        dec5 = self.activation(self.deconv5(self.interp5(dec4)))
        
        return dec5
        
        
        
    
    
        
        
        

class simple_conv(nn.Module):
    def __init__(self, params_path: Path, n_channels: int = None, middle_channels: List[int] = None, kernel_size: List[int] = None, stride: List[int] = None, padding: List[int] = None, output_padding: List[int] = None):
        super(simple_conv, self).__init__()
        model_params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 3)
        
        model_params = load_config(params_path, ["simple_conv"]).get("simple_conv", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", [10, 10, 10])
        self.kernel_size = kernel_size if kernel_size is not None else model_params.get("kernel_size", [1, 1, 1])
        self.stride = stride if stride is not None else model_params.get("stride", [7, 7, 7])
        self.padding = padding if padding is not None else model_params.get("padding", [5, 5, 5])
        self.output_padding = output_padding if output_padding is not None else model_params.get("output_padding", [5, 5, 5])
        
        
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

    def forward(self, x: th.tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class conv_unet(nn.Module):
    def __init__(self, params_path: Path, n_channels: int = None, middle_channels: List[int] = None, kernel_size: List[int] = None, stride: List[int] = None, padding: List[int] = None, output_padding: List[int] = None):
        super(conv_unet, self).__init__()
        model_params = load_config(params_path, ["dataset_params"]).get("dataset_params", {})
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 1)
        
        model_params = load_config(params_path, ["conv_unet"]).get("conv_unet", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", [10, 10, 10])
        self.kernel_size = kernel_size if kernel_size is not None else model_params.get("kernel_size", [7, 7, 7])
        self.stride = stride if stride is not None else model_params.get("stride", [11, 11, 11])
        self.padding = padding if padding is not None else model_params.get("padding", [3, 3, 3])
        self.output_padding = output_padding if output_padding is not None else model_params.get("output_padding", [4, 4, 4])
        
        self.encoder1 = nn.Conv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[0])  # 64x64 -> 32x32
        self.encoder2 = nn.Conv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1])  # 32x32 -> 16x16
        self.encoder3 = nn.Conv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[2])  # 16x16 -> 8x8
        
        self.decoder1 = nn.ConvTranspose2d(self.middle_channels[2], self.middle_channels[1], kernel_size=self.kernel_size[2], stride=self.stride[2], padding=self.padding[0], output_padding=self.output_padding[0])  # 8x8 -> 16x16
        self.decoder2 = nn.ConvTranspose2d(self.middle_channels[1], self.middle_channels[0], kernel_size=self.kernel_size[1], stride=self.stride[1], padding=self.padding[1], output_padding=self.output_padding[1])
        self.decoder3 = nn.ConvTranspose2d(self.middle_channels[0], self.n_channels, kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[2], output_padding=self.output_padding[2])
        
        self.relu = nn.ReLU()

    def forward(self, x: th.tensor):
        enc1 = self.relu(self.encoder1(x))
        enc2 = self.relu(self.encoder2(enc1))
        enc3 = self.relu(self.encoder3(enc2))
        dec1 = self.relu(self.decoder1(enc3))
        
        dec2 = self.relu(self.decoder2(dec1 + enc2))
        dec3 = nn.Sigmoid()(self.decoder3(dec2 + enc1))
        
        return dec3

class conv_maxpool(nn.Module):
    def __init__(self, params_path: Path, n_channels: int = None, middle_channels: list = None, kernel_size: int = None, stride: int = None, pool_size: int = None, up_kernel: int = None, up_stride: int = None, print_sizes: bool = None):
        super(conv_maxpool, self).__init__()
        model_params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 1)
        
        model_params = load_config(params_path, ["conv_maxpool"]).get("conv_maxpool", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", [12, 12, 12, 12, 12])
        self.kernel_size = kernel_size if kernel_size is not None else model_params.get("kernel_size", 5)
        self.stride = stride if stride is not None else model_params.get("stride", 5)
        self.pool_size = pool_size if pool_size is not None else model_params.get("pool_size", 5)
        self.up_kernel = up_kernel if up_kernel is not None else model_params.get("up_kernel", 5)
        self.up_stride = up_stride if up_stride is not None else model_params.get("up_stride", 5)
        self.print_sizes = print_sizes if print_sizes is not None else model_params.get("print_sizes", False)
        
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
    
    def _create_conv_block(self, n_channels, out_channels, kernel_size, stride, activation):
        """Helper method to create a convolutional block."""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation,
        )
    
    def forward(self, x: th.tensor) -> th.Tensor:
        
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