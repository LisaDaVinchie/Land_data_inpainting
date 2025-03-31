import torch as th
import torch.nn as nn
from pathlib import Path
from typing import List
import math
from src.utils.import_params_json import load_config
from PartialConv import PartialConv2d
from PartialPool import PartialMaxPool2D

n_channels_string = "n_channels"
image_nrows_string = "cutted_nrows"
image_ncols_string = "cutted_ncols"

def conv_output_size_same_padding(in_size, pool_size):
    return  math.ceil(in_size / pool_size)

def conv_output_size(in_size, kernel, padding, stride):
    return  math.ceil((in_size - kernel + 2 * padding) / stride + 1)

class DINCAE_pconvs(nn.Module):
    def __init__(self, params_path: Path, n_channels: int = None, image_nrows: int = None, image_ncols: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None, output_size: int = None):
        super(DINCAE_pconvs, self).__init__()
        
        model_params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.n_channels = n_channels if n_channels is not None else model_params.get(n_channels_string, 3)
        self.image_nrows = image_nrows if image_nrows is not None else model_params.get(image_nrows_string, 64)
        self.image_ncols = image_ncols if image_ncols is not None else model_params.get(image_ncols_string, 64)
        
        model_params = load_config(params_path, ["DINCAE_pconvs"]).get("DINCAE_pconvs", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", [10, 10, 10, 10, 10])
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else model_params.get("kernel_sizes", [2, 2, 2, 2, 2])
        self.pooling_sizes = pooling_sizes if pooling_sizes is not None else model_params.get("pooling_sizes", [7, 7, 7, 7, 7])
        self.interp_mode = interp_mode if interp_mode is not None else model_params.get("interp_mode", "bilinear")
        self.output_size = self.n_channels
        
        self.n_layers = len(self.middle_channels)
        self.w, self.h = self._calculate_sizes()
        
        self.pconv1 = PartialConv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2)
        self.pool1 = PartialMaxPool2D(self.pooling_sizes[0], ceil_mode=True)
        
        self.pconv2 = PartialConv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_sizes[1], padding='same')
        self.pool2 = PartialMaxPool2D(self.pooling_sizes[1], ceil_mode=True)
        
        self.pconv3 = PartialConv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_sizes[2], padding='same')
        self.pool3 = PartialMaxPool2D(self.pooling_sizes[2], ceil_mode=True)
        
        self.pconv4 = PartialConv2d(self.middle_channels[2], self.middle_channels[3], kernel_size=self.kernel_sizes[3], padding='same')
        self.pool4 = PartialMaxPool2D(self.pooling_sizes[3], ceil_mode=True)
        
        self.pconv5 = PartialConv2d(self.middle_channels[3], self.middle_channels[4], kernel_size=self.kernel_sizes[4], padding='same')
        self.pool5 = PartialMaxPool2D(self.pooling_sizes[4], ceil_mode=True)
        
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
        self.pdeconv5 = PartialConv2d(self.middle_channels[0], self.output_size, kernel_size=self.kernel_sizes[0], padding='same')
        
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
        x1, mask1 = self.pool1(x1, mask1)
        
        x2, mask2 = self.pconv2(x1, mask1)
        x2 = self.activation(x2)
        x2, mask2 = self.pool2(x2, mask2)
        
        x3, mask3 = self.pconv3(x2, mask2)
        x3 = self.activation(x3)
        x3, mask3 = self.pool3(x3, mask3)
        
        x4, mask4 = self.pconv4(x3, mask3)
        x4 = self.activation(x4)
        x4, mask4 = self.pool4(x4, mask4)
        
        x5, mask5 = self.pconv5(x4, mask4)
        x5 = self.activation(x5)
        x5, mask5 = self.pool5(x5, mask5)
        
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
        
        return dec5, dmask5