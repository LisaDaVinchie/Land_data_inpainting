import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)
        # Inherit nn.Conv2d parameters
        
        # Initialize mask updater and total number of elements in convolution kernel
        self.weight_maskUpdater = th.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
        
    
    def forward(self, x: th.Tensor, mask = None):
        if mask is None:
            mask = th.ones_like(x)
            
        with th.no_grad():
            # Make sure that weight_maskUpdater is the same type as x
            if self.weight_maskUpdater.type() != x.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(x)
            
            # Calculate the number of contributing pixels for each sliding window
            self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            
            # Calculate normalization factor
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
            self.update_mask = th.clamp(self.update_mask, 0, 1)
            self.mask_ratio = th.mul(self.mask_ratio, self.update_mask)
        
        # Perform convolution on masked input
        raw_out = super(PartialConv2d, self).forward(th.mul(x, mask))
        
        # Normalize the output
        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = th.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = th.mul(output, self.update_mask)
        else:
            output = th.mul(raw_out, self.mask_ratio)
        
        return output, self.update_mask