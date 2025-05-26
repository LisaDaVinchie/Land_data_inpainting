import torch as th
from torch import nn
from typing import List
import math
from PartialConv import PartialConv2d
from preprocessing.mask_data import mask_inversemask_image

def conv_output_size_same_padding(in_size, pool_size):
    return  math.ceil(in_size / pool_size)

def conv_output_size(in_size, kernel, padding, stride):
    return  math.ceil((in_size - kernel + 2 * padding) / stride + 1)


n_channels_string = "n_channels"
image_nrows_string = "cutted_nrows"
image_ncols_string = "cutted_ncols"

model_cathegory_string: str = "models"
dataset_cathegory_string: str = "dataset"

def initialize_model_and_dataset_kind(params, model_kind: str, dataset_params = None) -> tuple[nn.Module, str]:
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
    elif model_kind == "DINCAE_like":
        model = DINCAE_like(params)
        dataset_kind = "extended"
    elif model_kind == "DINCAE_pconvs":
        model = DINCAE_pconvs(params)
        dataset_kind = "minimal"
    elif model_kind == "DINCAE_pconvs_1":
        model = DINCAE_pconvs_1(params = params)
        dataset_kind = "minimal"
    elif model_kind == "dummy":
        model = DummyModel()
        dataset_kind = "minimal"
    elif model_kind == "dummier":
        model = DummierModel()
        dataset_kind = "minimal"
    else:
        raise ValueError(f"Model kind {model_kind} not recognized")
    
    if dataset_params is not None:
        model.override_load_dataset_configurations(dataset_params)
    model.n_channels = 13
    
    model.layers_setup()
    
    return model, dataset_kind

class DummierModel(nn.Module):
    def __init__(self, params = None, n_channels: int = 13, total_days: int = 9):
        """Dummy model for testing purposes. Returns the mean of the previous and following days.

        Args:
            params (_type_, optional): _description_. Defaults to None.
            n_channels (int, optional): _description_. Defaults to 13.
            total_days (int, optional): _description_. Defaults to 9.
        """
        super(DummierModel, self).__init__()
        
        self.n_channels = n_channels
        self.total_days = total_days
    
    def forward(self, images: th.Tensor, masks: th.Tensor) -> th.Tensor:
        
        B, _, H, W = images.shape
        
        return th.ones((B, 2, H, W), dtype=th.float32, requires_grad = True) * 4
    
    def layers_setup(self):
        """Dummy method to satisfy the interface."""
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        
    
    def override_load_dataset_configurations(self, params):
        pass

class DummyModel(nn.Module):
    def __init__(self, params = None, n_channels: int = 13, total_days: int = 9):
        """Dummy model for testing purposes. Returns the mean of the previous and following days.

        Args:
            params (_type_, optional): _description_. Defaults to None.
            n_channels (int, optional): _description_. Defaults to 13.
            total_days (int, optional): _description_. Defaults to 9.
        """
        super(DummyModel, self).__init__()
        
        self.n_channels = n_channels
        self.total_days = total_days
    
    def forward(self, images: th.Tensor, masks: th.Tensor) -> th.Tensor:
        c = self.total_days // 2

        # Select all the channels of the SST except the one to predict
        known_channels = range(self.total_days)
        known_channels = [i for i in known_channels if i != c]
        
        known_images = images[:, known_channels, :, :]
        
        mean_image = known_images.mean(dim=1, keepdim=True)
        
        return mean_image.requires_grad_(True)
    
    def layers_setup(self):
        """Dummy method to satisfy the interface."""
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        
    
    def override_load_dataset_configurations(self, params):
        pass

class DINCAE_pconvs_1(nn.Module):
    def __init__(self, params, n_channels: int = None, image_nrows: int = None, image_ncols: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None):
        super(DINCAE_pconvs_1, self).__init__()
        
        self.model_name = "DINCAE_pconvs"
        
        self.n_channels = n_channels
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        self.output_channels = 2
                
        self._load_model_configurations(params)

    def layers_setup(self):
        self.w, self.h = self._calculate_sizes()
        
        self.pconv1 = PartialConv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2)
        self.pool1 = nn.MaxPool2d(self.pooling_sizes[0], stride=2, ceil_mode=True)
        
        self.pconv2 = PartialConv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_sizes[1], padding='same')
        self.pool2 = nn.MaxPool2d(self.pooling_sizes[1], stride=2, ceil_mode=True)
        
        self.pconv3 = PartialConv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_sizes[2], padding='same')
        self.pool3 = nn.MaxPool2d(self.pooling_sizes[2], stride=2, ceil_mode=True)
        
        self.pconv4 = PartialConv2d(self.middle_channels[2], self.middle_channels[3], kernel_size=self.kernel_sizes[3], padding='same')
        self.pool4 = nn.MaxPool2d(self.pooling_sizes[3], stride=2, ceil_mode=True)
        
        self.pconv5 = PartialConv2d(self.middle_channels[3], self.middle_channels[4], kernel_size=self.kernel_sizes[4], padding='same')
        self.pool5 = nn.MaxPool2d(self.pooling_sizes[4], stride=2, ceil_mode=True)
        
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
        self.pdeconv5 = PartialConv2d(self.middle_channels[0], self.output_channels, kernel_size=self.kernel_sizes[0], padding='same')
        
        self.print = True
        
    def print_shapes(self, layer_name, x):
        if self.print:
            print(f"{layer_name} shape: ", x.shape)

    def _load_model_configurations(self, params):
        if params is not None:
            model_params = params[model_cathegory_string].get(self.model_name, None)
            if self.middle_channels is None:
                self.middle_channels = model_params.get("middle_channels", None)
            if self.kernel_sizes is None:
                self.kernel_sizes = model_params.get("kernel_sizes", None)
            if self.pooling_sizes is None:
                self.pooling_sizes = model_params.get("pooling_sizes", None)
            if self.interp_mode is None:
                self.interp_mode = model_params.get("interp_mode", None)
            
        for var in [self.middle_channels, self.kernel_sizes, self.pooling_sizes, self.interp_mode]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
    
    def _load_dataset_configurations(self, params):
        if params is not None:
            dataset_params = params[dataset_cathegory_string]
            if self.image_nrows is None:
                self.image_nrows = dataset_params.get(image_nrows_string, None)
            if self.image_ncols is None:
                self.image_ncols = dataset_params.get(image_ncols_string, None)
            if self.n_channels is None:
                dataset_kind = dataset_params.get("dataset_kind", None)
                channels_to_keep = dataset_params[dataset_kind].get("channels_to_keep", None)
                self.n_channels = len(channels_to_keep) + 1 + 8
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")
    
    def override_load_dataset_configurations(self, params):
        if params is not None:
            self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
            self.n_channels = len(channels_to_keep)
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")
        
    def _calculate_sizes(self):
        """Calculate the output sizes of the convolutions and the downsampling and upsampling layers."""
        w = []
        h = []
        w.append(self.image_nrows)
        h.append(self.image_ncols)
        for i in range(1, len(self.middle_channels)):
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
        
        
        self.print = False
        self.print_shapes("input", x)
        x1, mask1 = self.pconv1(x, mask)
        self.print_shapes("pconv1", x1)
        x1 = self.activation(x1)
        x1 = self.pool1(x1)
        self.print_shapes("pool1", x1)
        mask1 = self.pool1(mask1)
        
        x2, mask2 = self.pconv2(x1, mask1)
        self.print_shapes("pconv2", x2)
        x2 = self.activation(x2)
        x2 = self.pool2(x2)
        self.print_shapes("pool2", x2)
        mask2 = self.pool2(mask2)
        
        x3, mask3 = self.pconv3(x2, mask2)
        self.print_shapes("pconv3", x3)
        x3 = self.activation(x3)
        x3 = self.pool3(x3)
        self.print_shapes("pool3", x3)
        mask3 = self.pool3(mask3)
        
        x4, mask4 = self.pconv4(x3, mask3)
        self.print_shapes("pconv4", x4)
        x4 = self.activation(x4)
        x4 = self.pool4(x4)
        self.print_shapes("pool4", x4)
        mask4 = self.pool4(mask4)
        
        x5, mask5 = self.pconv5(x4, mask4)
        self.print_shapes("pconv5", x5)
        x5 = self.activation(x5)
        x5 = self.pool5(x5)
        mask5 = self.pool5(mask5)
        self.print_shapes("pool5", x5)
        
        dec1, dmask1= self.interp1(x5), self.interp1(mask5)
        self.print_shapes("interp1", dec1)
        dec1, dmask1 = self.pdeconv1(dec1, dmask1)
        self.print_shapes("pdeconv1", dec1)
        dec1 = self.activation(dec1)
        x = dec1 + x4
        
        mask = dmask1 + mask4
        
        dec2, dmask2 = self.interp2(x), self.interp2(mask)
        self.print_shapes("interp2", dec2)
        dec2, dmask2 = self.pdeconv2(dec2, dmask2)
        self.print_shapes("pdeconv2", dec2)
        dec2 = self.activation(dec2)
        x = dec2 + x3
        mask = dmask2 + mask3
        
        dec3, dmask3 = self.interp3(x), self.interp3(mask)
        self.print_shapes("interp3", dec3)
        dec3, dmask3 = self.pdeconv3(dec3, dmask3)
        self.print_shapes("pdeconv3", dec3)
        dec3 = self.activation(dec3)
        x = dec3 + x2
        mask = dmask3 + mask2
        
        dec4, dmask4 = self.interp4(x), self.interp4(mask)
        self.print_shapes("interp4", dec4)
        dec4, dmask4 = self.pdeconv4(dec4, dmask4)
        self.print_shapes("pdeconv4", dec4)
        dec4 = self.activation(dec4)
        x = dec4 + x1
        mask = dmask4 + mask1
        
        dec5, dmask5 = self.interp5(x), self.interp5(mask)
        self.print_shapes("interp5", dec5)
        dec5, dmask5 = self.pdeconv5(dec5, dmask5)
        self.print_shapes("pdeconv5", dec5)
        dec5 = self.activation(dec5)
        
        self.output_mask = dmask5
        
        
        return dec5

class DINCAE_like(nn.Module):
    def __init__(self, params, n_channels: int = None, image_nrows: int = None, image_ncols: int = None, middle_channels: List[int] = None, kernel_sizes: List[int] = None, pooling_sizes: List[int] = None, interp_mode: str = None, placeholder: float = None):
        super(DINCAE_like, self).__init__()
        
        self.model_name: str = "DINCAE_like"
        
        self.n_channels = n_channels
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        self.placeholder = placeholder
        
        self._load_model_configurations(params)
        
        self._load_dataset_configurations(params)
        
        if self.placeholder is None:
            self.placeholder = params["training"].get("placeholder", None)
        
        if self.placeholder is None:
            raise ValueError(f"Variable placeholder is None. Please provide a value for it.")

    def layers_setup(self):
        w, h = self._calculate_sizes()
        
        self.conv1 = nn.Conv2d(self.n_channels, self.middle_channels[0], self.kernel_sizes[0], padding = self.kernel_sizes[0] // 2)
        self.pool1 = nn.MaxPool2d(self.pooling_sizes[0], stride=2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(self.middle_channels[0], self.middle_channels[1], self.kernel_sizes[1], padding = 'same')
        self.pool2 = nn.MaxPool2d(self.pooling_sizes[1], stride=2, ceil_mode=True)
        
        self.conv3 = nn.Conv2d(self.middle_channels[1], self.middle_channels[2], self.kernel_sizes[2], padding = 'same')
        self.pool3 = nn.MaxPool2d(self.pooling_sizes[2], stride=2, ceil_mode=True)
        
        self.conv4 = nn.Conv2d(self.middle_channels[2], self.middle_channels[3], self.kernel_sizes[3], padding = 'same')
        self.pool4 = nn.MaxPool2d(self.pooling_sizes[3], stride=2, ceil_mode=True)
        
        self.conv5 = nn.Conv2d(self.middle_channels[3], self.middle_channels[4], self.kernel_sizes[4], padding = 'same')
        self.pool5 = nn.MaxPool2d(self.pooling_sizes[4], stride=2, ceil_mode=True)
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
        
    def _load_dataset_configurations(self, params):
        if params is not None:
            if self.image_nrows is None:
                self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            if self.image_ncols is None:
                self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            if self.n_channels is None:
                dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
                channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
                self.n_channels = len(channels_to_keep) + 1
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")
    
    def _load_model_configurations(self, params):
        if params is not None:
            model_params = params[model_cathegory_string].get(self.model_name, None)
            if self.middle_channels is None:
                self.middle_channels = model_params.get("middle_channels", None)
            if self.kernel_sizes is None:
                self.kernel_sizes = model_params.get("kernel_sizes", None)
            if self.pooling_sizes is None:
                self.pooling_sizes = model_params.get("pooling_sizes", None)
            if self.interp_mode is None:
                self.interp_mode = model_params.get("interp_mode", None)
            
        for var in [self.middle_channels, self.kernel_sizes, self.pooling_sizes, self.interp_mode]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
    
    def override_load_dataset_configurations(self, params):
        if params is not None:
            self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, None)
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")

    def _calculate_sizes(self):
        w = []
        h = []
        w.append(self.image_nrows)
        h.append(self.image_ncols)
        for i in range(1, len(self.middle_channels)):
            w.append(conv_output_size_same_padding(w[i-1], self.pooling_sizes[i-1]))
            h.append(conv_output_size_same_padding(h[i-1], self.pooling_sizes[i-1]))
        return w,h
        
    def forward(self, images: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Forward pass

        Args:
            images (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), not containing NaNs
            masks (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), 1 where x is valid, 0 where x is masked

        Returns:
            th.Tensor: output image
        """
        
        x, _ = mask_inversemask_image(images, masks, self.placeholder)
        
        enc1 = self.pool1(self.activation(self.conv1(x)))
        enc2 = self.pool2(self.activation(self.conv2(enc1)))
        enc3 = self.pool3(self.activation(self.conv3(enc2)))
        enc4 = self.pool4(self.activation(self.conv4(enc3)))
        enc5 = self.pool5(self.activation(self.conv5(enc4)))
        dec1 = self.activation(self.deconv1(self.interp1(enc5)))
        images = dec1 + enc4
        dec2 = self.activation(self.deconv2(self.interp2(images)))
        images = dec2 + enc3
        dec3 = self.activation(self.deconv3(self.interp3(images)))
        images = dec3 + enc2
        dec4 = self.activation(self.deconv4(self.interp4(images)))
        images = dec4 + enc1
        dec5 = self.activation(self.deconv5(self.interp5(images)))
        
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
        
        self._load_model_configurations(params)
        # self._load_dataset_configurations(params)

    def layers_setup(self):
        self.w, self.h = self._calculate_sizes()
        
        self.pconv1 = PartialConv2d(self.n_channels, self.middle_channels[0], kernel_size=self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2)
        self.pool1 = nn.MaxPool2d(self.pooling_sizes[0], stride=2, ceil_mode=True)
        
        self.pconv2 = PartialConv2d(self.middle_channels[0], self.middle_channels[1], kernel_size=self.kernel_sizes[1], padding='same')
        self.pool2 = nn.MaxPool2d(self.pooling_sizes[1], stride=2, ceil_mode=True)
        
        self.pconv3 = PartialConv2d(self.middle_channels[1], self.middle_channels[2], kernel_size=self.kernel_sizes[2], padding='same')
        self.pool3 = nn.MaxPool2d(self.pooling_sizes[2], stride=2, ceil_mode=True)
        
        self.pconv4 = PartialConv2d(self.middle_channels[2], self.middle_channels[3], kernel_size=self.kernel_sizes[3], padding='same')
        self.pool4 = nn.MaxPool2d(self.pooling_sizes[3], stride=2, ceil_mode=True)
        
        self.pconv5 = PartialConv2d(self.middle_channels[3], self.middle_channels[4], kernel_size=self.kernel_sizes[4], padding='same')
        self.pool5 = nn.MaxPool2d(self.pooling_sizes[4], stride=2, ceil_mode=True)
        
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

    def _load_model_configurations(self, params):
        if params is not None:
            model_params = params[model_cathegory_string].get(self.model_name, None)
            if self.middle_channels is None:
                self.middle_channels = model_params.get("middle_channels", None)
            if self.kernel_sizes is None:
                self.kernel_sizes = model_params.get("kernel_sizes", None)
            if self.pooling_sizes is None:
                self.pooling_sizes = model_params.get("pooling_sizes", None)
            if self.interp_mode is None:
                self.interp_mode = model_params.get("interp_mode", None)
            
        for var in [self.middle_channels, self.kernel_sizes, self.pooling_sizes, self.interp_mode]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
    
    def _load_dataset_configurations(self, params):
        if params is not None:
            dataset_params = params[dataset_cathegory_string]
            if self.image_nrows is None:
                self.image_nrows = dataset_params.get(image_nrows_string, None)
            if self.image_ncols is None:
                self.image_ncols = dataset_params.get(image_ncols_string, None)
            if self.n_channels is None:
                dataset_kind = dataset_params.get("dataset_kind", None)
                channels_to_keep = dataset_params[dataset_kind].get("channels_to_keep", None)
                self.n_channels = len(channels_to_keep) + 1
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")
    
    def override_load_dataset_configurations(self, params):
        if params is not None:
            self.image_nrows = params[dataset_cathegory_string].get(image_nrows_string, None)
            self.image_ncols = params[dataset_cathegory_string].get(image_ncols_string, None)
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
            self.n_channels = len(channels_to_keep) + 1
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        if self.image_nrows is None:
            raise ValueError(f"Variable image_nrows is None. Please provide a value for it.")
        if self.image_ncols is None:
            raise ValueError(f"Variable image_ncols is None. Please provide a value for it.")
        
    def _calculate_sizes(self):
        """Calculate the output sizes of the convolutions and the downsampling and upsampling layers."""
        w = []
        h = []
        w.append(self.image_nrows)
        h.append(self.image_ncols)
        for i in range(1, len(self.middle_channels)):
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
        
        self._load_model_configurations(params)
        
        self._load_dataset_configurations(params)
        
        self.encoder = None
        self.decoder = None  
    
    def layers_setup(self):
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
        
    def _load_model_configurations(self, params):
        if params is not None:
            self.middle_channels = params[model_cathegory_string][self.model_name].get("middle_channels", None)
            self.kernel_size = params[model_cathegory_string][self.model_name].get("kernel_size", None)
            self.stride = params[model_cathegory_string][self.model_name].get("stride", None)
            self.padding = params[model_cathegory_string][self.model_name].get("padding", None)
            self.output_padding = params[model_cathegory_string][self.model_name].get("output_padding", None)
        
        for var in [self.middle_channels, self.kernel_size, self.stride, self.padding, self.output_padding]:
            if var is None:
                raise ValueError(f"Variable {var} is None. Please provide a value for it.")
    
    def _load_dataset_configurations(self, params):
        if params is not None:
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
            self.n_channels = len(channels_to_keep) + 1
        
        if self.n_channels is None:
            raise ValueError(f"Variable {self.n_channels} is None. Please provide a value for it.")
    
    def override_load_dataset_configurations(self, params):
        self.n_channels = None
        if params is not None:
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
            self.n_channels = len(channels_to_keep) + 1
            
        if self.n_channels is None:
            raise ValueError(f"Variable {self.n_channels} is None. Please provide a value for it.")
        
    def forward(self, x: th.Tensor, masks: th.Tensor) -> th.Tensor:
        images, _ = mask_inversemask_image(x, masks, 0.0)
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)
        return decoded