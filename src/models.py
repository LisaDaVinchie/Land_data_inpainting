import torch as th
from torch import nn
from typing import List
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

def get_model_class(params, model_kind: str) -> tuple[nn.Module, str]:
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
    elif model_kind == "DINCAE_like":
        model = DINCAE_like()
    elif model_kind == "DINCAE_pconvs":
        model = DINCAE_pconvs()
    elif model_kind == "dummy":
        model = DummyModel()
    elif model_kind == "dummier":
        model = DummierModel()
    else:
        raise ValueError(f"Model kind {model_kind} not recognized")
    
    # if dataset_params is not None:
    #     model.override_load_dataset_configurations(dataset_params)
    
    return model

def mask_image(images, masks, placeholder):
    return th.where(masks.bool(), images, placeholder * th.ones_like(images))
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
        
        self.layers_setup()
    
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
        self.layers_setup()
        # Add a dummy parameter that does nothing but enables gradient clipping
    
    def forward(self, images: th.Tensor, masks: th.Tensor) -> th.Tensor:
        # c = self.total_days // 2

        # # Select all the channels of the SST except the one to predict
        # known_channels = range(self.total_days)
        # known_channels = [i for i in known_channels if i != c]
        self.dummy_param = nn.Parameter(th.zeros(1), requires_grad=True)
        
        c = 4
        known_channels = [0, 1, 2, 3, 5, 6, 7, 8]
        
        known_images = images[:, known_channels, :, :]
        
        known_images = th.where(masks[:, known_channels, :, :].bool(), known_images, th.nan)
        
        mean_image = th.nanmean(known_images, dim=1, keepdim=True)
        dummystd = th.zeros_like(mean_image)
        mean_image = th.where(masks[:, c:c+1, :, :].bool(), images[:, c:c+1, :, :], mean_image)
        mean_image = th.cat([mean_image, dummystd], dim=1) * (1 + self.dummy_param)
        
        return mean_image
    
    def layers_setup(self):
        """Dummy method to satisfy the interface."""
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        
    
    def override_load_dataset_configurations(self, params):
        pass

class DINCAE_pconvs(nn.Module):
    def __init__(self, n_channels: int = 13, middle_channels: List[int] = [16, 30, 58, 110, 209], kernel_sizes: List[int] = [3, 3, 3, 3, 3], pooling_sizes: List[int] = [2, 2, 2, 2, 2], interp_mode: str = "bilinear"):
        super(DINCAE_pconvs, self).__init__()
        
        self.model_name = "DINCAE"
        
        self.n_channels = n_channels
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        self.output_channels = 2
       
        self.print = False
        
        self.layers_setup()

    def layers_setup(self):
        
        self.enc1 = EncoderBlock(self.n_channels, self.middle_channels[0], self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2, pooling_size=self.pooling_sizes[0])
        self.enc2 = EncoderBlock(self.middle_channels[0], self.middle_channels[1], self.kernel_sizes[1], padding='same', pooling_size=self.pooling_sizes[1])
        self.enc3 = EncoderBlock(self.middle_channels[1], self.middle_channels[2], self.kernel_sizes[2], padding='same', pooling_size=self.pooling_sizes[2])
        self.enc4 = EncoderBlock(self.middle_channels[2], self.middle_channels[3], self.kernel_sizes[3], padding='same', pooling_size=self.pooling_sizes[3])
        self.enc5 = EncoderBlock(self.middle_channels[3], self.middle_channels[4], self.kernel_sizes[4], padding='same', pooling_size=self.pooling_sizes[4])
        
        self.dec6 = DecoderBlock(self.middle_channels[4], self.middle_channels[3], self.interp_mode, kernel_size=self.kernel_sizes[4])
        self.dec7 = DecoderBlock(self.middle_channels[3], self.middle_channels[2], self.interp_mode, kernel_size=self.kernel_sizes[3])
        self.dec8 = DecoderBlock(self.middle_channels[2], self.middle_channels[1], self.interp_mode, kernel_size=self.kernel_sizes[2])
        self.dec9 = DecoderBlock(self.middle_channels[1], self.middle_channels[0], self.interp_mode, kernel_size=self.kernel_sizes[1])
        self.dec10 = DecoderBlock(self.middle_channels[0], self.output_channels, self.interp_mode, kernel_size=self.kernel_sizes[0])
                
    def print_shapes(self, layer_name, x):
        if self.print:
            print(f"{layer_name} shape: {x.shape}", flush=True)

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
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            channels_to_keep = params[dataset_cathegory_string][dataset_kind].get("channels_to_keep", None)
            self.n_channels = len(channels_to_keep)
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
    
    def forward(self, x: th.Tensor, mask: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass

        Args:
            x (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), can contain NaNs
            mask (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), 1 where x is valid, 0 where x is masked

        Returns:
            th.Tensor: output image and mask
        """
        
        self.print_shapes("input", x)
        x1, mask1 = self.enc1(x, mask)
        self.print_shapes("enc1", x1)
        x2, mask2 = self.enc2(x1, mask1)
        self.print_shapes("enc2", x2)
        x3, mask3 = self.enc3(x2, mask2)
        self.print_shapes("enc3", x3)
        x4, mask4 = self.enc4(x3, mask3)
        self.print_shapes("enc4", x4)
        x5, mask5 = self.enc5(x4, mask4)
        self.print_shapes("enc5", x5)
        
        dec6, dmask6 = self.dec6(x5, mask5, x4, mask4)
        self.print_shapes("dec6", dec6)
        dec7, dmask7 = self.dec7(dec6, dmask6, x3, mask3)
        self.print_shapes("dec7", dec7)
        dec8, dmask8 = self.dec8(dec7, dmask7, x2, mask2)
        self.print_shapes("dec8", dec8)
        dec9, dmask9 = self.dec9(dec8, dmask8, x1, mask1)
        self.print_shapes("dec9", dec9)
        dec10, dmask10 = self.dec10(dec9, dmask9)
        
        self.output_mask = dmask10
        
        return dec10

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pooling_size, stride = 2):
        super(EncoderBlock, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pooling_size, stride=stride, ceil_mode=True)
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor, mask: th.Tensor) -> th.Tensor:
        x, mask = self.pconv(x, mask)
        x = self.activation(x)
        x = self.pool(x)
        mask = self.pool(mask)
        return x, mask
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interp_mode, kernel_size = 3, padding = 'same'):
        super(DecoderBlock, self).__init__()
        self.interp_mode = interp_mode
        self.pdeconv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor, mask: th.Tensor, e_x: th.Tensor = None, e_mask: th.Tensor = None) -> th.Tensor:
        if e_x is not None and e_mask is not None:
            self.interp = nn.Upsample(size=(e_x.shape[2], e_x.shape[3]), mode=self.interp_mode)
        else:
            self.interp = nn.Upsample(scale_factor=2, mode=self.interp_mode)
        x = self.interp(x)
        mask = self.interp(mask)
        x, mask = self.pdeconv(x, mask)
        x = self.activation(x)
        if e_x is not None and e_mask is not None:
            x = x + e_x
            # mask = mask * e_mask
            mask = th.clamp(mask + e_mask, 0, 1)
            # mask = mask + e_mask
            
            # x = th.cat([x, e_x], dim=1)
            # mask = th.cat([mask, e_mask], dim=1)
            
        
        return x, mask.float()

class DINCAE_like(nn.Module):
    def __init__(self, params = None, n_channels: int = 13, placeholder: float = -2.0, middle_channels: List[int] = [16, 30, 58, 110, 209], kernel_sizes: List[int] = [3, 3, 3, 3, 3], pooling_sizes: List[int] = [2, 2, 2, 2, 2], interp_mode: str = "bilinear"):
        super(DINCAE_like, self).__init__()
        
        self.model_name: str = "DINCAE"
        
        self.n_channels = n_channels
        
        self.middle_channels = middle_channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.interp_mode = interp_mode
        self.placeholder = placeholder
        self.output_channels: int = 2
        self.print = False
        
        if params is not None:
            self._load_model_configurations(params)
        
        if params is not None:
            self.placeholder = params["training"]["placeholder"]
        
        self.layers_setup()

    def layers_setup(self):
        self.enc1 = EncoderBlockConv(self.n_channels, self.middle_channels[0], self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2, pooling_size=self.pooling_sizes[0])
        self.enc2 = EncoderBlockConv(self.middle_channels[0], self.middle_channels[1], self.kernel_sizes[1], padding='same', pooling_size=self.pooling_sizes[1])
        self.enc3 = EncoderBlockConv(self.middle_channels[1], self.middle_channels[2], self.kernel_sizes[2], padding='same', pooling_size=self.pooling_sizes[2])
        self.enc4 = EncoderBlockConv(self.middle_channels[2], self.middle_channels[3], self.kernel_sizes[3], padding='same', pooling_size=self.pooling_sizes[3])
        self.enc5 = EncoderBlockConv(self.middle_channels[3], self.middle_channels[4], self.kernel_sizes[4], padding='same', pooling_size=self.pooling_sizes[4])
        
        self.dec6 = DecoderBlockConv(self.middle_channels[4], self.middle_channels[3], self.interp_mode, kernel_size=self.kernel_sizes[4])
        self.dec7 = DecoderBlockConv(self.middle_channels[3], self.middle_channels[2], self.interp_mode, kernel_size=self.kernel_sizes[3])
        self.dec8 = DecoderBlockConv(self.middle_channels[2], self.middle_channels[1], self.interp_mode, kernel_size=self.kernel_sizes[2])
        self.dec9 = DecoderBlockConv(self.middle_channels[1], self.middle_channels[0], self.interp_mode, kernel_size=self.kernel_sizes[1])
        self.dec10 = DecoderBlockConv(self.middle_channels[0], self.output_channels, self.interp_mode, kernel_size=self.kernel_sizes[0])
        
    def print_shapes(self, layer_name, x):
        if self.print:
            print(f"{layer_name} shape: {x.shape}", flush=True)
            
    def _load_model_configurations(self, params):
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
            dataset_kind = params[dataset_cathegory_string].get("dataset_kind", None)
            self.n_channels = params[dataset_cathegory_string][dataset_kind].get(n_channels_string, None)
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
        
    def forward(self, images: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Forward pass

        Args:
            images (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), not containing NaNs
            masks (th.Tensor): tensor of shape (batch_size, n_channels, image_nrows, image_ncols), 1 where x is valid, 0 where x is masked

        Returns:
            th.Tensor: output image
        """
        
        x = mask_image(images, masks, self.placeholder)
        self.print_shapes("input", x)
        x1 = self.enc1(x)
        self.print_shapes("enc1", x1)
        x2 = self.enc2(x1)
        self.print_shapes("enc2", x2)
        x3 = self.enc3(x2)
        self.print_shapes("enc3", x3)
        x4 = self.enc4(x3)
        self.print_shapes("enc4", x4)
        x5 = self.enc5(x4)
        self.print_shapes("enc5", x5)
        
        x6 = self.dec6(x5, x4)
        self.print_shapes("dec6", x6)
        x7 = self.dec7(x6, x3)
        self.print_shapes("dec7", x7)
        x8 = self.dec8(x7, x2)
        self.print_shapes("dec8", x8)
        x9 = self.dec9(x8, x1)
        self.print_shapes("dec9", x9)
        x10 = self.dec10(x9)
        self.print_shapes("dec10", x10)
        
        return x10
    
class EncoderBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pooling_size, stride = 2):
        super(EncoderBlockConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pooling_size, stride=stride, ceil_mode=True)
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x
    
class DecoderBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, interp_mode, kernel_size = 3, padding = 'same'):
        super(DecoderBlockConv, self).__init__()
        self.interp_mode = interp_mode
        self.deconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor, e_x: th.Tensor = None) -> th.Tensor:
        if e_x is not None:
            self.interp = nn.Upsample(size=(e_x.shape[2], e_x.shape[3]), mode=self.interp_mode)
        else:
            self.interp = nn.Upsample(scale_factor=2, mode=self.interp_mode)
        x = self.interp(x)
        x = self.deconv(x)
        x = self.activation(x)
        if e_x is not None:
            x = x + e_x
        
        return x

class simple_conv(nn.Module):
    def __init__(self, params = None, n_channels: int = None, placeholder = -300, middle_channels: List[int] = None, kernel_size: List[int] = None, stride: List[int] = None, padding: List[int] = None, output_padding: List[int] = None):
        super(simple_conv, self).__init__()
        
        self.model_name: str = "simple_conv"
        self.n_channels = n_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.placeholder = placeholder
        
        self._load_model_configurations(params)
        
        self._load_dataset_configurations(params)
        
        self.layers_setup()
        
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
        images = mask_image(x, masks, self.placeholder)
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)
        return decoded