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
        model = DINCAE_pconvs()
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

class DINCAE_pconvs(nn.Module):
    def __init__(self, params = None, n_channels: int = 13, image_nrows: int = 128, image_ncols: int = 128, middle_channels: List[int] = [16, 30, 58, 110, 209], kernel_sizes: List[int] = [3, 3, 3, 3, 3], pooling_sizes: List[int] = [2, 2, 2, 2, 2], interp_mode: str = "bilinear"):
        super(DINCAE_pconvs, self).__init__()
        
        self.model_name = "DINCAE_pconvs"
        
        self.n_channels = n_channels
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        
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
    
    def _load_dataset_configurations(self, params):
        if params is not None:
            dataset_params = params[dataset_cathegory_string]
            if self.n_channels is None:
                dataset_kind = dataset_params.get("dataset_kind", None)
                channels_to_keep = dataset_params[dataset_kind].get("channels_to_keep", None)
                self.n_channels = len(channels_to_keep) + 1 + 8
        if self.n_channels is None:
            raise ValueError(f"Variable n_channels is None. Please provide a value for it.")
    
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
            mask = mask.bool() & e_mask.bool()
        
        return x, mask.float()

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