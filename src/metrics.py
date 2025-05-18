import torch as th
import torch.nn.functional as F
from typing import Optional, Sequence, Union

class SSIM(th.nn.Module):
    """
    Computes Structural Similarity Index Measure

    - ``update`` must receive output of the form ``(y_pred, y)``. They have to be of the same type.
        Valid :class:`th.dtype` are the following:
        - on CPU: `th.float32`, `th.float64`.
        - on CUDA: `th.float16`, `th.bfloat16`, `th.float32`, `th.float64`.

    Args:
        data_range: Range of the image. Typically, ``1.0`` or ``255``.
        kernel_size: Size of the kernel. Default: 11
        sigma: Standard deviation of the gaussian kernel.
            Argument is used if ``gaussian=True``. Default: 1.5
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
        gaussian: ``True`` to use gaussian kernel, ``False`` to use uniform kernel
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.
        ndims: Number of dimensions of the input image: 2d or 3d. Accepted values: 2, 3. Default: 2

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        ``y_pred`` and ``y`` can be un-normalized or normalized image tensors. Depending on that, the user might need
        to adjust ``data_range``. ``y_pred`` and ``y`` should have the same shape.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SSIM(data_range=1.0)
            metric.attach(default_evaluator, 'ssim')
            preds = th.rand([4, 3, 16, 16])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['ssim'])

        .. testoutput::

            0.9218971...

    .. versionadded:: 0.4.2

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    .. versionchanged:: 0.5.2
        ``ndims`` argument is added.
    """

    def __init__(self, nan_placeholder: float, data_range: Union[int, float] = 1.0, kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian: bool = True,
        device: Union[str, th.device] = th.device("cpu")
    ):
        super(SSIM, self).__init__()
        
        ndims = 2
        self.ndims = ndims

        if isinstance(kernel_size, int):
            self.kernel_size: Sequence[int] = [kernel_size for _ in range(ndims)]
        elif isinstance(kernel_size, Sequence):
            if len(kernel_size) != ndims:
                raise ValueError(f"Expected kernel_size to have length of {ndims}. Got {len(kernel_size)}.")
            self.kernel_size = kernel_size
        else:
            raise ValueError(f"Argument kernel_size should be either int or a sequence of int of length {ndims}.")

        if isinstance(sigma, float):
            self.sigma: Sequence[float] = [sigma for _ in range(ndims)]
        elif isinstance(sigma, Sequence):
            if len(sigma) != ndims:
                raise ValueError(f"Expected sigma to have length of {ndims}. Got {len(sigma)}.")
            self.sigma = sigma
        else:
            raise ValueError(f"Argument sigma should be either float or a sequence of float of length {ndims}.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        self.gaussian = gaussian
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self.pad_d = None
        self._device = device
        self.nan_placeholder = nan_placeholder
        self._kernel_nd = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma, ndims=self.ndims)
        self._kernel: Optional[th.Tensor] = None
        
        self.reset()

    def reset(self) -> None:
        self._sum_of_ssim = th.tensor(0.0, dtype=th.float32, device=self._device)
        self._num_examples = 0

    def _uniform(self, kernel_size: int) -> th.Tensor:
        kernel = th.zeros(kernel_size, device=self._device)

        start_uniform_index = max(kernel_size // 2 - 2, 0)
        end_uniform_index = min(kernel_size // 2 + 3, kernel_size)

        min_, max_ = -2.5, 2.5
        kernel[start_uniform_index:end_uniform_index] = 1 / (max_ - min_)

        return kernel  # (kernel_size)

    def _gaussian(self, kernel_size: int, sigma: float) -> th.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = th.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        gauss = th.exp(-0.5 * (kernel / sigma).pow(2))
        return gauss / gauss.sum()  # (kernel_size)

    def _gaussian_or_uniform_kernel(
        self, kernel_size: Sequence[int], sigma: Sequence[float], ndims: int
    ) -> th.Tensor:
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        result = th.einsum("i,j->ij", kernel_x, kernel_y)
        
        return result

    def _check_type_and_shape(self, y_pred: th.Tensor, y: th.Tensor) -> None:
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        # 2 dimensions are reserved for batch and channel
        if len(y_pred.shape) - 2 != self.ndims or len(y.shape) - 2 != self.ndims:
            raise ValueError(
                "Expected y_pred and y to have BxCxHxW or BxCxDxHxW shape. "
                f"Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

    def forward(self, image: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:

        self._check_type_and_shape(image, target)
        
        valid_mask = self._get_valid_mask(target, masks)
        if valid_mask.sum() == 0:  # if all pixels are masked, return 0
            return th.zeros(1, device=image.device, requires_grad=True)

        self.nb_channel = image.size(1)
        if self._kernel is None or self._kernel.shape[0] != self.nb_channel:
            self._kernel = self._kernel_nd.expand(self.nb_channel, 1, *[-1 for _ in range(self.ndims)])


        image_masked = image * valid_mask.float()
        target_masked = target * valid_mask.float()
        
        image, target, valid_mask = self.apply_pad(valid_mask, image_masked, target_masked)

        if image.dtype != self._kernel.dtype:
            self._kernel = self._kernel.to(dtype=image.dtype)
        # norm = F.conv2d(valid_mask, self._kernel, groups=nb_channel).clamp(min=1e-8)
        
        self.norm = F.conv2d(valid_mask.float(), self._kernel, groups=self.nb_channel).clamp(min=1e-8)
        
        mu_pred_sq, mu_target_sq, mu_pred_target = self.compute_mu_components(image, target)

        sigma_pred_sq, sigma_target_sq, sigma_pred_target = self.compute_covariance_components(image, target, mu_pred_sq, mu_target_sq, mu_pred_target)
        

        ssim_idx = self.new_method(mu_pred_sq, mu_target_sq, mu_pred_target, sigma_pred_sq, sigma_target_sq, sigma_pred_target)

        # In case when ssim_idx can be MPS tensor and self._device is not MPS
        # self._double_dtype is float64.
        # As MPS does not support float64 we should set dtype to float32
        
        valid_mask = valid_mask[..., self.pad_h:-self.pad_h, self.pad_w:-self.pad_w]  # Crop padding
        ssim_score = ssim_idx[valid_mask.bool()].mean()
        
        return ssim_score.item()

    def compute_covariance_components(self, image, target, mu_pred_sq, mu_target_sq, mu_pred_target):
        sigma_pred_sq = F.conv2d(image * image, self._kernel, groups=self.nb_channel) / self.norm
        sigma_pred_sq = sigma_pred_sq - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self._kernel, groups=self.nb_channel) / self.norm
        sigma_target_sq = sigma_target_sq - mu_target_sq
        sigma_pred_target = F.conv2d(image * target, self._kernel, groups=self.nb_channel) / self.norm
        sigma_pred_target = sigma_pred_target - mu_pred_target
        return sigma_pred_sq,sigma_target_sq,sigma_pred_target

    def compute_mu_components(self, image, target):
        mu_pred = F.conv2d(image, self._kernel, groups=self.nb_channel)
        mu_pred = mu_pred / self.norm
        mu_target = F.conv2d(target, self._kernel, groups=self.nb_channel)
        mu_target = mu_target / self.norm
        
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target
        return mu_pred_sq,mu_target_sq,mu_pred_target

    def apply_pad(self, valid_mask, image_masked, target_masked):
        padding_shape = [self.pad_w, self.pad_w, self.pad_h, self.pad_h]
        image = F.pad(image_masked, padding_shape, mode="reflect")
        target = F.pad(target_masked, padding_shape, mode="reflect")
        padded_valid_mask = F.pad(valid_mask.float(), padding_shape, mode="reflect")
        return image, target, padded_valid_mask

    def new_method(self, mu_pred_sq, mu_target_sq, mu_pred_target, sigma_pred_sq, sigma_target_sq, sigma_pred_target):
        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        return ssim_idx

    def _get_nan_mask(self, target: th.Tensor) -> th.Tensor:
        """
        Computes the NaN mask for the tensors, True where the NaN values are.

        Args:
            target (th.Tensor): Second tensor, shape (B, C, H, W).

        Returns:
            th.Tensor: The NaN mask for the tensors.
        """
        
        return th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
    
    def _get_valid_mask(self, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """
        Computes the valid mask for the tensors, True for valid pixels.

        Args:
            target (th.Tensor): Second tensor, shape (B, C, H, W).
            masks (th.Tensor): th.bool mask with True for pixels to consider, shape (B, C, H, W).

        Returns:
            th.Tensor: The valid mask, True for valid pixels.
        """
        
        nan_mask = self._get_nan_mask(target)
        
        # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
        return ~(masks | nan_mask)

# class SSIM(th.nn.Module):
#     def __init__(self, nan_placeholder: float, kernel_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0):
#         super().__init__()
        
#         self.nan_placeholder = nan_placeholder
#         self.kernel_size = [kernel_size, kernel_size]
#         self.sigma = [sigma, sigma]
#         self.k1 = k1
#         self.k2 = k2
#         self.data_range = data_range
#         self.channel = 1
        
#         self.C1 = (self.k1 * self.data_range) ** 2
#         self.C2 = (self.k2 * self.data_range) ** 2
        
#         self.pad_h = (self.kernel_size[0] - 1) // 2
#         self.pad_w = (self.kernel_size[1] - 1) // 2
        
#         kernel_x = self._gaussian_kernel(self.kernel_size[0], self.sigma[0])
#         kernel_y = self._gaussian_kernel(self.kernel_size[1], self.sigma[1])
#         kernel = th.einsum("i,j->ij", kernel_x, kernel_y)
#         self.register_buffer("kernel", kernel)
        
#     # def _gaussian_kernel(self, size: int, sigma: float) -> th.Tensor:
#     #     coords = th.arange(size, dtype=th.float)
#     #     coords -= size // 2
#     #     g = coords ** 2
#     #     g = (-g / (2 * sigma ** 2)).exp()
#     #     g /= g.sum()
#     #     g = g.reshape(1, 1, size, 1) * g.reshape(1, 1, 1, size)  # Outer product
#     #     return g  # Shape: [1, 1, size, size]
    
#     def _gaussian_kernel(self, kernel_size: int, sigma: float) -> th.Tensor:
#         ksize_half = (kernel_size - 1) * 0.5
#         kernel = th.linspace(-ksize_half, ksize_half, steps=kernel_size)
#         gauss = th.exp(-0.5 * (kernel / sigma).pow(2))
#         return gauss / gauss.sum()  # (kernel_size)
        
        
#     def forward(self, image: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
#         """Calculates the SSIM between two tensors, only on the masked pixels.

#         Args:
#             image (th.Tensor): image tensor, shape (C, H, W).
#             target (th.Tensor): Second tensor, shape (C, H, W).
#             masks (th.Tensor): bool mask with False for masked (= to consider) pixels, shape (C, H, W).

#         Returns:
#             th.Tensor: The SSIM between the two tensors.
#         """
#         nan_mask = self._get_nan_mask(target)
#         vaild_mask = self._get_valid_mask(masks, nan_mask)
#         if vaild_mask.sum() == 0: # if all pixels are masked, return 0
#             return th.zeros(1, device=image.device, requires_grad=True)
        
#         batch_size = image.shape[0]
#         self.channel = image.shape[1]
#         # Create a [C, 1, K, K] kernel for each channel
#         kernel = self.kernel.repeat(self.channel, 1, 1, 1)
        
#         padding_shape = [self.pad_w, self.pad_w, self.pad_h, self.pad_h]
#         # Pad the image and target tensors
#         image = F.pad(image, padding_shape, mode='replicate')
#         target = F.pad(target, padding_shape, mode='replicate')
        
#         input_list = [image, target, image * image, target * target, image * target]
        
#         # Set the non valid pixels to 0.0
#         # masked_image = image.masked_fill(~vaild_mask, 0.0)
#         # masked_target = target.masked_fill(~vaild_mask, 0.0)
        
#         outputs = F.conv2d(th.cat(input_list, dim=1), kernel, groups=self.channel)
#         output_list = [outputs[x * batch_size : (x + 1) * batch_size] for x in range(len(input_list))]
        
#         mu_pred_sq = output_list[0].pow(2)
#         mu_target_sq = output_list[1].pow(2)
#         mu_pred_target = output_list[0] * output_list[1]
        
#         sigma_pred_sq = output_list[2] - mu_pred_sq
#         sigma_target_sq = output_list[3] - mu_target_sq
#         sigma_pred_target = output_list[4] - mu_pred_target
        
#         a1 = 2 * mu_pred_target + self.c1
#         a2 = 2 * sigma_pred_target + self.c2
#         b1 = mu_pred_sq + mu_target_sq + self.c1
#         b2 = sigma_pred_sq + sigma_target_sq + self.c2

#         ssim_map = (a1 * a2) / (b1 * b2)
        
#         # mean from all dimensions except batch
#         self._sum_of_ssim += th.mean(ssim_map, list(range(1, 2 + 2))).sum()

#         self._num_examples += image.shape[0]
        
#         # meaningful_mask = F.conv2d(vaild_mask.float(), kernel, padding=self.kernel_size // 2, groups=self.channel)
        
        
        
#         return 1 - self._sum_of_ssim / self._num_examples

#     def _get_valid_mask(self, masks: th.Tensor, nan_mask: th.Tensor) -> th.Tensor:
#         """
#         Computes the valid mask for the tensors, True for valid pixels.

#         Args:
#             masks (th.Tensor): th.bool mask with False for pixels to consider, shape (B, C, H, W).
#             nan_mask (th.Tensor): Mask for NaN values, True where NaN is shape (B, C, H, W).

#         Returns:
#             th.Tensor: The valid mask, True for valid pixels.
#         """
        
#         # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
#         return ~(masks | nan_mask)
    
#     def _get_nan_mask(self, target: th.Tensor) -> th.Tensor:
#         """
#         Computes the NaN mask for the tensors, True where the NaN values are.

#         Args:
#             target (th.Tensor): Second tensor, shape (B, C, H, W).

#         Returns:
#             th.Tensor: The NaN mask for the tensors.
#         """
        
#         return th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
        
        
class PCC(th.nn.Module):
    """
    Computes the Pearson Correlation Coefficient (PCC) between two tensors.
    """

    def __init__(self, nan_placeholder: float):
        super().__init__()
        
        self.nan_placeholder = nan_placeholder

    def forward(self, image: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """
        Computes the PCC between two tensors.

        Args:
            image (th.Tensor): image tensor, shape (B, C, H, W).
            target (th.Tensor): Second tensor, shape (B, C, H, W).
            masks (th.Tensor): th.bool mask with False for masked (= to consider) pixels, shape (B, C, H, W).
        Returns:
            th.Tensor: The PCC between the two tensors.
        """
        
        # Create a mask for the NaN values, True where the NaN values are
        nan_mask = self._get_nan_mask(target)
        
        # Create a mask for the valid pixels, True for valid pixels
        # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
        valid_mask = self._get_valid_mask(masks, nan_mask)
        
        # Count the number of valid pixels
        n_valid_pixels = valid_mask.sum().float()
        
        n_valid_pixels_inv = 1.0 / n_valid_pixels
        
        if n_valid_pixels == 0: # if all pixels are masked, return 0
            return 0
        
        # Set the non valid pixels to NaN
        image = image.masked_fill(~valid_mask, th.nan)
        target = target.masked_fill(~valid_mask, th.nan)
        
        # Flatten the tensors for each channel
        # Obtain the shape (B, C, H*W)
        x_flat = image.view(image.shape[0], image.shape[1], -1)
        y_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute the mean for each channel, shape (B, C, 1)
        x_mean = th.nansum(x_flat, dim=2, keepdim=True) * n_valid_pixels_inv
        y_mean = th.nansum(y_flat, dim=2, keepdim=True) * n_valid_pixels_inv
        
        # Center the tensors by subtracting the mean
        x_flat -= x_mean
        y_flat -= y_mean
        
        # cov = (x_centered * y_centered).mean(dim=2)
        cov = th.nansum(x_flat * y_flat, dim=2) * n_valid_pixels_inv
        
        # x_std = th.sqrt((x_flat ** 2).sum(dim=2))
        # y_std = th.sqrt((x_flat ** 2).sum(dim=2))
        
        x_std = th.sqrt(th.nansum(x_flat ** 2, dim=2) * n_valid_pixels_inv)
        y_std = th.sqrt(th.nansum(y_flat ** 2, dim=2) * n_valid_pixels_inv)
        
        if th.any(x_std == 0) or th.any(y_std == 0):
            raise ValueError("Standard deviation is zero, cannot compute PCC.")
        
        return cov / (x_std * y_std)
    
    def _get_nan_mask(self, target: th.Tensor) -> th.Tensor:
        """
        Computes the NaN mask for the tensors, True where the NaN values are.

        Args:
            target (th.Tensor): Second tensor, shape (B, C, H, W).

        Returns:
            th.Tensor: The NaN mask for the tensors.
        """
        
        return th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
    
    def _get_valid_mask(self, masks: th.Tensor, nan_mask: th.Tensor) -> th.Tensor:
        """
        Computes the valid mask for the tensors, True for valid pixels.

        Args:
            masks (th.Tensor): th.bool mask with True for pixels to consider, shape (B, C, H, W).
            nan_mask (th.Tensor): Mask for NaN values, True where NaN is shape (B, C, H, W).

        Returns:
            th.Tensor: The valid mask, True for valid pixels.
        """
        
        # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
        return ~(masks | nan_mask)
        
        
        
        