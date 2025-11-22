import importlib
import torch
import torch.nn as nn
from src.stft import complex_to_mag_pha
from src.utils import load_model_config_from_checkpoint, ConfigDict

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class BAFNet(torch.nn.Module):
    def __init__(self,
                 win_len=400,
                 hop_len=100,
                 fft_len=400,
                 conv_depth=4, 
                 conv_channels=16, 
                 conv_kernel_size=7,
                 learnable_sigmoid=True,
                 args_mapping=None, 
                 args_masking=None, 
                 checkpoint_mapping=None,
                 checkpoint_masking=None,
                 ):
        super(BAFNet, self).__init__()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        
        self.conv_depth = conv_depth
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_kernel_size // 2
        self.learnable_sigmoid = learnable_sigmoid

        # Auto-load mapping model config from checkpoint if not provided
        if args_mapping is None:
            if checkpoint_mapping is None:
                raise ValueError("Either args_mapping or checkpoint_mapping must be provided")
            print(f"[BAFNet] Auto-loading mapping model config from: {checkpoint_mapping}")
            mapping_config = load_model_config_from_checkpoint(checkpoint_mapping)
            args_mapping = ConfigDict(mapping_config)

        # Auto-load masking model config from checkpoint if not provided
        if args_masking is None:
            if checkpoint_masking is None:
                raise ValueError("Either args_masking or checkpoint_masking must be provided")
            print(f"[BAFNet] Auto-loading masking model config from: {checkpoint_masking}")
            masking_config = load_model_config_from_checkpoint(checkpoint_masking)
            args_masking = ConfigDict(masking_config)

        module_mapping = importlib.import_module("src.models." + args_mapping.model_lib)
        module_masking = importlib.import_module("src.models." + args_masking.model_lib)

        mapping_class = getattr(module_mapping, args_mapping.model_class)
        masking_class = getattr(module_masking, args_masking.model_class)

        # Convert ConfigDict to dict if needed for ** unpacking
        mapping_params = args_mapping.param.to_dict() if hasattr(args_mapping.param, 'to_dict') else args_mapping.param
        masking_params = args_masking.param.to_dict() if hasattr(args_masking.param, 'to_dict') else args_masking.param

        self.mapping = mapping_class(**mapping_params)
        self.masking = masking_class(**masking_params)
        
        for i in range(self.conv_depth):
            setattr(self, f"convblock_{i}", nn.Sequential(
                nn.Conv2d(in_channels=self.conv_channels if i != 0 else 1, 
                          out_channels=self.conv_channels if i != self.conv_depth - 1 else 1, 
                          kernel_size=self.conv_kernel_size, 
                          stride=1, 
                          padding=self.conv_padding),
                nn.BatchNorm2d(self.conv_channels if i != self.conv_depth - 1 else 1),
                nn.PReLU() if i != self.conv_depth - 1 else nn.Identity()
            ))
        
        if self.learnable_sigmoid:
            self.sigmoid = LearnableSigmoid2d(self.fft_len // 2 + 1)
        else:
            self.sigmoid = nn.Sigmoid()
        
        
        self.init_modules()

        if checkpoint_mapping is not None:
            self.mapping.load_state_dict(torch.load(checkpoint_mapping, weights_only=False)['model'])
        if checkpoint_masking is not None:
            self.masking.load_state_dict(torch.load(checkpoint_masking, weights_only=False)['model'])
        
    def init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, input):
        """
        Forward pass for BAFNet with STFT inputs.

        Args:
            input: Tuple of (bcs_com, acs_com) where each is [B, F, T, 2]
                   bcs_com: Bone conduction signal complex spectrogram (processed by mapping model)
                   acs_com: Air conduction signal complex spectrogram (processed by masking model)

        Returns:
            est_mag: Estimated magnitude [B, F, T]
            est_pha: Estimated phase [B, F, T]
            est_com: Estimated complex spectrogram [B, F, T, 2]
        """
        # Unpack input tuple (compatible with solver's input_type='acs+bcs')
        if isinstance(input, tuple):
            bcs_com, acs_com = input
        else:
            raise ValueError("BAFNet requires tuple input: (bcs_com, acs_com)")


        # Call mapping and masking submodels
        # Both models expect [B, F, T, 2] and return (mag, pha, com)
        bcs_mag, bcs_pha, bcs_com_out = self.mapping(bcs_com)
        acs_mag, acs_pha, acs_com_out = self.masking(acs_com)

        # Calculate mask from masking model output
        # mask represents the magnitude ratio (enhanced / noisy)
        acs_input_mag = torch.sqrt(acs_com[:, :, :, 0]**2 + acs_com[:, :, :, 1]**2 + 1e-8)
        mask = acs_mag / (acs_input_mag + 1e-8)  # [B, F, T]
        mask = torch.clamp(mask, min=0.0, max=10.0)

        # Normalize by RMS energy (원본 BAFNet 방식)
        bcs_power = bcs_com_out[:, :, :, 0]**2 + bcs_com_out[:, :, :, 1]**2  # [B, F, T]
        bcs_energy = (bcs_power.mean(dim=[1, 2], keepdim=True) + 1e-8).sqrt().unsqueeze(-1)  # [B, 1, 1, 1]

        acs_power = acs_com_out[:, :, :, 0]**2 + acs_com_out[:, :, :, 1]**2  # [B, F, T]
        acs_energy = (acs_power.mean(dim=[1, 2], keepdim=True) + 1e-8).sqrt().unsqueeze(-1)  # [B, 1, 1, 1]

        bcs_com_norm = bcs_com_out / (bcs_energy + 1e-8)
        acs_com_norm = acs_com_out / (acs_energy + 1e-8)

        # Compute alpha mask from mask using CNN layers
        alpha = mask.unsqueeze(1)  # [B, 1, F, T]
        for i in range(self.conv_depth):
            alpha = getattr(self, f"convblock_{i}")(alpha)
        alpha = alpha.squeeze(1)  # [B, F, T]
        alpha = self.sigmoid(alpha).unsqueeze(-1)  # [B, F, T, 1]

        # Blend normalized complex spectrograms using alpha
        est_com_norm = bcs_com_norm * alpha + acs_com_norm * (1 - alpha)

        # Denormalize using average of both energies (원본 BAFNet 방식)
        avg_energy = (bcs_energy + acs_energy) / 2.0  # [B, 1, 1, 1]
        est_com = est_com_norm * avg_energy

        est_mag, est_pha = complex_to_mag_pha(est_com)

        return est_mag, est_pha, est_com
