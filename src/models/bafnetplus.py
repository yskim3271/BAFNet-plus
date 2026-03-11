import importlib
import torch
import torch.nn as nn
from src.stft import complex_to_mag_pha
from src.utils import ConfigDict
from src.models.backbone import CausalConv1d, CausalConv2d, get_padding, get_padding_2d


class BAFNetPlus(torch.nn.Module):
    def __init__(self,
                 conv_depth=4,
                 conv_channels=16,
                 conv_kernel_size=7,
                 calibration_hidden_channels=16,
                 calibration_depth=2,
                 calibration_kernel_size=5,
                 calibration_max_common_log_gain=0.5,
                 calibration_max_relative_log_gain=1.0,
                 args_mapping=None,
                 args_masking=None,
                 checkpoint_mapping=None,
                 checkpoint_masking=None,
                 load_pretrained_weights=True,
                 ):
        super(BAFNetPlus, self).__init__()

        self.conv_depth = conv_depth
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        if isinstance(conv_kernel_size, int):
            self.alpha_kernel_size = (conv_kernel_size, conv_kernel_size)
        else:
            self.alpha_kernel_size = conv_kernel_size
        self.alpha_padding = get_padding_2d(self.alpha_kernel_size)
        self.calibration_hidden_channels = calibration_hidden_channels
        self.calibration_depth = calibration_depth
        self.calibration_kernel_size = calibration_kernel_size
        self.calibration_max_common_log_gain = calibration_max_common_log_gain
        self.calibration_max_relative_log_gain = calibration_max_relative_log_gain

        # Load checkpoints once (reused for both config extraction and weight loading)
        self._checkpoint_data_mapping = None
        self._checkpoint_data_masking = None
        if checkpoint_mapping is not None:
            self._checkpoint_data_mapping = torch.load(checkpoint_mapping, map_location='cpu', weights_only=False)
        if checkpoint_masking is not None:
            self._checkpoint_data_masking = torch.load(checkpoint_masking, map_location='cpu', weights_only=False)

        # Auto-load mapping model config from checkpoint if not provided
        if args_mapping is None:
            if self._checkpoint_data_mapping is None:
                raise ValueError("Either args_mapping or checkpoint_mapping must be provided")
            print(f"[BAFNetPlus] Auto-loading mapping model config from: {checkpoint_mapping}")
            mapping_config = self._extract_model_config(self._checkpoint_data_mapping, checkpoint_mapping)
            args_mapping = ConfigDict(mapping_config)

        # Auto-load masking model config from checkpoint if not provided
        if args_masking is None:
            if self._checkpoint_data_masking is None:
                raise ValueError("Either args_masking or checkpoint_masking must be provided")
            print(f"[BAFNetPlus] Auto-loading masking model config from: {checkpoint_masking}")
            masking_config = self._extract_model_config(self._checkpoint_data_masking, checkpoint_masking)
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

        # Alpha fusion conv stack (3ch input: bcs_mag_cal, acs_mag_cal, acs_mask)
        alpha_in_channels = 3
        self.alpha_convblocks = nn.ModuleList()
        for i in range(self.conv_depth):
            in_ch = alpha_in_channels if i == 0 else self.conv_channels
            self.alpha_convblocks.append(nn.Sequential(
                CausalConv2d(
                    in_channels=in_ch,
                    out_channels=self.conv_channels,
                    kernel_size=self.alpha_kernel_size,
                    stride=1,
                    padding=self.alpha_padding,
                ),
                nn.BatchNorm2d(self.conv_channels),
                nn.PReLU(self.conv_channels),
            ))
        self.alpha_out = nn.Conv2d(self.conv_channels, 2, kernel_size=1)

        # Calibration encoder (5ch input: bcs/acs log-energy, diff, mask mean/var)
        calibration_in_channels = 5
        calibration_layers = []
        for i in range(self.calibration_depth):
            in_ch = calibration_in_channels if i == 0 else self.calibration_hidden_channels
            calibration_layers.append(nn.Sequential(
                CausalConv1d(
                    in_channels=in_ch,
                    out_channels=self.calibration_hidden_channels,
                    kernel_size=self.calibration_kernel_size,
                    padding=get_padding(self.calibration_kernel_size),
                ),
                nn.PReLU(self.calibration_hidden_channels),
            ))
        self.calibration_encoder = nn.Sequential(*calibration_layers)
        self.common_gain_head = nn.Conv1d(self.calibration_hidden_channels, 1, kernel_size=1)
        self.relative_gain_head = nn.Conv1d(self.calibration_hidden_channels, 1, kernel_size=1)

        # Init only BAFNetPlus-owned layers (not pretrained backbones)
        self._init_fusion_modules()

        if load_pretrained_weights:
            if self._checkpoint_data_mapping is not None:
                print(f"[BAFNetPlus] Loading pretrained mapping weights from: {checkpoint_mapping}")
                self.mapping.load_state_dict(self._checkpoint_data_mapping["model"])
            if self._checkpoint_data_masking is not None:
                print(f"[BAFNetPlus] Loading pretrained masking weights from: {checkpoint_masking}")
                self.masking.load_state_dict(self._checkpoint_data_masking["model"])
        else:
            print("[BAFNetPlus] Skipping pretrained weights loading (will load from checkpoint)")

        # Free checkpoint data after loading
        del self._checkpoint_data_mapping, self._checkpoint_data_masking

    @staticmethod
    def _extract_model_config(checkpoint_data, checkpoint_path):
        """Extract model config dict from pre-loaded checkpoint data."""
        if 'args' not in checkpoint_data:
            raise KeyError(f"Checkpoint does not contain 'args' field: {checkpoint_path}")
        args = checkpoint_data['args']
        if not hasattr(args, 'model'):
            raise KeyError(f"Checkpoint args does not contain 'model' field: {checkpoint_path}")
        model_config = args.model
        return {
            'model_lib': model_config.model_lib,
            'model_class': model_config.model_class,
            'param': dict(model_config.param),
        }

    def _init_fusion_modules(self):
        """Initialize only BAFNetPlus-owned layers (alpha conv, calibration, gain heads)."""
        owned_modules = [self.alpha_convblocks, self.alpha_out,
                         self.calibration_encoder, self.common_gain_head, self.relative_gain_head]
        for parent in owned_modules:
            for m in parent.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _build_calibration_features(self, bcs_mag, acs_mag, acs_mask):
        """Build frame-wise causal calibration features."""
        eps = 1e-8
        bcs_log_energy = torch.log(bcs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        acs_log_energy = torch.log(acs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        log_energy_diff = bcs_log_energy - acs_log_energy
        acs_mask_mean = acs_mask.mean(dim=1, keepdim=True)
        acs_mask_var = acs_mask.var(dim=1, keepdim=True, unbiased=False)
        return torch.cat(
            [bcs_log_energy, acs_log_energy, log_energy_diff, acs_mask_mean, acs_mask_var],
            dim=1,
        )

    def _apply_calibration(self, bcs_com_out, acs_com_out, bcs_mag, acs_mag, acs_mask):
        """Apply frame-wise causal calibration before fusion."""
        calibration_feat = self._build_calibration_features(bcs_mag, acs_mag, acs_mask)
        calibration_hidden = self.calibration_encoder(calibration_feat)

        common_log_gain = torch.tanh(self.common_gain_head(calibration_hidden))
        common_log_gain = common_log_gain * self.calibration_max_common_log_gain
        relative_log_gain = torch.tanh(self.relative_gain_head(calibration_hidden))
        relative_log_gain = relative_log_gain * self.calibration_max_relative_log_gain

        bcs_gain = torch.exp(common_log_gain - 0.5 * relative_log_gain)
        acs_gain = torch.exp(common_log_gain + 0.5 * relative_log_gain)

        bcs_gain = bcs_gain.transpose(1, 2).unsqueeze(1)
        acs_gain = acs_gain.transpose(1, 2).unsqueeze(1)

        bcs_com_cal = bcs_com_out * bcs_gain
        acs_com_cal = acs_com_out * acs_gain
        return bcs_com_cal, acs_com_cal

    def _build_alpha_features(self, bcs_com_cal, acs_com_cal, acs_mask):
        """Build TF-wise fusion features from calibrated branch outputs."""
        eps = 1e-8
        bcs_mag_cal = torch.sqrt(bcs_com_cal[:, :, :, 0] ** 2 + bcs_com_cal[:, :, :, 1] ** 2 + eps)
        acs_mag_cal = torch.sqrt(acs_com_cal[:, :, :, 0] ** 2 + acs_com_cal[:, :, :, 1] ** 2 + eps)
        return torch.stack([bcs_mag_cal, acs_mag_cal, acs_mask], dim=1).transpose(2, 3)

    def forward(self, input):
        """
        Forward pass for BAFNetPlus with frame-wise causal calibration.

        Args:
            input: Tuple of (bcs_com, acs_com), each shaped [B, F, T, 2]

        Returns:
            est_mag: Estimated magnitude [B, F, T]
            est_pha: Estimated phase [B, F, T]
            est_com: Estimated complex spectrogram [B, F, T, 2]
        """
        if isinstance(input, tuple):
            bcs_com, acs_com = input
        else:
            raise ValueError("BAFNetPlus requires tuple input: (bcs_com, acs_com)")

        bcs_mag, _, bcs_com_out = self.mapping(bcs_com)
        acs_mag, _, acs_com_out, acs_mask = self.masking(acs_com, return_mask=True)

        bcs_com_cal, acs_com_cal = self._apply_calibration(
            bcs_com_out=bcs_com_out,
            acs_com_out=acs_com_out,
            bcs_mag=bcs_mag,
            acs_mag=acs_mag,
            acs_mask=acs_mask,
        )

        alpha = self._build_alpha_features(bcs_com_cal, acs_com_cal, acs_mask)
        for block in self.alpha_convblocks:
            alpha = block(alpha)

        alpha = self.alpha_out(alpha)
        alpha = alpha.transpose(2, 3)
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)

        est_com = bcs_com_cal * alpha_bcs + acs_com_cal * alpha_acs
        est_mag, est_pha = complex_to_mag_pha(est_com)
        return est_mag, est_pha, est_com
