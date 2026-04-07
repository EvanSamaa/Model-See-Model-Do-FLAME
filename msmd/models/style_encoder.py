import torch.nn as nn
import sys
import torch
from .common import PositionalEncoding
from .flame import FLAME, FLAMEConfig

def get_style_encoder(args, style_encoder_model_style="diffposetalk"):
    print(args.dataset_type)
    if style_encoder_model_style[:18] == "model_see_model_do":
        print("training model: model_see_model_do")
        return StyleExampleScrambler(args)
    elif style_encoder_model_style == "diffposetalk":
        if args.dataset_type[:9] == 'HDTF_TFHP':
            print("training model: StyleEncoder")
            return StyleEncoder(args)
        elif args.dataset_type == "celebv-text-medium+ravdess-FLMAE":
            return StyleEncoder(args)
        elif args.dataset_type == 'celebv-text-toy':
            print("training model: StyleEncoder_celebv")
            return StyleEncoder_celebv(args)
        elif args.dataset_type == 'celebv-text':
            print("training model: StyleEncoder_celebv")
            return StyleEncoder_celebv(args)
        elif args.dataset_type == 'celebv-text-v2':
            print("training model: StyleEncoder_celebv")
            return StyleEncoder_celebv(args)
        elif args.dataset_type == "ravdess+celebv-text-medium":
            print("training model: StyleEncoder_celebv")
            return StyleEncoder_celebv(args)
        elif args.dataset_type == "ravdess+celebv-text-full":
            print("training model: StyleEncoder_celebv")
            return StyleEncoder_celebv(args)
        elif args.dataset_type == "celebv-text-medium":
            return StyleEncoder_celebv(args)
        else:
            return StyleEncoder(args)
    elif style_encoder_model_style == "vae":
        print("training model: StyleEncoder_VAE")
        return StyleEncoder_VAE(args)
    elif style_encoder_model_style == "vae2":
        print("training model: StyleEncoder_VAE2")
        return StyleEncoder_VAE2(args)
    elif style_encoder_model_style == "vae2_with_audio_feat":
        print("training model: StyleEncoder_VAE2_with_audio_feat")
        return StyleEncoder_VAE2_with_audio_feat(args)
    elif style_encoder_model_style == "vae2_with_stats":
        print("training model: StyleEncoder_VAE2_explicit_stats")
        return StyleEncoder_VAE2_explicit_stats(args)
    elif style_encoder_model_style == "vae2_with_lip_stats":
        print("training model: StyleEncoder_VAE2_lip_stats")
        return StyleEncoder_VAE2_lip_stats(args)
    elif style_encoder_model_style == "basic_encoder_transformer":
        print("training model: StyleEncoder_basic_encoder_transformer")
        return StyleEncoder_basic_encoder_transformer(args)
    elif style_encoder_model_style == "basic_encoder_transformer_fft":
        print("training model: StyleEncoder_basic_encoder_transformer")
        return StyleEncoder_basic_encoder_transformer(args, fft_encoder=True)
    elif style_encoder_model_style == "basic_encoder":
        print("training model: StyleEncoder_basic_encoder")
        return StyleEncoder_basic_encoder(args)
    elif style_encoder_model_style == "fft_encoder":
        print("training model: StyleEncoder_FFT")
        return StyleEncoder_basic_encoder(args, fft_encoder=True)
    elif style_encoder_model_style == "no_style":
        print("training model: StyleEncoder_no_style")
        return StyleEncoder_no_style(args)
    elif style_encoder_model_style == "gst":
        print("training model: StyleEncoder_GST")
        return StyleEncoder_GST(args)
    else:
        raise ValueError(f"Style Encoder Model style {style_encoder_model_style} not recognized")

def batched_fft(input_tensor):
    # Ensure the input tensor is on a complex format
    complex_input =  torch.complex(input_tensor, torch.zeros_like(input_tensor))
    # Perform FFT along the second dimension (time_steps)
    fft_result = torch.fft.fft(complex_input, dim=1)
    magnitude = torch.abs(fft_result)
    phase = torch.angle(fft_result) 
    
    # concatenate the magnitude and phase
    output_fft_features = torch.cat((magnitude, phase), dim=-1)
    return output_fft_features
    # return torch.view_as_real(fft_result)

class StyleEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        self.motion_coef_dim = 100
        if args.rot_repr == 'aa':
            self.motion_coef_dim += 1 if args.no_head_pose else 4
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')
        self.motion_coef_dim = 106
        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)

        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)

        return feat

class StyleEncoder_celebv(StyleEncoder):
    def __init__(self, args):
        super().__init__(args)

        # Model parameters
        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54
        self.motion_coef_dim = self.input_dim

        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)

        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)

        return feat

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims  # Tuple of dimensions to permute

    def forward(self, x):
        return x.permute(*self.dims)

class StyleEncoder_VAE(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2 * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ReLU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.ReLU(),
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:]
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class StyleEncoder_VAE2(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 106
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:] # this cannot have a relu since we need negative values!!!!!
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class StyleEncoder_VAE2_with_audio_feat(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54
        
        self.input_dim += 768

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:] # this cannot have a relu since we need negative values!!!!!
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



class StyleEncoder_VAE2_explicit_stats(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2

        self.motion_stats_layers = [
            nn.Linear(self.motion_coef_dim * 4, self.motion_coef_dim * 2),
            nn.ELU(),
            nn.LayerNorm(self.motion_coef_dim * 2),
            nn.Linear(self.motion_coef_dim * 2, self.output_size),
            nn.ELU(),
            nn.LayerNorm(self.output_size)
        ]
        self.motion_stats_layers = nn.Sequential(*self.motion_stats_layers)

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        motion_mean = motion_coef.mean(dim=1)
        motion_max = torch.max(motion_coef, dim=1).values
        motion_min = torch.min(motion_coef, dim=1).values
        motion_std = motion_coef.std(dim=1)
        motion_stats = torch.cat((motion_mean, motion_max, motion_min, motion_std), dim=-1)
        motion_stats_out = self.motion_stats_layers(motion_stats)
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len
        out = out + motion_stats_out

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:] # this cannot have a relu since we need negative values!!!!!
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class StyleEncoder_VAE2_lip_stats(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.flame = FLAME(FLAMEConfig)
        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2

        self.motion_stats_layers = [
            nn.Linear(16, self.motion_coef_dim * 2),
            nn.ELU(),
            nn.LayerNorm(self.motion_coef_dim * 2),
            nn.Linear(self.motion_coef_dim * 2, self.output_size),
            nn.ELU(),
            nn.LayerNorm(self.output_size)
        ]
        self.motion_stats_layers = nn.Sequential(*self.motion_stats_layers)

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, pre_normalized_coef_dict, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        self.flame.to(motion_coef.device)
        batch_size, seq_len, _ = motion_coef.shape
        __, __, lm3D = self.flame(shape_params=pre_normalized_coef_dict['shape'].reshape(-1, 100) * 0.0, 
                                expression_params=pre_normalized_coef_dict['exp'].reshape(-1, 50),
                                pose_params=pre_normalized_coef_dict['pose'].reshape(-1, 6), return_lm3d=True, return_lm2d=False)

        lm3D = lm3D.reshape(batch_size, seq_len, lm3D.shape[-2], lm3D.shape[-1])
        lm_upper_lower_distance_outer = torch.sum(torch.square(lm3D[:, :, 62] - lm3D[:, :, 66]), dim = -1)
        lm_upper_lower_distance_inner = torch.sum(torch.square(lm3D[:, :, 51] - lm3D[:, :, 57]), dim = -1)
        lm_left_right_distance_inner = torch.sum(torch.square(lm3D[:, :, 54] - lm3D[:, :, 48]), dim = -1)
        lm_left_right_distance_outer = torch.sum(torch.square(lm3D[:, :, 60] - lm3D[:, :, 64]), dim = -1)

        lm_upper_lower_distance_outer_min = lm_upper_lower_distance_outer.min(dim=-1, keepdim=True).values    
        lm_upper_lower_distance_outer_max = lm_upper_lower_distance_outer.max(dim=-1, keepdim=True).values
        lm_upper_lower_distance_outer_mean = lm_upper_lower_distance_outer.mean(dim=-1, keepdim=True)
        lm_upper_lower_distance_outer_std = lm_upper_lower_distance_outer.std(dim=-1, keepdim=True)

        lm_upper_lower_distance_inner_min = lm_upper_lower_distance_inner.min(dim=-1, keepdim=True).values
        lm_upper_lower_distance_inner_max = lm_upper_lower_distance_inner.max(dim=-1, keepdim=True).values
        lm_upper_lower_distance_inner_mean = lm_upper_lower_distance_inner.mean(dim=-1, keepdim=True)
        lm_upper_lower_distance_inner_std = lm_upper_lower_distance_inner.std(dim=-1, keepdim=True)

        lm_left_right_distance_inner_min = lm_left_right_distance_inner.min(dim=-1, keepdim=True).values
        lm_left_right_distance_inner_max = lm_left_right_distance_inner.max(dim=-1, keepdim=True).values
        lm_left_right_distance_inner_mean = lm_left_right_distance_inner.mean(dim=-1, keepdim=True)
        lm_left_right_distance_inner_std = lm_left_right_distance_inner.std(dim=-1, keepdim=True)

        lm_left_right_distance_outer_min = lm_left_right_distance_outer.min(dim=-1, keepdim=True).values
        lm_left_right_distance_outer_max = lm_left_right_distance_outer.max(dim=-1, keepdim=True).values
        lm_left_right_distance_outer_mean = lm_left_right_distance_outer.mean(dim=-1, keepdim=True)
        lm_left_right_distance_outer_std = lm_left_right_distance_outer.std(dim=-1, keepdim=True)

        motion_stats = torch.cat([
            lm_upper_lower_distance_outer_min, lm_upper_lower_distance_outer_max, lm_upper_lower_distance_outer_mean, lm_upper_lower_distance_outer_std,  
            lm_upper_lower_distance_inner_min, lm_upper_lower_distance_inner_max, lm_upper_lower_distance_inner_mean, lm_upper_lower_distance_inner_std,
            lm_left_right_distance_inner_min, lm_left_right_distance_inner_max, lm_left_right_distance_inner_mean, lm_left_right_distance_inner_std,
            lm_left_right_distance_outer_min, lm_left_right_distance_outer_max, lm_left_right_distance_outer_mean, lm_left_right_distance_outer_std          
        ], dim=-1)
        motion_stats_out = self.motion_stats_layers(motion_stats)
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len
        out = out + motion_stats_out

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:] # this cannot have a relu since we need negative values!!!!!
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class StyleEncoder_basic_encoder(nn.Module):
    def __init__(self, args, fft_encoder=False) -> None:
        super().__init__()
        

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54
        self.fft_encoder = fft_encoder
        if fft_encoder:
            self.motion_coef_dim = self.input_dim * 2 # we nee phase and magnitude
        else:
            self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style
        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        # perform fast fourier transform to the input features
        # self.fft_layer = 
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        if self.fft_encoder:
            motion_coef = batched_fft(motion_coef)
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        return out

class StyleExampleScrambler(nn.Module):
    def __init__(self, args):
        super().__init__()
        pass

    def forward(self, x):
        # scramble the input at dimension 1
        batchsize, seq_len, feature_dim = x.shape
        # get the indices
        indices = torch.randperm(seq_len)
        # apply the permutation
        x = x[:, indices, :]
        return x

class GSTLayer(nn.Module):
    def __init__(self, token_size, token_count, attention_heads_count):
        super().__init__()
        self.token_size = token_size
        self.token_count = token_count
        self.attention_heads_count = attention_heads_count

        # Learnable style tokens
        self.token_embedding = nn.Parameter(torch.randn(token_count, token_size))
        
        # Multi-head attention for all tokens combined
        self.attention = nn.MultiheadAttention(token_size, attention_heads_count, batch_first=True)

    def forward(self, x):
        """
        :param x: (batch_size, feature_dim)
        :return: (batch_size, token_size) - aggregated style embedding
        """
        batch_size, feature_dim = x.shape
        # Add a sequence dimension to x (new shape: (batch_size, 1, feature_dim))
        x = x.unsqueeze(1)

        # Expand token embeddings to match the batch size (new shape: (batch_size, token_count, token_size))
        token_embedding = self.token_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply attention mechanism - x attends to the style tokens
        attn_output, attn_weights = self.attention(x, token_embedding, token_embedding)  # (batch_size, 1, token_size)
        
        # Remove the sequence dimension to produce a single style embedding per sample
        style_embedding = attn_output.squeeze(1)  # (batch_size, token_size)

        return style_embedding
 
class StyleEncoder_GST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output_size = args.d_style
        self.style_token_count = args.style_token_count
        self.style_token_attention_heads = args.style_token_attention_heads

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio

        self.gst_layer = GSTLayer(self.feature_dim, self.style_token_count, self.style_token_attention_heads)

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })
        self.output_layer = nn.Linear(self.feature_dim, self.output_size)
    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)
        # take the last dimension
        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)
        # apply the gst layer
        feat = torch.tanh(feat)
        gst = self.gst_layer(feat)
        out_feat = self.output_layer(gst)
        return out_feat

class StyleEncoder_basic_encoder_transformer(nn.Module):
    def __init__(self, args, fft_encoder=False) -> None:
        super().__init__()
        
        self.fft_encoder = fft_encoder
        
        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        if fft_encoder:
            self.motion_coef_dim = self.input_dim * 2 # we need phase and magnitude
        else:
            self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style
        
        # Input layer
        self.input_layer = nn.Linear(self.motion_coef_dim, self.conv_feature_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(self.conv_feature_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # Positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.conv_feature_dim, self.output_size),
        )
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        if self.fft_encoder:
            motion_coef = batched_fft(motion_coef)
        
        # Input layer
        motion_feat = self.input_layer(motion_coef)
        motion_feat = self.dropout(motion_feat)
        motion_feat = self.layer_norm(motion_feat)
        
        # Transformer encoder
        motion_feat = self.PE(motion_feat)
        feat = self.transformer_encoder(motion_feat)
        
        # Output layer
        out = self.output_layer(feat.mean(dim=1))
        
        return out

class StyleEncoder_no_style(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.output_size = args.d_style

    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        out = torch.zeros(batch_size, self.output_size).to(motion_coef.device)

        return out

if __name__ == "__main__":
    # Test StyleEncoder
    # get one batch
    samples = []
    for i in range(0, 20):
        samples.append(train_loader.dataset[1])
    collated_sample = train_loader.dataset.get_collate_fn(False)(samples)    
    collated_sample[1][0]["motion"].shape
    enc = StyleEncoder_VAE(args)
    style = enc(collated_sample[1][0]["motion"])