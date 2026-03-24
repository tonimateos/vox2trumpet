import torch
import torch.nn as nn
import torch.nn.functional as F
from synth import HarmonicSynthesizer, FilteredNoiseSynthesizer

# ==============================================================================
# Research Note: Vox2Trumpet Architecture (Brain & Body)
# ==============================================================================
# 1. The "Brain" (Decoder):
#    - We use a GRU (Recurrent Neural Network) because audio generation 
#      is fundamentally sequential and stateful (reverberation, decay).
#    - Input: Log-F0 and Log-Loudness (Perceptually relevant features).
#    - Inductive Bias: We don't ask the network to generate samples. 
#      We ask it to "perform" the instrument by controlling the knobs 
#      of the synthesizers.
#
# 2. Parameter Mapping:
#    - Harmonic Amplitudes: Force to sum-to-1 via Softmax (distribution).
#      Overall amplitude is controlled by the input Loudness capability.
#    - Noise Magnitudes: Sigmoid activation to bound filter response [0, 1].
# ==============================================================================

class Vox2Trumpet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        # Mandatory parameters from config - will raise KeyError if missing
        self.n_harmonics = config['n_harmonics']
        self.n_noise_bands = config['n_noise_bands']
        self.hidden_size = config['hidden_size']
        self.sample_rate = config['sample_rate']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_noise = config['use_noise']
        
        # --- The Body (DSP) ---
        self.harmonic_synth = HarmonicSynthesizer(self.n_harmonics, self.sample_rate)
        self.noise_synth = FilteredNoiseSynthesizer(self.n_noise_bands)
        
        # --- The Brain (Decoder) ---
        # Input features: f0 (1) + loudness (1) = 2
        self.gru = nn.GRU(
            input_size=2, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        self.mlp = nn.Linear(self.hidden_size, self.n_harmonics + self.n_noise_bands)
        
        # Stability: Small initial weights for the projection layer 
        # to prevent NaN explosions at start of training
        nn.init.normal_(self.mlp.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.mlp.bias, 0)
        


    def forward(self, f0: torch.Tensor, loudness: torch.Tensor, return_controls: bool = False) -> torch.Tensor:
        """
        Args:
            f0 (torch.Tensor): Fundamental frequency in Hz [Batch, Time, 1]
            loudness (torch.Tensor): Loudness signal (normalized dB usually) [Batch, Time, 1]
            
        Returns:
            torch.Tensor: Synthesized Audio [Batch, Time]
        """
        # 1. Feature Preprocessing
        # Log-scale f0 helps the network linearize pitch space
        # Trumpet range is roughly 150Hz - 1200Hz. 
        # log(400) is ~6.0. Let's center it.
        log_f0 = (torch.log(f0 + 1e-7) - 6.0) / 2.0
        
        decoder_input = torch.cat([log_f0, loudness], dim=-1) # [B, T, 2]
        
        # 2. Decoder (GRU)
        # x: [B, T, hidden_size]
        # _: we don't need the hidden state of the GRU
        x, _ = self.gru(decoder_input)
        
        # 3. Parameter Projection
        # params: [B, T, H + N]
        params = self.mlp(x)
        
        # Split params
        harm_params = params[..., :self.n_harmonics]
        noise_params = params[..., self.n_harmonics:]
        
        # 4. Activation / Mapping
        # For the spectral envelope, we use a Softmax to ensure the harmonics 
        # form a valid distribution (sum to 1.0).
        harm_dist = F.softmax(harm_params, dim=-1)
        
        # To enforce the volume contour, we scale the distribution by the 
        # input loudness. Since our preprocessing extracts linear RMS amplitude,
        # we can multiply directly to get the final harmonic amplitudes.
        harm_amps = harm_dist * loudness 
        
        # Noise Params:
        # Magnitudes in [0, 1]
        noise_mags = torch.sigmoid(noise_params)
        
        # 5. Synthesis
        harmonic_audio = self.harmonic_synth(f0, harm_amps)
        
        if self.use_noise:
            noise_audio = self.noise_synth(noise_mags)
            final_audio = harmonic_audio + noise_audio
        else:
            noise_audio = torch.zeros_like(harmonic_audio)
            final_audio = harmonic_audio

        if return_controls:
            return {
                'audio': final_audio,
                'harm_amps': harm_amps,
                'noise_mags': noise_mags,
                'harm_dist': harm_dist,
                'log_f0': log_f0
            }

        return final_audio
