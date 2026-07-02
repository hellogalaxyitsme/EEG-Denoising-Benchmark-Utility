"""Model definitions for the EEG denoising benchmark utility."""

from .controlled_backbone import (
    DSConv1D,
    ECA1D,
    HaarDWT1D,
    LinearFIRDenoiser,
    PatchTransformerDenoiser,
    TinyDenoiser,
    TinyTCNDenoiser,
    count_trainable_parameters,
)
from .eegdn_baselines import EEGDNComplexCNN, EEGDNRNNLSTM, build_eegdn_baseline
from .microwavenet import MicroWaveNet

__all__ = [
    "DSConv1D",
    "ECA1D",
    "EEGDNComplexCNN",
    "EEGDNRNNLSTM",
    "HaarDWT1D",
    "LinearFIRDenoiser",
    "MicroWaveNet",
    "PatchTransformerDenoiser",
    "TinyDenoiser",
    "TinyTCNDenoiser",
    "build_eegdn_baseline",
    "count_trainable_parameters",
]
