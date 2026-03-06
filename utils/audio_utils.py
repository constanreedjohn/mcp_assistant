"""
Audio processing utilities for the MCP Assistant.
Handles audio loading, normalization, and conversion.
"""
import io
import base64
import urllib.request
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Union, List

from .logging_utils import logger
from config import SAMPLE_RATE, MAX_ASR_INPUT_SECONDS, MIN_ASR_INPUT_SECONDS

# Type alias for audio input
AudioLike = Union[str, Tuple[np.ndarray, int]]  # wav path / URL / base64 / (waveform, sr)


def is_probably_base64(s: str) -> bool:
    """Check if a string is likely base64 encoded audio."""
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    """Decode base64 string to bytes."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def is_url(s: str) -> bool:
    """Check if a string is a valid URL."""
    from urllib.parse import urlparse
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    """Load audio from various sources (URL, base64, file path).
    
    Args:
        x: Audio source - can be URL, base64 string, or file path
        
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr


def ensure_list(x) -> List:
    """Ensure input is a list."""
    return x if isinstance(x, list) else [x]


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono.
    
    Args:
        audio: Audio array (can be 1D or 2D)
        
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 range [-1, 1].
    
    Args:
        audio: Audio array
        
    Returns:
        Normalized audio array
    """
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    # If decoded audio is int-like scaled or out-of-range, normalize conservatively.
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def normalize_audio_input(a: AudioLike) -> np.ndarray:
    """Normalize one audio input to mono 16k float32 waveform in [-1, 1].

    Supported inputs:
        - str: local file path / https URL / base64 audio string
        - (np.ndarray, sr): waveform and sampling rate

    Args:
        a: Audio input (path, URL, base64, or tuple of waveform and sr)
        
    Returns:
        Mono 16k float32 waveform in [-1, 1].
    """
    if isinstance(a, str):
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")

    audio = to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    audio = float_range_normalize(audio)
    return audio


def normalize_audios(audios: Union[AudioLike, List[AudioLike]]) -> List[np.ndarray]:
    """Normalize multiple audio inputs.
    
    Args:
        audios: Single audio input or list of audio inputs
        
    Returns:
        List of normalized audio arrays
    """
    items = ensure_list(audios)
    return [normalize_audio_input(a) for a in items]


def read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Read WAV audio from bytes.
    
    Args:
        audio_bytes: Raw WAV audio bytes
        
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def save_audio_to_file(audio: np.ndarray, sr: int, filepath: str) -> None:
    """Save audio to WAV file.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        filepath: Output file path
    """
    sf.write(filepath, audio, sr)
    logger.info(f"Saved audio to {filepath}")

