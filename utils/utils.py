import io
import base64
import requests
import librosa
import traceback
import numpy as np
import urllib.request
import soundfile as sf
from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse


from .logging_utils import logger

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

# Audio config
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
MIN_ASR_INPUT_SECONDS = 0.5

AudioLike = Union[
    str,                      # wav path / URL / base64
    Tuple[np.ndarray, int],   # (waveform, sr)
]
MaybeList = Union[Any, List[Any]]

async def make_nws_request(url: str) -> dict[str, Any]:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30.0)
        print(f"+++ GOT RESPONSE: {response.status_code}")
        if response.status_code != 200:
            return Exception(f"Fail to get response")
        else:
            return response.json()
    except Exception as e:
        logger.info(f"Fail to get response - {e}")
        logger.info(traceback.format_exc())
        return e

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props: dict = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)

def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False
    
def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
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

def ensure_list(x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")

def float_range_normalize(audio: np.ndarray) -> np.ndarray:
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
    """
    Normalize one audio input to mono 16k float32 waveform in [-1, 1].

    Supported inputs:
        - str: local file path / https URL / base64 audio string
        - (np.ndarray, sr): waveform and sampling rate

    Returns:
        np.ndarray:
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
    items = ensure_list(audios)
    return [normalize_audio_input(a) for a in items]

def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)