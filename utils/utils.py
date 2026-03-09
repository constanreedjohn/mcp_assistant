"""
General utilities for the MCP Assistant.
This module re-exports functionality from specialized utility modules.
"""
# Re-export audio utilities
from .audio_utils import (
    is_probably_base64,
    decode_base64_bytes,
    is_url,
    load_audio_any,
    ensure_list,
    to_mono,
    float_range_normalize,
    normalize_audio_input,
    normalize_audios,
    read_wav_from_bytes,
    save_audio_to_file,
    encode_image,
    AudioLike,
)

# Re-export weather utilities
from .weather_utils import (
    make_nws_request,
    format_alert,
    get_weather_alerts,
    get_weather_forecast,
)

# Re-export constants
from .constants import (
    SAMPLE_RATE,
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    MIN_ASR_INPUT_SECONDS,
    NWS_API_BASE,
    USER_AGENT,
)

# Keep backward compatibility - keep the old function name
_read_wav_from_bytes = read_wav_from_bytes

