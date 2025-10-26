import modelbit, sys
from TTS.api import TTS
import os
import torch
import numpy as np
import io
import soundfile as sf
import base64

MODEL_PATH = 'phase2_training/checkpoints/voice_model_20241220_143000_epoch_268.pt'
REFERENCE_AUDIO_PATH = 'dataset/wavs/qCGSu_5.wav'

# main function
def generate_voice_audio(text: str, language: str = "en") -> str:
    """
    Generate audio from text using your trained voice model
    """
    try:
        print(f"🎙️ Generating audio for: '{text[:50]}...'")

        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Set TTS cache directory to a writable location
        import os
        import tempfile

        # Create a temporary directory for TTS cache
        temp_dir = tempfile.mkdtemp()
        os.environ['TTS_HOME'] = temp_dir

        # Load model inside the function
        from TTS.api import TTS
        import torch

        # Force CPU usage
        torch.set_num_threads(1)

        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Load your trained weights
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            print("✅ Trained weights loaded")

        # Generate audio using XTTS
        audio_data = tts_model.tts(
            text=text,
            speaker_wav=REFERENCE_AUDIO_PATH,
            language=language
        )

        # Convert to numpy array if needed
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)

        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

        # Convert to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, 22050, format='WAV')
        audio_bytes.seek(0)

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode('utf-8')

        print(f"✅ Generated audio successfully ({len(audio_base64)} chars)")
        return audio_base64

    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        raise e

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = generate_voice_audio(...)
#   print(result)