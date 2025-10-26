import modelbit, sys
from TTS.api import TTS
import os
import torch
import base64
import io
import tempfile
import shutil

MODEL_PATH = 'phase2_training/checkpoints/voice_model_20241220_143000_epoch_268.pt'
REFERENCE_AUDIO_PATH = 'dataset/wavs/qCGSu_5.wav'

# main function
def generate_voice_audio(text: str, language: str = "en") -> str:
    """
    Generate audio from text using locally downloaded TTS model
    """
    try:
        print(f"🎙️ Generating audio for: '{text[:50]}...'")

        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Load model from local path
        from TTS.api import TTS
        import torch
        import numpy as np
        import soundfile as sf
        import base64
        import io

        # Force CPU usage
        torch.set_num_threads(1)

        # Set up writable cache directory for TTS
        cache_dir = tempfile.mkdtemp()
        os.environ['TTS_CACHE'] = cache_dir
        
        # Load TTS model from local directory
        tts_model = TTS("tts_models_local/tts/tts_models--multilingual--multi-dataset--xtts_v2")

        # Load your trained weights if available
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            if 'model_state_dict' in checkpoint and checkpoint['model_state_dict'] is not None:
                tts_model.synthesizer.tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Trained weights applied to model")
            else:
                print("⚠️  No trained weights found in checkpoint")

        # Generate audio using XTTS
        try:
            audio_data = tts_model.tts(
                text=text,
                speaker_wav=REFERENCE_AUDIO_PATH,
                language=language
            )
        except Exception as e:
            print(f"❌ Error in TTS generation: {e}")
            raise e

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
    finally:
        # Clean up temporary cache directory
        try:
            if 'cache_dir' in locals():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except:
            pass