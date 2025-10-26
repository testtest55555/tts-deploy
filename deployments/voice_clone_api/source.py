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
MODEL_DIR = 'tts_models_local/tts/tts_models--multilingual--multi-dataset--xtts_v2'

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

        # Set up writable cache directory for TTS BEFORE importing TTS
        cache_dir = tempfile.mkdtemp()
        os.environ['TTS_CACHE'] = cache_dir
        os.environ['TTS_HOME'] = cache_dir
        os.environ['XDG_CACHE_HOME'] = cache_dir
        os.environ['HOME'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['TORCH_HOME'] = cache_dir
        os.environ['TTS_DISABLE_TOS'] = '1'  # Disable terms of service prompt
        
        print(f"🔧 Using cache directory: {cache_dir}")
        
        # Pre-accept terms of service to avoid interactive prompts
        model_dir = os.path.abspath(MODEL_DIR)  # Use absolute path
        tos_file = os.path.join(model_dir, "tos_agreed.txt")
        if not os.path.exists(tos_file):
            os.makedirs(model_dir, exist_ok=True)
            with open(tos_file, 'w') as f:
                f.write("1")  # Accept terms of service
            print("✅ Pre-accepted terms of service")
        
        # Debug: Check what files are in the model directory
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"📁 Model directory contents: {files}")
            
            # Ensure config.json exists and is readable
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                print(f"✅ Config file found: {config_path}")
            else:
                print(f"❌ Config file missing: {config_path}")
        else:
            print(f"❌ Model directory does not exist: {model_dir}")
        
        # Monkey patch input function to avoid interactive prompts
        import builtins
        original_input = builtins.input
        def mock_input(prompt=""):
            print(f"🤖 Auto-accepting prompt: {prompt.strip()}")
            return "y"
        builtins.input = mock_input
        
        # Load model from local path
        from TTS.api import TTS
        import torch
        import numpy as np
        import soundfile as sf
        import base64
        import io

        # Force CPU usage
        torch.set_num_threads(1)
        
        # Monkey patch torch.load to fix weights_only issue
        original_torch_load = torch.load
        def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            # Force weights_only=False for TTS model loading
            if weights_only is None:
                weights_only = False
            return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kwargs)
        torch.load = patched_torch_load
        
        # Also patch TTS internal loading functions
        try:
            from TTS.utils.io import load_fsspec
            original_load_fsspec = load_fsspec
            def patched_load_fsspec(f, map_location=None, **kwargs):
                return original_load_fsspec(f, map_location=map_location, weights_only=False, **kwargs)
            import TTS.utils.io
            TTS.utils.io.load_fsspec = patched_load_fsspec
            print("✅ Patched TTS internal loading functions")
        except Exception as e:
            print(f"⚠️  Could not patch TTS internal functions: {e}")
        
        # Load TTS model from local directory
        try:
            print("🔄 Loading TTS model...")
            
            # Method 1: Load Synthesizer with explicit file paths
            try:
                print("🔄 Trying Synthesizer with explicit paths...")
                from TTS.utils.synthesizer import Synthesizer
                
                # Define explicit paths
                model_path = os.path.join(model_dir, "model.pth")
                config_path = os.path.join(model_dir, "config.json")
                speakers_path = os.path.join(model_dir, "speakers_xtts.pth")
                vocab_path = os.path.join(model_dir, "vocab.json")
                
                # Verify all files exist
                for path, name in [(model_path, "model"), (config_path, "config"), (speakers_path, "speakers"), (vocab_path, "vocab")]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"{name} file not found: {path}")
                
                print(f"✅ All required files found")
                
                # Load Synthesizer with explicit paths
                tts_model = Synthesizer(
                    tts_checkpoint=model_path,
                    tts_config_path=config_path,
                    speakers_file_path=speakers_path,
                    vocab_path=vocab_path,
                    use_cuda=False
                )
                print("✅ TTS model loaded successfully with explicit paths")
                
            except Exception as e1:
                print(f"❌ Method 1 failed: {e1}")
                
                # Method 2: Try copying files to cache directory
                try:
                    print("🔄 Trying cache directory approach...")
                    
                    # Create model directory in cache
                    cache_model_dir = os.path.join(cache_dir, "tts_models--multilingual--multi-dataset--xtts_v2")
                    os.makedirs(cache_model_dir, exist_ok=True)
                    
                    # Copy all model files to cache
                    import shutil
                    for file in ["model.pth", "config.json", "speakers_xtts.pth", "vocab.json", "hash.md5", "tos_agreed.txt"]:
                        src = os.path.join(model_dir, file)
                        dst = os.path.join(cache_model_dir, file)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    
                    print(f"✅ Copied model files to cache: {cache_model_dir}")
                    
                    # Try loading from cache
                    tts_model = TTS(model_path=cache_model_dir)
                    print("✅ TTS model loaded successfully from cache")
                    
                except Exception as e2:
                    print(f"❌ Method 2 failed: {e2}")
                    raise e1
                    
        except Exception as e:
            print(f"❌ All methods failed: {e}")
            raise e
        finally:
            # Restore original functions
            builtins.input = original_input
            torch.load = original_torch_load
            try:
                import TTS.utils.io
                TTS.utils.io.load_fsspec = original_load_fsspec
            except:
                pass

        # Load your trained weights if available
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint and checkpoint['model_state_dict'] is not None:
                # Handle both TTS and Synthesizer objects
                if hasattr(tts_model, 'synthesizer'):
                    # TTS object
                    tts_model.synthesizer.tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif hasattr(tts_model, 'tts_model'):
                    # Synthesizer object
                    tts_model.tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    print("⚠️  Unknown model object type")
                print("✅ Trained weights applied to model")
            else:
                print("⚠️  No trained weights found in checkpoint")

        # Generate audio using XTTS
        try:
            # Check if we have a TTS object or Synthesizer object
            if hasattr(tts_model, 'tts'):
                # TTS object
                audio_data = tts_model.tts(
                    text=text,
                    speaker_wav=REFERENCE_AUDIO_PATH,
                    language=language
                )
            elif hasattr(tts_model, 'tts_model'):
                # Synthesizer object - use tts method
                audio_data = tts_model.tts(
                    text=text,
                    speaker_wav=REFERENCE_AUDIO_PATH,
                    language=language
                )
            else:
                raise ValueError("Unknown model object type - cannot generate audio")
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
        # Clean up temporary cache directory and memory
        try:
            if 'cache_dir' in locals():
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            # Force garbage collection to prevent memory issues
            import gc
            gc.collect()
            
            # Clear any remaining references
            if 'tts_model' in locals():
                del tts_model
            if 'audio_data' in locals():
                del audio_data
                
        except:
            pass