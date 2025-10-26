import modelbit, sys
import os
import torch
import base64
import io
import tempfile
import shutil
import numpy as np
import soundfile as sf

MODEL_PATH = 'phase2_training/checkpoints/voice_model_20241220_143000_epoch_268.pt'
REFERENCE_AUDIO_PATH = 'dataset/wavs/qCGSu_5.wav'
MODEL_DIR = 'tts_models_local/tts/tts_models--multilingual--multi-dataset--xtts_v2'

def generate_voice_audio(text: str, language: str = "en") -> str:
    """
    Generate audio from text using locally downloaded TTS model.
    Uses three different methods to load XTTS model based on what's available.
    """
    try:
        print(f"🎙️ Generating audio for: '{text[:50]}...'")

        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Set up writable cache directory
        cache_dir = tempfile.mkdtemp()
        os.environ['TTS_CACHE'] = cache_dir
        os.environ['TTS_HOME'] = cache_dir
        os.environ['XDG_CACHE_HOME'] = cache_dir
        os.environ['HOME'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['TORCH_HOME'] = cache_dir
        os.environ['TTS_DISABLE_TOS'] = '1'
        
        print(f"🔧 Using cache directory: {cache_dir}")
        
        # Get absolute path to model directory
        model_dir = os.path.abspath(MODEL_DIR)
        
        # Pre-accept terms of service
        tos_file = os.path.join(model_dir, "tos_agreed.txt")
        if not os.path.exists(tos_file):
            os.makedirs(model_dir, exist_ok=True)
            with open(tos_file, 'w') as f:
                f.write("1")
            print("✅ Pre-accepted terms of service")
        
        # Check model directory contents
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"📁 Model directory contents: {files}")
        else:
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        
        # Monkey patch input function to avoid interactive prompts
        import builtins
        original_input = builtins.input
        def mock_input(prompt=""):
            print(f"🤖 Auto-accepting prompt: {prompt.strip()}")
            return "y"
        builtins.input = mock_input
        
        # Force CPU usage
        torch.set_num_threads(1)
        
        # Patch torch.load to avoid weights_only issues
        original_torch_load = torch.load
        def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            if weights_only is None:
                weights_only = False
            return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                     weights_only=weights_only, **kwargs)
        torch.load = patched_torch_load
        
        # Patch TTS internal loading
        try:
            from TTS.utils.io import load_fsspec
            original_load_fsspec = load_fsspec
            def patched_load_fsspec(f, map_location=None, **kwargs):
                kwargs.pop('weights_only', None)
                return original_load_fsspec(f, map_location=map_location, **kwargs)
            import TTS.utils.io
            TTS.utils.io.load_fsspec = patched_load_fsspec
            print("✅ Patched TTS internal loading functions")
        except Exception as e:
            print(f"⚠️  Could not patch TTS internal functions: {e}")
        
        # Define file paths
        config_file = os.path.join(model_dir, "config.json")
        model_file = os.path.join(model_dir, "model.pth")
        vocab_file = os.path.join(model_dir, "vocab.json")
        
        # Verify required files exist
        for file_path, name in [(config_file, "config.json"), (model_file, "model.pth"), 
                                 (vocab_file, "vocab.json")]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        print("✅ All required files verified")
        
        # Try three different methods to load the model
        tts_model = None
        method_used = None
        
        # METHOD 1: Direct XTTS class loading (recommended for local models)
        try:
            print("🔄 Method 1: Loading with direct XTTS class...")
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            config = XttsConfig()
            config.load_json(config_file)
            
            tts_model = Xtts.init_from_config(config)
            tts_model.load_checkpoint(
                config, 
                checkpoint_dir=model_dir,
                vocab_path=vocab_file,
                eval=True,
                use_deepspeed=False
            )
            
            print("✅ Method 1 succeeded: Direct XTTS class loading")
            method_used = "direct_xtts"
            
        except Exception as e1:
            print(f"❌ Method 1 failed: {e1}")
            
            # METHOD 2: TTS API with model_name (auto-download from cache)
            try:
                print("🔄 Method 2: Loading with TTS API...")
                from TTS.api import TTS
                
                # Copy files to cache location where TTS expects them
                cache_model_dir = os.path.join(cache_dir, "tts_models--multilingual--multi-dataset--xtts_v2")
                os.makedirs(cache_model_dir, exist_ok=True)
                
                # Copy all model files
                for file in os.listdir(model_dir):
                    src = os.path.join(model_dir, file)
                    dst = os.path.join(cache_model_dir, file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                
                print(f"✅ Copied files to: {cache_model_dir}")
                
                # Load using TTS API with model name
                tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                
                print("✅ Method 2 succeeded: TTS API loading")
                method_used = "tts_api"
                
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                
                # METHOD 3: Synthesizer approach
                try:
                    print("🔄 Method 3: Loading with Synthesizer...")
                    from TTS.utils.synthesizer import Synthesizer
                    
                    tts_model = Synthesizer(
                        tts_checkpoint=model_file,
                        tts_config_path=config_file,
                        use_cuda=False
                    )
                    
                    print("✅ Method 3 succeeded: Synthesizer loading")
                    method_used = "synthesizer"
                    
                except Exception as e3:
                    print(f"❌ Method 3 failed: {e3}")
                    raise Exception(f"All loading methods failed. Last error: {e3}")
        
        # Restore original functions
        builtins.input = original_input
        torch.load = original_torch_load
        try:
            import TTS.utils.io
            TTS.utils.io.load_fsspec = original_load_fsspec
        except:
            pass
        
        # Load custom trained weights if available
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                print(f"🔄 Loading custom weights from: {MODEL_PATH}")
                checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint and checkpoint['model_state_dict'] is not None:
                    # Handle different model structures
                    if method_used == "direct_xtts":
                        # Direct XTTS model
                        tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    elif method_used == "tts_api":
                        # TTS API has synthesizer wrapper
                        if hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'tts_model'):
                            tts_model.synthesizer.tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            print("⚠️  Could not find tts_model in TTS API object")
                    elif method_used == "synthesizer":
                        # Synthesizer has tts_model attribute
                        if hasattr(tts_model, 'tts_model'):
                            tts_model.tts_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            print("⚠️  Could not find tts_model in Synthesizer object")
                    
                    print("✅ Custom trained weights loaded successfully")
                else:
                    print("⚠️  No model_state_dict found in checkpoint")
                    
            except Exception as e:
                print(f"⚠️  Could not load custom weights: {e}")
        
        # Generate audio using the appropriate method
        print("🔄 Generating audio...")
        
        try:
            if method_used == "direct_xtts":
                # Direct XTTS model uses get_conditioning_latents + inference
                gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
                    audio_path=[REFERENCE_AUDIO_PATH]
                )
                out = tts_model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=0.7
                )
                audio_data = out["wav"]
                
            elif method_used == "tts_api":
                # TTS API uses tts method
                audio_data = tts_model.tts(
                    text=text,
                    speaker_wav=REFERENCE_AUDIO_PATH,
                    language=language
                )
                
            elif method_used == "synthesizer":
                # Synthesizer uses tts method
                audio_data = tts_model.tts(
                    text=text,
                    speaker_wav=REFERENCE_AUDIO_PATH,
                    language=language
                )
            else:
                raise ValueError("Unknown method used for loading model")
                
        except Exception as e:
            print(f"❌ Error during audio generation: {e}")
            raise e
        
        # Convert to numpy array if needed
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)
        elif torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()
        
        # Ensure correct dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Convert to WAV bytes
        audio_bytes = io.BytesIO()
        
        # XTTS outputs at 24kHz, but check if different
        sample_rate = 24000 if method_used == "direct_xtts" else 22050
        sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
        audio_bytes.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode('utf-8')
        
        print(f"✅ Audio generated successfully ({len(audio_base64)} chars) using {method_used}")
        return audio_base64
        
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    finally:
        # Cleanup
        try:
            if 'cache_dir' in locals() and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear model references
            if 'tts_model' in locals():
                del tts_model
            if 'audio_data' in locals():
                del audio_data
            if 'config' in locals():
                del config
                
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {cleanup_error}")