import modelbit, sys
from TTS.utils.synthesizer import Synthesizer
import torch
import io
import torchaudio
import base64

synthesizer = modelbit.load_value("data/synthesizer.pkl") # Synthesizer( (tts_model): Tacotron2( (embedding): Embedding(67, 512, padding_idx=0) (encoder): Encoder( (convolutions): ModuleList( (0-2): 3 x ConvBNBlock( (convolution1d): Conv1d(512, 512, kernel_siz...

# main function
def generate_speech(text: str) -> str:
    with torch.no_grad():
        # Generate audio using synthesizer
        wav = synthesizer.tts(text)
        
        # Convert numpy array to tensor
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        
        # Convert audio to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor, sample_rate=22050, format="wav")
        audio_bytes = buffer.getvalue()
        
        # Convert bytes to base64 string
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        return base64_audio

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = generate_speech(...)
#   print(result)