import modelbit, sys
from content.tts_wrapper import TTSModel
import torch
import io
import torchaudio
import base64

# main function
def generate_speech(text: str) -> bytes:
    """Convert text to speech and return audio bytes"""
    model = TTSModel()
    model_state = torch.load("tts_model_state.pth")
    model.load_state_dict(model_state)
    
    with torch.no_grad():
        wav = model(text)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor, sample_rate=22050, format="wav")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = generate_speech(...)
#   print(result)