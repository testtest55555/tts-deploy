import modelbit, sys
import base64

def text_to_speech(text, model_path='/content/best_model.pth'):
    """Convert text to speech using TTS CLI and return base64 encoded audio"""
    import base64
    import subprocess
    import tempfile
    import os

    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        output_path = temp_file.name

    # Run TTS CLI command
    cmd = f'tts --text "{text}" --model_path {model_path} --config_path training_output/config.json --out_path {output_path}'
    subprocess.run(cmd, shell=True, check=True)

    # Read the audio file and convert to base64
    with open(output_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    # Clean up temporary file
    os.unlink(output_path)

    return audio_base64


# main function
def generate_speech(text: str) -> str:
    """Generate speech from text and return base64 encoded audio"""
    return text_to_speech(text)

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = generate_speech(...)
#   print(result)