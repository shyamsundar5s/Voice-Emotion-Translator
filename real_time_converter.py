import torch
import sounddevice as sd
from model import VITSModel  # Load the same model as used in training
from utils import preprocess_audio, postprocess_audio  # Audio preprocessing utilities

# Load the trained model
model = VITSModel()
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

def emotion_converter(input_stream, target_emotion):
    # Preprocess input stream
    audio_features = preprocess_audio(input_stream)

    # Convert emotion
    with torch.no_grad():
        converted_features = model(audio_features, target_emotion)

    # Postprocess and return the audio stream
    return postprocess_audio(converted_features)

if __name__ == "__main__":
    print("Starting real-time emotion translator...")
    target_emotion = "happy"  # Default emotion

    # Use the microphone as input and speaker as output
    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = emotion_converter(indata, target_emotion)

    with sd.Stream(callback=callback):
        print("Listening... Press Ctrl+C to stop.")
        sd.sleep(100000)
