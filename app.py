from flask import Flask, request, jsonify
from models.emotion_model import EmotionModel
from models.voice_conversion_model import VoiceConversionModel
from utils.audio_utils import save_audio, load_audio

app = Flask(__name__)

# Load pre-trained models
emotion_model = EmotionModel.load("models/emotion_model_weights.pth")
voice_model = VoiceConversionModel.load("models/voice_conversion_weights.pth")

@app.route('/translate', methods=['POST'])
def translate():
    # Get input audio and target emotion
    audio_file = request.files['audio']
    target_emotion = request.form.get('emotion')

    # Save and preprocess audio
    input_audio_path = save_audio(audio_file)
    audio_data = load_audio(input_audio_path)

    # Apply emotion translation
    emotion_embedding = emotion_model.get_emotion_embedding(target_emotion)
    translated_audio = voice_model.apply_emotion(audio_data, emotion_embedding)

    # Save output audio and return response
    output_audio_path = "output/translated_audio.wav"
    translated_audio.save(output_audio_path)
    return jsonify({"success": True, "output_audio": output_audio_path})

if __name__ == '__main__':
    app.run(debug=True)
