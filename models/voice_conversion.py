import torch
import torch.nn as nn

class VoiceConversionModel(nn.Module):
    def __init__(self):
        super(VoiceConversionModel, self).__init__()
        self.encoder = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.decoder = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

    def forward(self, audio_features, emotion_embedding):
        # Combine audio features and emotion embedding
        combined_input = torch.cat((audio_features, emotion_embedding), dim=1)
        encoded, _ = self.encoder(combined_input)
        decoded, _ = self.decoder(encoded)
        return decoded

    @staticmethod
    def apply_emotion(audio_features, emotion_embedding):
        model = VoiceConversionModel()
        return model.forward(audio_features, emotion_embedding)

    @staticmethod
    def load(model_path):
        model = VoiceConversionModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
