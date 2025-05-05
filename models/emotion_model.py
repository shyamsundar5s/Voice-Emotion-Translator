import torch
import torch.nn as nn

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=4, embedding_dim=128)  # Four emotions: happy, sad, angry, neutral

    def forward(self, emotion_label):
        return self.embedding_layer(emotion_label)

    @staticmethod
    def get_emotion_embedding(emotion_name):
        emotion_map = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}
        emotion_label = torch.tensor(emotion_map[emotion_name])
        return EmotionModel().forward(emotion_label)

    @staticmethod
    def load(model_path):
        model = EmotionModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
