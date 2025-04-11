import os
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
from transformers import pipeline

from config import SAMPLE_RATE, DATABASE_DIR, SPEAKER_THRESHOLD

spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
asr = pipeline("automatic-speech-recognition", model="HuangJordan/whisper-base-chinese-cer")

def encode_audio(path):
    emb = spk_model.encode_batch(spk_model.load_audio(path))
    return emb.squeeze().detach().cpu().numpy().flatten()

def build_database():
    db = {}
    for f in os.listdir(DATABASE_DIR):
        if f.endswith(".wav"):
            name = os.path.splitext(f)[0]
            db[name] = encode_audio(os.path.join(DATABASE_DIR, f))
    return db

def recognize_speaker(embedding, database, threshold=SPEAKER_THRESHOLD):
    best_match = None
    best_score = 1
    for name, db_emb in database.items():
        score = cosine(embedding, db_emb)
        if score < best_score:
            best_score = score
            best_match = name
    if best_score < threshold:
        return best_match, 1 - best_score
    return "Unknown", 1 - best_score

def transcribe(path):
    return asr(path)["text"]