import os
import time
import sqlite3
import pyttsx3
import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import pipeline
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition

import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai

# === Gemini 設定 ===
genai.configure(api_key="AIzaSyAWiWCIFgk0njfDIw_7IBPkYWitOgoCEIY")
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# === 語音回應 ===
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# === 系統參數 ===
SAMPLE_RATE = 16000
DATABASE_DIR = "voice_db"
DB_FILE = "voice_log.db"

# === 模型載入 ===
print("[INFO] 載入模型中...")
asr = pipeline("automatic-speech-recognition", model="Jingmiao/whisper-small-chinese_base")
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("[INFO] 模型載入完成")

# === 錄音函式（無秒數限制）===
def record_audio_free(filename):
    print("👉 按 Enter 開始錄音...")
    input()
    print("🎤 錄音中... 講完後請按 Enter 停止")

    samplerate = SAMPLE_RATE
    channels = 1
    dtype = 'float32'
    recording = []

    stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype)
    stream.start()

    try:
        print("[錄音中... 持續錄音直到你按 Enter]")
        if os.name == 'nt':
            import msvcrt
            while True:
                data, _ = stream.read(1024)
                recording.append(data)
                if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                    break
        else:
            import sys, select
            while True:
                data, _ = stream.read(1024)
                recording.append(data)
                if select.select([sys.stdin], [], [], 0)[0]:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] 錄音中斷")

    stream.stop()
    stream.close()

    audio_data = np.concatenate(recording, axis=0)
    sf.write(filename, audio_data, samplerate)
    print(f"[INFO] 音檔儲存為：{filename}")

# === 將音檔轉成語者 embedding ===
def encode_audio(path):
    emb = spk_model.encode_batch(spk_model.load_audio(path))
    return emb.squeeze().detach().cpu().numpy().flatten()

# === 建立資料庫中 speaker 聲紋 ===
def build_database():
    db = {}
    for f in os.listdir(DATABASE_DIR):
        if f.endswith(".wav"):
            name = os.path.splitext(f)[0]
            db[name] = encode_audio(os.path.join(DATABASE_DIR, f))
    return db

# === 語者辨識 ===
def recognize_speaker(embedding, database, threshold=0.45):
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

# === Whisper 語音轉文字 ===
def transcribe(path):
    return asr(path)["text"]

# === Gemini 回應 ===
def get_gemini_reply(user_text, speaker):
    history = get_history(speaker)
    prompt = f"你正在與使用者 {speaker} 對話。以下是歷史紀錄：\n{history}\n使用者現在說：{user_text}\n請用中文回應："
    response = model.generate_content(prompt)
    return response.text.strip()

# === 查詢歷史對話 ===
def get_history(speaker):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transcript FROM voice_transcripts
        WHERE speaker=? ORDER BY timestamp 
    """, (speaker,))
    rows = cursor.fetchall()
    conn.close()
    history = "\n".join([row[0] for row in reversed(rows)])
    return history

# === 建立資料表（只需跑一次）===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker TEXT,
            confidence REAL,
            transcript TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# === 儲存紀錄 ===
def save_to_db(speaker, confidence, transcript):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO voice_transcripts (speaker, confidence, transcript)
        VALUES (?, ?, ?)
    """, (speaker, confidence, transcript))
    conn.commit()
    conn.close()

# === 主流程 ===
init_db()

def main():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    database = build_database()

    while True:
        temp_path = "temp_input.wav"
        record_audio_free(temp_path)

        test_emb = encode_audio(temp_path)
        speaker, confidence = recognize_speaker(test_emb, database)
        print(f"辨識結果：{speaker} (confidence：{confidence:.2f})")

        new_path = temp_path
        is_new_user = False

        if speaker == "Unknown" and confidence < 0.40:
            auto_index = 1
            while os.path.exists(os.path.join(DATABASE_DIR, f"auto_user_{auto_index}.wav")):
                auto_index += 1
            new_path = os.path.join(DATABASE_DIR, f"auto_user_{auto_index}.wav")
            os.rename(temp_path, new_path)
            database[f"auto_user_{auto_index}"] = encode_audio(new_path)
            print(f"偵測到新的使用者，增新 speaker: auto_user_{auto_index}")
            speaker = f"auto_user_{auto_index}"
            is_new_user = True
        elif speaker == "Unknown":
            print("偵測無效，不儲存")

        text = transcribe(new_path)
        print(f"語音內容：{text}")

        if is_new_user:
            welcome = f"很高興認識你，{speaker}！"
            speak(welcome)
            print(f"{welcome}")

        reply = get_gemini_reply(text, speaker)
        print(f"Gemini 回覆：{reply}")
        speak(reply)

        combined_text = f"User: {text} | Gemini: {reply}"
        save_to_db(speaker, confidence, combined_text)
        print("已儲存至資料庫\n")

        time.sleep(1)

if __name__ == "__main__":
    main()
