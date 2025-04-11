from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from config import DATABASE_DIR, NEW_USER_THRESHOLD
from recognizer import build_database, encode_audio, recognize_speaker, transcribe
from gemini_config import get_gemini_reply
from database import init_db, save_to_db, get_history
from gtts import gTTS
import os
import shutil
import time
from datetime import datetime
import subprocess

app = Flask(__name__)
CORS(app)  # 允許跨域呼叫
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs("response_audio", exist_ok=True)
init_db()

# 建立全域 speaker 資料庫
database = build_database()

@app.route("/upload", methods=["POST"])
def upload_audio():

    start_total = datetime.now()
    if 'file' not in request.files:
        return jsonify({"error": "未收到音檔"}), 400

    file = request.files['file']
    uploaded_path = "temp_input_orig.webm" 
    file.save(uploaded_path)

    #  將 webm 轉為 wav 格式（16kHz, mono）
    temp_path = "temp_input.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", uploaded_path,
        "-ar", "16000", "-ac", "1", temp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # === 語者辨識 ===
    start_asr = datetime.now()
    global database
    test_emb = encode_audio(temp_path)
    speaker, confidence = recognize_speaker(test_emb, database)
    new_path = temp_path
    end_asr = datetime.now()

    if speaker == "Unknown" and confidence < NEW_USER_THRESHOLD:
        auto_index = 1
        while os.path.exists(os.path.join(DATABASE_DIR, f"auto_user_{auto_index}.wav")):
            auto_index += 1
        new_path = os.path.join(DATABASE_DIR, f"auto_user_{auto_index}.wav")
        shutil.copy(temp_path, new_path)
        database[f"auto_user_{auto_index}"] = encode_audio(new_path)
        speaker = f"auto_user_{auto_index}"
    elif speaker == "Unknown":
        print("辨識失敗，請再試一次")

    # === Whisper 語音轉文字 ===
    user_text = transcribe(new_path)

    # === Gemini 回覆 ===
    start_ai= datetime.now()
    history = get_history(speaker)
    reply = get_gemini_reply(user_text, speaker, history)
    end_ai=datetime.now()

    # === Gemini 回覆轉為語音檔（儲存 .mp3）===
    start_ai_rec= datetime.now()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{speaker}_{timestamp}.mp3"
    audio_path = os.path.join("response_audio", filename)
    tts = gTTS(reply, lang="zh-tw")
    tts.save(audio_path)
    end_ai_rec= datetime.now()

    # === 儲存對話紀錄到 SQLite ===
    combined = f"User: {user_text} | Gemini: {reply}"
    save_to_db(speaker, confidence, combined)
    end_total = datetime.now()

    return jsonify({
        "speaker": speaker,
        "confidence": round(confidence, 2),
        "user_text": user_text,
        "reply": reply,
        "audio_url": f"/audio/{filename}",
        "time_asr": round((end_asr - start_asr).total_seconds(), 2),
        "time_ai": round((end_ai - start_ai).total_seconds(), 2),
        "time_ai_rec": round((end_ai_rec - start_ai_rec).total_seconds(), 2),
        "time_total": round((end_total - start_total).total_seconds(), 2),
    })

@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory("response_audio", filename)

if __name__ == "__main__":
    app.run(debug=True)
