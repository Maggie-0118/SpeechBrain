import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3

from config import SAMPLE_RATE

def record_audio_free(filename):
    print("\U0001F449 按 Enter 開始錄音...")
    input()
    print("\U0001F3A4 錄音中... 講完後請按 Enter 停止")

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

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()