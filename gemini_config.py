from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

def get_gemini_reply(user_text, speaker, history):
    prompt = f"你正在與使用者 {speaker} 對話。以下是歷史紀錄：\n{history}\n使用者現在說：{user_text}\n請用繁體中文回應："
    response = model.generate_content(prompt)
    return response.text.strip()
