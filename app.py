from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("https://probable-bassoon-q796j5rvgrvv2w7r-8000.app.github.dev/"))

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_audio(video_path):
    clip = VideoFileClip(video_path)
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    audio_path = extract_audio(video_path)

    with open(audio_path, "rb") as audio_file:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sleep expert. Analyze snoring or sleep sounds and give specific advice to improve sleep quality."},
                {"role": "user", "content": "Here is the sleep audio recording."}
            ],
            audio=audio_file
        )

    text = response.choices[0].message["content"]
    return jsonify({"advice": text})

if __name__ == "__main__":
    app.run(debug=True)
