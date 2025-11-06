from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key="YOUR_API_KEY_HERE")

SYSTEM_PROMPT = """
You are Luma, a friendly chatbot that gives sleep and wellness tips.
You are shift-aware — you give advice that fits day or night shifts.

If the user works NIGHT shifts:
- Talk about staying alert overnight and sleeping during the day.
- Suggest blackout curtains, regular rest, and hydration.

If the user works DAY shifts:
- Suggest morning sunlight, a balanced routine, and early bedtime.

Always sound kind, clear, and helpful.
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)
