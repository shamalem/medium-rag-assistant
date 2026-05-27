from flask import Flask, request, jsonify, render_template_string
import requests

app = Flask(__name__)

PROMPT_URL = "https://medium-rag-assistant-kappa.vercel.app/api/prompt"

HTML = """
<h2>Medium RAG Assistant</h2>

<form method="POST">
  <textarea name="question" rows="4" cols="70" placeholder="Ask a question..."></textarea><br><br>
  <button type="submit">Ask</button>
</form>

{% if answer %}
<h3>Answer:</h3>
<pre>{{ answer }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None

    if request.method == "POST":
        question = request.form.get("question")

        res = requests.post(
            PROMPT_URL,
            json={"question": question}
        )

        answer = res.text

    return render_template_string(HTML, answer=answer)
