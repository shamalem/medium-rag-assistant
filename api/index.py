from flask import Flask, request, render_template_string
import requests
import json

app = Flask(__name__)

PROMPT_URL = "https://medium-rag-assistant-kappa.vercel.app/api/prompt"
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Medium RAG Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #ffffff;
            color: #111111;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 1000px;
            margin: auto;
        }

        h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }

        p {
            color: #555;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 14px;
            font-size: 15px;
            border: 1px solid #111;
            border-radius: 8px;
            resize: vertical;
        }

        button {
            margin-top: 14px;
            padding: 12px 24px;
            background: #111;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
        }

        button:hover {
            background: #333;
        }

        .card {
            margin-top: 30px;
            border: 1px solid #111;
            border-radius: 10px;
            padding: 20px;
            background: #fafafa;
        }

        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.5;
            margin: 0;
        }

        .label {
            font-weight: bold;
            margin-bottom: 12px;
            font-size: 18px;
        }
    </style>
</head>
<body>

<div class="container">

    <h1>Medium RAG Assistant</h1>
    <p>Ask a question and view the full JSON response.</p>

    <form method="POST">
        <textarea name="question" placeholder="Ask a question..."></textarea>
        <br>
        <button type="submit">Ask</button>
    </form>

    {% if output_json %}
    <div class="card">
        <div class="label">Output format (JSON)</div>
        <pre>{{ output_json }}</pre>
    </div>
    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    output_json = None

    if request.method == "POST":
        question = request.form.get("question")

        try:
            res = requests.post(
                PROMPT_URL,
                json={"question": question}
            )

            data = res.json()

            output_json = json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )

        except Exception as e:
            output_json = json.dumps(
                {"error": str(e)},
                indent=2,
                ensure_ascii=False
            )

    return render_template_string(
        HTML,
        output_json=output_json
    )


if __name__ == "__main__":
    app.run(debug=True)
