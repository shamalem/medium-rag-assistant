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
</head>
<body>

<h2>Medium RAG Assistant</h2>

<form method="POST">
    <textarea name="question"
              rows="4"
              cols="70"
              placeholder="Ask a question..."></textarea>
    <br><br>
    <button type="submit">Ask</button>
</form>

{% if answer %}

<h3>Answer</h3>
<div style="
    white-space: pre-wrap;
    border: 1px solid #ccc;
    padding: 12px;
    margin-bottom: 20px;
">
{{ answer }}
</div>

<h3>Retrieved Context</h3>
<pre style="
    white-space: pre-wrap;
    border: 1px solid #ccc;
    padding: 12px;
">
{{ context }}
</pre>

<h3>Augmented Prompt</h3>
<pre style="
    white-space: pre-wrap;
    border: 1px solid #ccc;
    padding: 12px;
">
{{ augmented_prompt }}
</pre>

{% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    answer = None
    context = None
    augmented_prompt = None

    if request.method == "POST":

        question = request.form.get("question")

        try:

            res = requests.post(
                PROMPT_URL,
                json={"question": question}
            )

            data = res.json()

            answer = data.get("response", "")

            context = json.dumps(
                data.get("context", []),
                indent=2,
                ensure_ascii=False
            )

            augmented_prompt = json.dumps(
                data.get("Augmented_prompt", {}),
                indent=2,
                ensure_ascii=False
            )

        except Exception as e:

            answer = f"Error: {str(e)}"

    return render_template_string(
        HTML,
        answer=answer,
        context=context,
        augmented_prompt=augmented_prompt
    )


if __name__ == "__main__":
    app.run(debug=True)
