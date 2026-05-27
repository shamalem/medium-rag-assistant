from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI
from pinecone import Pinecone


TOP_K = 7

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "4UHRUIN-gpt-5-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "4UHRUIN-text-embedding-3-small")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-rag")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you (metadata
and article passages). You must not use any external knowledge, the open
internet, or information that is not explicitly contained in the retrieved
context. If the answer cannot be determined from the provided context,
respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or
paraphrasing the relevant article passage or metadata when helpful.
"""


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        question = data.get("question", "")

        embedding_response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=question
        )

        question_vector = embedding_response.data[0].embedding

        results = pinecone_index.query(
            vector=question_vector,
            top_k=TOP_K,
            include_metadata=True
        )

        context = []

        for match in results["matches"]:
            metadata = match["metadata"]

            context.append({
                "article_id": metadata.get("article_id", ""),
                "title": metadata.get("title", ""),
                "chunk": metadata.get("chunk", ""),
                "score": match["score"]
            })

        context_text = ""

        for i, item in enumerate(context, start=1):
            context_text += f"""
Context chunk {i}
Article ID: {item["article_id"]}
Title: {item["title"]}
Passage:
{item["chunk"]}
"""

        user_prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context_text}

Question:
{question}
"""

        chat_response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        final_answer = chat_response.choices[0].message.content

        response = {
            "response": final_answer,
            "context": context,
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt
            }
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
