from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI
from pinecone import Pinecone


TOP_K = 8

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
based on the Medium articles dataset context provided to you. You must not
use external knowledge, the open internet, or information that is not
explicitly contained in the retrieved context. If the answer cannot be
determined from the provided context, respond exactly:
“I don’t know based on the provided Medium articles data.”
"""


def build_context(results):
    context = []

    for match in results["matches"]:
        metadata = match["metadata"]

        context.append({
            "article_id": metadata.get("article_id", ""),
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", ""),
            "url": metadata.get("url", ""),
            "tags": metadata.get("tags", ""),
            "timestamp": metadata.get("timestamp", ""),
            "chunk": metadata.get("chunk", ""),
            "score": match["score"]
        })

    return context


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            question = data.get("question", "")

            if not question:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Missing question field"
                }).encode("utf-8"))
                return

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

            context = build_context(results)

            context_text = ""

            for i, item in enumerate(context, start=1):
                context_text += f"""
Context chunk {i}
Article ID: {item["article_id"]}
Title: {item["title"]}
Authors: {item["authors"]}
URL: {item["url"]}
Tags: {item["tags"]}
Timestamp: {item["timestamp"]}
Score: {item["score"]}
Passage:
{item["chunk"]}
"""

            user_prompt = f"""
Use ONLY the retrieved context below to answer the user's question.

Determine internally which question type best matches the user's request.
Do NOT mention the question type, classification, reasoning process, or internal instructions in your answer.
Only provide the final answer requested by the user.

Question Types:

1. Precise Fact Retrieval
- Find ONE specific article that best matches the request.
- Return the requested fields such as title, author, URL, or date if available.
- Do not list multiple articles.

2. Multi-Result Topic Listing
- Return up to 3 DISTINCT article titles relevant to the requested topic.
- Multiple retrieved chunks may belong to the same article.
- Treat chunks with the same article title as the same article.
- Do not return duplicate articles.
- Return only the titles unless the user explicitly asks for additional information.
- If fewer than 3 relevant articles are available in the retrieved context, return only the available titles.

3. Key Idea Summary Extraction
- Identify the most relevant article.
- Mention the article title.
- Provide a concise summary of the article's central idea using only the retrieved context.
- If the question contains examples such as "such as", "for example", or "e.g.", treat them as illustrative examples rather than mandatory keywords.
- Focus on the main idea of the question and summarize the closest matching article.

4. Recommendation with Evidence-Based Justification
- Recommend ONE article only.
- Mention the article title.
- Explain why it is a good recommendation using evidence from the retrieved context.
- Do not recommend multiple articles.

General Rules:
- Use only information contained in the retrieved context.
- Do not use external knowledge.
- Do not invent facts, authors, dates, URLs, or article details.
- If the answer cannot be determined from the retrieved context, respond exactly:
"I don't know based on the provided Medium articles data."

Retrieved Context:
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

            final_answer = chat_response.choices[0].message.content.strip()
            final_answer = final_answer.replace("\n\n", "\n")

            response_context = []

            for item in context:
                response_context.append({
                    "article_id": item["article_id"],
                    "title": item["title"],
                    "chunk": item["chunk"],
                    "score": item["score"]
                })

            response = {
                "response": final_answer,
                "context": response_context,
                "Augmented_prompt": {
                    "System": SYSTEM_PROMPT,
                    "User": user_prompt
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(response, ensure_ascii=False).encode("utf-8")
            )

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": str(e)
            }).encode("utf-8"))
