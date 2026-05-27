from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI
from pinecone import Pinecone


TOP_K = 7

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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response = {
            "response": "prompt endpoint is working",
            "context": [],
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": ""
            }
        }

        self.wfile.write(json.dumps(response).encode("utf-8"))
