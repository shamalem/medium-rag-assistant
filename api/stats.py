from http.server import BaseHTTPRequestHandler
import json

CHUNK_SIZE =1000
OVERLAP_RATIO = 0.2
TOP_K = 7


class handler(BaseHTTPRequestHandler):

    def do_GET(self):

        response = {
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(json.dumps(response).encode("utf-8"))
