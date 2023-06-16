import http.server
import socketserver
from pathlib import Path

PORT = 8000
DIRECTORY = Path(r"C:\Users\austi\dev\hsi_moss\stiff_outputs")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()