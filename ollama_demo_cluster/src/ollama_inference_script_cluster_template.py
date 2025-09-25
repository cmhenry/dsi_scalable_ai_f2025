#!/usr/bin/env python3
"""
Sanity-check for Ollama v0.9.x server via /v1/completions.
"""

import os
import sys
import time
import requests
import ollama


# Tell the client where your server is
os.environ["OLLAMA_API_URL"] = "http://127.0.0.1:11434"

# Pull the model you need
print("Pulling llama3.2:3b …")
ollama.pull("llama3.2:3b")
print("Done.")

# Wait for 5 seconds to ensure the model is ready
time.sleep(5)

resp = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role":"user","content":"Why is the sky blue?"}]
)
print(resp.message.content)

# End Script Gracefully
print("Script completed successfully.")
