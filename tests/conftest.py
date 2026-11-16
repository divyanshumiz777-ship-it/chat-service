"""Shared test setup.

The app's background warm-up (Qdrant bootstrap, embedding-model load,
product-help indexing — app.main._warm_up) would otherwise start on every
TestClient and keep loading the model in a worker thread while other tests
stub it. tests/test_startup.py turns it back on explicitly.
"""

import os

os.environ.setdefault("STARTUP_WARMUP", "false")
