"""One-shot ingestion script — reads from MongoDB, pushes vectors to Qdrant cloud.

Run from the services/chat-service directory:
    python run_ingest.py
"""

import asyncio
import json
import sys
import os

# Load .env before importing anything that reads settings
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Import after env is loaded
    from app.core.mongo import get_client, get_db
    from app.rag.qdrant_client import bootstrap_collections, get_qdrant_status
    from app.rag.ingest_sync import ingest_all
    from app.config import get_settings

    settings = get_settings()

    print(f"Qdrant URL : {settings.qdrant_url}")
    print(f"Mongo URI  : {settings.mongo_uri[:40]}...")
    print(f"Mongo DB   : {settings.mongo_db}")
    print()

    # Check Qdrant reachability
    print("Checking Qdrant connection...")
    status = await get_qdrant_status()
    if status != "healthy":
        print(f"ERROR: Qdrant is {status}. Check QDRANT_URL and QDRANT_API_KEY.")
        sys.exit(1)
    print("Qdrant: healthy")

    # Connect Mongo
    print("Connecting to MongoDB...")
    client = get_client()
    db = get_db()
    collections = await db.list_collection_names()
    print(f"MongoDB: connected ({len(collections)} collections)")
    print()

    # Bootstrap collections (idempotent)
    print("Bootstrapping Qdrant collections...")
    await bootstrap_collections()
    print("Collections ready: content_public, content_private")
    print()

    # Run full ingestion
    print("Starting ingestion: servers, channels, threads, reels, users, messages, notifications...")
    print("-" * 60)
    result = await ingest_all()

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    total = sum(r.get("processed", 0) for r in result.get("results", []))
    print(f"\nTotal points upserted to Qdrant: {total}")

    # Close Mongo connection
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
