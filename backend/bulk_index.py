"""
Bulk index all Obsidian vault files into the RAG system.
Run: cd /Users/alex/rag-eval-system/backend && source .venv/bin/activate && python3 bulk_index.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")

from rag_engine_v2 import RAGEngineV2

VAULT_PATH = os.path.expanduser("~/Obsidian Vault")
CHROMA_PERSIST = os.path.join(os.path.dirname(__file__), "chroma_db")

# Files/directories to SKIP
SKIP_DIRS = {"_archive", ".git", "__pycache__", "node_modules", ".venv", "raw"}
SKIP_FILES = {
    "SCHEMA.md", "index.md", "log.md",  # structural vault files (not personal data)
    "agent.db", "_slam_map.json",  # databases
}

# Content type mapping for adaptive chunking
CONTENT_TYPES = {
    "trading": "trading",
    "trade": "trading",
    "market": "trading",
    "faith": "general",
    "spiritual": "general",
    "bible": "general",
    "prayer": "general",
    "project": "technical",
    "technical": "technical",
    "code": "technical",
    "engineering": "technical",
    "job": "general",
    "career": "general",
    "resume": "general",
    "session": "general",
    "mistake": "general",
    "memory": "general",
}


def get_content_type(filename, dirpath):
    """Determine content type from file path and name."""
    full = (dirpath + "/" + filename).lower()
    for keyword, ctype in CONTENT_TYPES.items():
        if keyword in full:
            return ctype
    return "general"


def should_skip(path, filename):
    """Check if file should be skipped."""
    if filename in SKIP_FILES:
        return True
    if filename.startswith(".") and filename != ".env":
        return True
    for skip_dir in SKIP_DIRS:
        if skip_dir in path:
            return True
    if not filename.endswith(".md"):
        return True
    return False


def scan_vault():
    """Scan all markdown files in the Obsidian vault."""
    files = []
    for root, dirs, filenames in os.walk(VAULT_PATH):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".env"]
        for fname in filenames:
            if should_skip(root, fname):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, VAULT_PATH)
            content_type = get_content_type(fname, root)
            files.append({
                "path": full_path,
                "rel_path": rel_path,
                "filename": fname,
                "content_type": content_type,
            })
    return files


def read_file_content(filepath):
    """Read file content, handling encoding issues."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def clean_markdown(text):
    """Clean markdown for better indexing — remove wikilinks, frontmatter, etc."""
    lines = text.split("\n")
    cleaned = []
    in_frontmatter = False
    in_code_block = False

    for line in lines:
        # Skip YAML frontmatter
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue

        # Skip code blocks (keep content but mark it)
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        # Convert wikilinks to plain text: [[link]] → link
        import re
        line = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", line)

        # Remove HTML comments
        line = re.sub(r"<!--.*?-->", "", line, flags=re.DOTALL)

        # Remove empty lines and pure whitespace
        if line.strip():
            cleaned.append(line.strip())

    return "\n".join(cleaned)


def main():
    print("=" * 60)
    print("RAG Bulk Indexer — Obsidian Vault → ChromaDB")
    print("=" * 60)

    # Initialize engine
    engine = RAGEngineV2(persist_directory=CHROMA_PERSIST)

    # Scan vault
    files = scan_vault()
    print(f"\nFound {len(files)} markdown files in vault")

    # Group by content type
    by_type = {}
    for f in files:
        ct = f["content_type"]
        by_type.setdefault(ct, []).append(f)
    print("\nBy content type:")
    for ct, flist in sorted(by_type.items()):
        print(f"  {ct}: {len(flist)} files")

    # Index each file
    total_chunks = 0
    indexed = 0
    skipped = 0
    errors = 0

    print(f"\nIndexing {len(files)} files...\n")

    for i, file_info in enumerate(files, 1):
        filepath = file_info["path"]
        rel_path = file_info["rel_path"]
        content_type = file_info["content_type"]

        # Read content
        raw_content = read_file_content(filepath)
        if not raw_content or len(raw_content.strip()) < 50:
            skipped += 1
            continue

        # Clean content
        content = clean_markdown(raw_content)
        if len(content.strip()) < 50:
            skipped += 1
            continue

        # Create metadata
        metadata = {
            "source": rel_path,
            "filename": file_info["filename"],
            "content_type": content_type,
            "file_size": len(content),
        }

        try:
            # Rate-limit spacing
            if i > 1:
                time.sleep(0.5)

            chunks = engine.index_text(
                content,
                content_type=content_type,
                metadata=metadata
            )
            total_chunks += chunks
            indexed += 1

            if i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {indexed} indexed, {skipped} skipped, {errors} errors, {total_chunks} chunks")

        except Exception as e:
            errors += 1
            print(f"  ERROR [{i}/{len(files)}] {rel_path}: {str(e)[:80]}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"INDEXING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files found:    {len(files)}")
    print(f"  Indexed:        {indexed}")
    print(f"  Skipped:        {skipped} (too small or unreadable)")
    print(f"  Errors:         {errors}")
    print(f"  Total chunks:   {total_chunks}")
    print(f"\nRAG system now has {total_chunks} searchable chunks from your life.")
    print(f"Query at: http://localhost:8000/query")


if __name__ == "__main__":
    main()
