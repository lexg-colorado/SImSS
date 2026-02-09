# SImSS - Semantic Image Search System

A local-first semantic image search system using vision-language models. SImSS automatically describes, tags, and indexes your images for natural language querying.

Point it at a directory of photos, and it handles the rest—resizing for inference, extracting metadata, generating descriptions with AI, and building a searchable vector index.

> **Looking for the LM Studio version?** See [SImSS-LMSTUDIO](https://github.com/lexg-colorado/SImSS-LMSTUDIO).

## Key Features

- **Local-First**: All processing happens on your machine. No cloud uploads, no API costs, complete privacy.
- **Vision-Language AI**: Uses Ollama with qwen3-vl for intelligent image descriptions and nomic-embed-text for semantic embeddings.
- **Natural Language Search**: Find images by describing what you're looking for ("sunset at the beach", "group photo at birthday party").
- **EXIF Metadata Extraction**: Automatically extracts date taken, GPS coordinates, and camera information.
- **Multi-Format Support**: Handles JPEG, PNG, HEIC/HEIF, and RAW formats (CR2, NEF, ARW, DNG).
- **Resumable Processing**: Stop and restart anytime—SImSS picks up where it left off.
- **Change Detection**: Only reprocesses images that have actually changed.
- **Graceful Shutdown**: Ctrl+C safely stops processing without data corruption.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- ~8GB RAM recommended for vision model inference

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lexg-colorado/SImSS.git
   cd SImSS
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and pull required models**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull qwen3-vl:8b      # Vision model
   ollama pull nomic-embed-text  # Embedding model
   ```

5. **Initialize SImSS**
   ```bash
   python -m sims init
   ```

## Quick Start

```bash
# 1. Initialize the database and vector store
python -m sims init

# 2. Ingest your photos
python -m sims ingest ~/Photos

# 3. Search for images
python -m sims search "sunset over mountains"
```

## CLI Commands

### Initialize
```bash
python -m sims init
```
Creates the SQLite database and ChromaDB vector store. Run this once before first use.

### Ingest Images
```bash
# Process all images in a directory (recursive)
python -m sims ingest /path/to/photos

# Process only top-level directory (no subdirectories)
python -m sims ingest /path/to/photos --no-recursive

# Custom batch size
python -m sims ingest /path/to/photos --batch-size 5

# Use a different vision model (default: qwen3-vl:8b)
python -m sims ingest /path/to/photos --vision-model qwen3-vl:4b

# Quiet mode (no progress output)
python -m sims ingest /path/to/photos --quiet
```

### Resume Processing
```bash
# Resume any incomplete processing
python -m sims resume

# Resume with a specific vision model
python -m sims resume --vision-model qwen3-vl:4b
```
Useful if ingestion was interrupted or if you want to retry failed images.

### Search Images
```bash
# Basic search
python -m sims search "beach sunset"

# Limit results
python -m sims search "family gathering" --limit 10

# Filter by minimum match score (only 60%+ matches)
python -m sims search "sunset over water" --min-score 60

# Filter by GPS (only images with location data)
python -m sims search "mountain landscape" --has-gps

# Filter by date
python -m sims search "birthday party" --date-after 2023-01-01 --date-before 2023-12-31

# Combine filters
python -m sims search "vacation photos" --min-score 50 --has-gps --limit 20

# JSON output
python -m sims search "city skyline" --json

# Verbose output (shows descriptions, tags, metadata)
python -m sims search "dog playing" -v
```

### View Statistics
```bash
# Show system statistics
python -m sims stats

# JSON format
python -m sims stats --json
```

### Check Job Status
```bash
# Check status of a specific ingestion job
python -m sims status 1
```

### Reset Failed Images
```bash
# Reset error state for failed images to allow retry
python -m sims reset-errors

# Then resume processing
python -m sims resume
```

### Cleanup Orphaned Entries
```bash
# Remove database entries for images whose files no longer exist
python -m sims cleanup

# Skip confirmation prompt
python -m sims cleanup --force
```
This is useful when you've moved or deleted original image files and want to clean up stale database entries.

## How It Works

### Ingestion Pipeline

1. **Discovery**: Recursively scans directories for supported image formats, computes file hashes, and registers new/changed images in SQLite.

2. **Thumbnail Generation**: Creates 768px thumbnails for efficient processing, stored in cache with content-based naming.

3. **Metadata Extraction**: Parses EXIF data for date taken, GPS coordinates, camera make/model.

4. **Vision Description**: Sends thumbnails to qwen3-vl which generates structured descriptions including:
   - Scene description
   - Mood/atmosphere
   - Tags (5-10 keywords)
   - Dominant colors
   - Time of day

5. **Vector Embedding**: Converts descriptions to 768-dimensional vectors using nomic-embed-text and stores in ChromaDB.

### Search Process

1. Your query is converted to a vector embedding
2. ChromaDB finds the most similar image embeddings (cosine similarity)
3. Results are enriched with full metadata from SQLite
4. Ranked results returned with similarity scores

## Configuration

SImSS uses environment variables for configuration. Defaults work for most setups:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMS_HOME` | `/home/user/sims` | Base directory for SImSS data |
| `SIMS_CACHE_DIR` | `$SIMS_HOME/cache` | Thumbnail cache location |
| `SIMS_DB_PATH` | `$SIMS_HOME/data/sims.db` | SQLite database path |
| `SIMS_CHROMA_PATH` | `$SIMS_HOME/data/chroma` | ChromaDB storage path |
| `SIMS_OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `SIMS_VISION_MODEL` | `qwen3-vl:8b` | Vision model name |
| `SIMS_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `SIMS_BATCH_SIZE` | `10` | Images per processing batch |

### Vision Model Selection

You can switch vision models at runtime using the `--vision-model` flag:

```bash
# Use a smaller/faster model for quick processing
python -m sims ingest ~/Photos --vision-model qwen3-vl:4b

# Or set via environment variable for persistent override
export SIMS_VISION_MODEL=qwen3-vl:4b
python -m sims ingest ~/Photos
```

The CLI flag takes precedence over the environment variable.

## Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| JPEG | `.jpg`, `.jpeg` | Full support |
| PNG | `.png` | Full support |
| HEIC/HEIF | `.heic`, `.heif` | Requires pillow-heif |
| Canon RAW | `.cr2` | Requires rawpy |
| Nikon RAW | `.nef` | Requires rawpy |
| Sony RAW | `.arw` | Requires rawpy |
| Adobe DNG | `.dng` | Requires rawpy |

## Performance

Typical throughput on consumer hardware:

| Stage | Time per Image |
|-------|----------------|
| Thumbnail | ~0.1-0.5s (slower for RAW) |
| Vision Model | ~2-5s (GPU), ~10-30s (CPU) |
| Embedding | ~0.05s |

Expected throughput: **~720 images/hour** with GPU acceleration.

## Development

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=sims tests/

# Run specific test file
pytest tests/test_agent.py -v
```

## Project Structure

```
SImSS/
├── sims/
│   ├── __init__.py      # Package init
│   ├── __main__.py      # CLI entry point
│   ├── agent.py         # Ingestion orchestration
│   ├── config.py        # Configuration management
│   ├── db.py            # SQLite operations
│   ├── embeddings.py    # Text embedding generation
│   ├── metadata.py      # EXIF extraction
│   ├── thumbnail.py     # Image resizing
│   ├── vectorstore.py   # ChromaDB operations
│   ├── vision.py        # Vision model interface
│   └── walker.py        # Directory traversal
├── tests/               # Test suite
├── requirements.txt     # Dependencies
└── README.md
```

## Author

**Lex Gaines** - [GitHub](https://github.com/lexg-colorado)

## License

Apache 2.0 License - see LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Pillow](https://pillow.readthedocs.io/) for image processing
