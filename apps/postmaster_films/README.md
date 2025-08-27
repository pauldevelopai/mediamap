# 🎬 Postmaster Films - AI TV Studio

End-to-end video production pipeline with AI models, budget management, and asset reuse - integrated into MediaMap.

## Features

### 🚀 Core Pipeline
- **Script → Video**: Automated scene generation from raw scripts
- **Budget Management**: Smart routing between Veo 3 Fast ($0.40/sec) and free AnimateDiff/SVD
- **Multi-Model Support**: Veo 3 Fast via Gemini API + AnimateDiff/SVD via ComfyUI
- **Asset Reuse**: Reference frames, backgrounds, style templates
- **Job Queue**: RQ + Redis for background processing
- **Assembly & Post**: FFmpeg video assembly, ElevenLabs TTS, audio mixing

### 💰 Budget Routing
- HERO scenes → Veo 3 Fast (if budget available)
- FILLER scenes → AnimateDiff/SVD (free)
- Real-time budget tracking and allocation
- Cost estimation and remaining seconds calculator

### 🎛️ Management Interface
- **Streamlit Console**: Full production management UI
- **FastAPI Backend**: RESTful API for all operations
- **Asset Library**: Reference frames, style templates, backgrounds
- **Job Monitoring**: Real-time job status and progress tracking

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install fastapi uvicorn sqlalchemy pydantic redis rq python-dotenv opencv-python-headless streamlit pandas requests

# System requirements
# - ffmpeg (for video processing)
# - Redis (for job queue)
```

### 2. Environment Setup

Copy and customize the environment file:

```bash
# In your .env file:
DATABASE_URL=sqlite:///./mediamap_postmaster.db
REDIS_URL=redis://localhost:6379/0
MEDIA_ROOT=./_postmaster_media

# Optional: Enable real model APIs
USE_VEO=true
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_REGION=us-central1
USE_TTS=true
ELEVENLABS_API_KEY=your_elevenlabs_key

# ComfyUI for AnimateDiff/SVD
COMFYUI_BASE_URL=http://localhost:8188
```

### 3. Start Services

```bash
# Terminal 1: Start API
cd /path/to/mediamap
uvicorn apps.postmaster_films.backend.main:app --reload --port 8000

# Terminal 2: Start Worker (optional, jobs run sync otherwise)
python -m apps.postmaster_films.worker.worker

# Terminal 3: Start Streamlit UI
streamlit run apps/postmaster_films/ui/postmaster_films_app.py --server.port 8501
```

### 4. Create Your First Episode

1. Open Streamlit UI: http://localhost:8501
2. Go to "Create Episode"
3. Create a project (if none exist)
4. Paste your script content
5. Set budget (e.g., $50 = 125 seconds of Veo)
6. Click "Create Episode from Script"

### 5. Production Pipeline

1. **Render Scenes**: Click "Render All Scenes" 
2. **Assemble Episode**: Click "Assemble Episode"
3. **Add Voiceover**: Click "Add Voiceover" 
4. **Download**: Final video saved in `_postmaster_media/projects/{project_id}/episodes/`

## Architecture

### Backend (`apps/postmaster_films/backend/`)
- **FastAPI**: RESTful API with automatic OpenAPI docs
- **SQLAlchemy**: Models for projects, episodes, scenes, jobs, assets
- **Services**: Video generation, audio processing, asset management
- **Budget Router**: Smart model selection based on scene type and budget

### Worker (`apps/postmaster_films/worker/`)
- **RQ Worker**: Background job processing
- **Async Pipeline**: Scene rendering, episode assembly, voiceover mixing
- **Fallback**: Synchronous processing if Redis unavailable

### UI (`apps/postmaster_films/ui/`)
- **Streamlit**: Production management console
- **Real-time**: Job monitoring and progress tracking
- **Asset Management**: Upload, organize, and reuse assets

### Services Architecture

```
Script Input
     ↓
Scene Generation (shotlist.py)
     ↓
Budget Routing (router_budget.py)
     ↓
┌─ HERO Scenes → Veo 3 Fast (veo.py)
└─ FILLER Scenes → AnimateDiff (animdiff.py)
     ↓
Asset Management (assets.py)
     ↓
Video Assembly (ffmpeg_tools.py)
     ↓
Audio Processing (audio.py)
     ↓
Final Output
```

## API Documentation

Start the API and visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints

- `POST /episodes/from_script` - Create episode from script
- `POST /jobs/render_episode/{id}` - Render all scenes
- `POST /jobs/assemble/{id}` - Assemble final video
- `POST /jobs/mux_vo/{id}` - Add voiceover
- `GET /episodes/{id}/budget` - Budget information

## Configuration

### Model Configuration

```python
# Enable/disable AI models
USE_VEO = True  # Veo 3 Fast via Gemini
USE_TTS = True  # ElevenLabs TTS

# Budget settings
EPISODE_VEO_BUDGET_USD = 50.0
VEO_PRICE_PER_SEC = 0.40

# ComfyUI for open source models
COMFYUI_BASE_URL = "http://localhost:8188"
```

### Storage Configuration

```python
# Local storage (default)
MEDIA_ROOT = "./_postmaster_media"

# Optional: S3/GCS (implement in storage service)
# STORAGE_BACKEND = "s3"
# S3_BUCKET = "postmaster-media"
```

## Development

### Running Tests

```bash
cd apps/postmaster_films
python -m pytest tests/test_minimal.py -v
```

### Docker Development

```bash
# Start full stack with Docker
docker-compose -f docker-compose.postmaster.yml up --build

# Services available at:
# - API: http://localhost:8000
# - UI: http://localhost:8501
# - Redis: localhost:6379
```

### Adding New Models

1. Create service in `services/` (e.g., `new_model.py`)
2. Add routing logic in `router_budget.py`
3. Update job processor in `services/jobs.py`
4. Add model configuration to `settings.py`

## Production Deployment

### Environment Variables

```bash
# Production settings
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@host:5432/postmaster
REDIS_URL=redis://prod-redis:6379/0

# Google Cloud (required for Veo 3 Fast)
GOOGLE_CLOUD_PROJECT=your_production_project
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# API Keys (required for full functionality)
ELEVENLABS_API_KEY=your_production_key

# Storage
MEDIA_ROOT=/var/postmaster_media
# or
S3_BUCKET=postmaster-production
```

### Scaling

- **API**: Scale FastAPI with multiple uvicorn workers
- **Workers**: Scale RQ workers across multiple containers/machines
- **Storage**: Use S3/GCS for media files in production
- **Database**: Use PostgreSQL for production workloads

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install ffmpeg system package
2. **Redis connection failed**: Jobs will run synchronously 
3. **ComfyUI unavailable**: AnimateDiff falls back to mock videos
4. **API keys missing**: Models fall back to placeholder generation

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Check health endpoint
curl http://localhost:8000/health

# Monitor job queue
# Visit Streamlit UI → Job Monitor
```

### Performance Tips

1. **Parallel Processing**: Run multiple workers for scene rendering
2. **Asset Reuse**: Use reference frames for continuity
3. **Budget Optimization**: Reserve Veo for HERO scenes only
4. **Caching**: Implement asset caching for repeated elements

## Integration with MediaMap

Postmaster Films is designed as a modular app within MediaMap:

- **Shared Authentication**: Uses MediaMap's user system
- **Database Integration**: Can share database or use separate one
- **Asset Sharing**: Can reference MediaMap's existing assets
- **UI Integration**: Can be embedded in MediaMap dashboard

## License

Part of MediaMap project - see main repository for license details.

## Support

For issues and feature requests, see the main MediaMap repository or contact the development team.
