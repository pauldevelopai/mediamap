"""Asset management API routes"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import pathlib
from datetime import datetime

from ..db import get_db
from ..models import Asset
from ..schemas import AssetCreate, AssetOut
from ..settings import get_settings
from ..services.assets import (
    get_reference_frames,
    get_style_templates,
    create_style_template,
    create_asset_library
)

router = APIRouter()
settings = get_settings()

@router.get("/", response_model=List[AssetOut])
def list_assets(kind: str = None, db: Session = Depends(get_db)):
    """List all assets, optionally filtered by kind"""
    query = db.query(Asset)
    if kind:
        query = query.filter(Asset.kind == kind)
    assets = query.order_by(Asset.created_at.desc()).all()
    return assets

@router.post("/", response_model=AssetOut)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db)):
    """Create a new asset record"""
    # Verify file exists
    if not pathlib.Path(asset.path).exists():
        raise HTTPException(status_code=400, detail="Asset file does not exist")
    
    db_asset = Asset(**asset.dict())
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    return db_asset

@router.post("/upload")
def upload_asset(
    file: UploadFile = File(...),
    kind: str = "general",
    label: str = None,
    db: Session = Depends(get_db)
):
    """Upload an asset file"""
    if not label:
        label = file.filename
    
    # Create assets directory
    assets_dir = pathlib.Path(settings.MEDIA_ROOT) / "assets" / kind
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_path = assets_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create asset record
    db_asset = Asset(
        kind=kind,
        label=label,
        path=str(file_path),
        meta={
            "original_filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    )
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    
    return {"asset_id": db_asset.id, "path": str(file_path), "message": "Asset uploaded successfully"}

@router.get("/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: int, db: Session = Depends(get_db)):
    """Get a specific asset"""
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.delete("/{asset_id}")
def delete_asset(asset_id: int, db: Session = Depends(get_db)):
    """Delete an asset and its file"""
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # Delete file if it exists
    try:
        file_path = pathlib.Path(asset.path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Failed to delete asset file: {e}")
    
    # Delete database record
    db.delete(asset)
    db.commit()
    
    return {"message": "Asset deleted successfully"}

@router.get("/reference_frames/list")
def list_reference_frames(episode_id: int = None):
    """List available reference frames"""
    frames = get_reference_frames(episode_id)
    return {"reference_frames": frames}

@router.get("/style_templates/list")
def list_style_templates():
    """List available style templates"""
    templates = get_style_templates()
    return {"style_templates": templates}

@router.post("/style_templates/create")
def create_style_template_endpoint(
    name: str,
    style_prompt: str,
    metadata: dict = None
):
    """Create a new style template"""
    try:
        template_path = create_style_template(name, style_prompt, metadata)
        return {"message": "Style template created", "path": template_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/library/init")
def initialize_asset_library():
    """Initialize the asset library directory structure"""
    try:
        base_path = pathlib.Path(settings.MEDIA_ROOT) / "assets"
        create_asset_library(str(base_path))
        return {"message": "Asset library initialized", "path": str(base_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/library/stats")
def get_library_stats(db: Session = Depends(get_db)):
    """Get asset library statistics"""
    stats = {}
    
    # Count assets by kind
    kinds = db.query(Asset.kind).distinct().all()
    for (kind,) in kinds:
        count = db.query(Asset).filter(Asset.kind == kind).count()
        stats[kind] = count
    
    # Calculate total storage usage
    total_size = 0
    all_assets = db.query(Asset).all()
    for asset in all_assets:
        try:
            file_path = pathlib.Path(asset.path)
            if file_path.exists():
                total_size += file_path.stat().st_size
        except:
            pass
    
    stats["total_assets"] = len(all_assets)
    stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return {"library_stats": stats}

