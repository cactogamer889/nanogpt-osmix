"""
Osmix API Service - Universal API for all training and model operations
Super configurable API that adapts to all model types and categories
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Osmix API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = Path(__file__).parent
SYSTEM_DIR = BASE_DIR / "system"
DATASETS_DIR = BASE_DIR / "datasets"
FINE_TUNING_DIR = BASE_DIR / "fine-tunning"
PRODUCT_DIR = BASE_DIR / "product"

# Model categories structure
CATEGORIES = {
    "Multimodal": [
        "Audio-Text-to-Text",
        "Image-Text-to-Text",
        "Image-Text-to-Image",
        "Image-Text-to-Video",
        "Visual Question Answering",
        "Document Question Answering",
        "Video-Text-to-Text",
        "Visual Document Retrieval",
        "Any-to-Any"
    ],
    "Computer Vision": [
        "Depth Estimation",
        "Image Classification",
        "Object Detection",
        "Image Segmentation",
        "Text-to-Image",
        "Image-to-Text",
        "Image-to-Image",
        "Image-to-Video",
        "Unconditional Image Generation",
        "Video Classification",
        "Text-to-Video",
        "Zero-Shot Image Classification",
        "Mask Generation",
        "Zero-Shot Object Detection",
        "Text-to-3D",
        "Image-to-3D",
        "Image Feature Extraction",
        "Keypoint Detection",
        "Video-to-Video"
    ],
    "Natural Language Processing": [
        "Text Classification",
        "Token Classification",
        "Table Question Answering",
        "Question Answering",
        "Zero-Shot Classification",
        "Translation",
        "Summarization",
        "Feature Extraction",
        "Text Generation",
        "Fill-Mask",
        "Sentence Similarity",
        "Text Ranking"
    ],
    "Audio": [
        "Text-to-Speech",
        "Text-to-Audio",
        "Automatic Speech Recognition",
        "Audio-to-Audio",
        "Audio Classification",
        "Voice Activity Detection"
    ],
    "Tabular": [
        "Tabular Classification",
        "Tabular Regression",
        "Time Series Forecasting"
    ],
    "Reinforcement Learning": [
        "Reinforcement Learning"
    ],
    "Robotics": [
        "Robotics"
    ],
    "Other": [
        "Graph Machine Learning"
    ]
}

# Request models
class TrainingConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    category: str
    subcategory: str
    model_name: str
    batch_size: int = 12
    block_size: int = 1024
    gradient_accumulation_steps: int = 5
    learning_rate: float = 6e-4
    max_iters: int = 600000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    output_format: str = "ckpt"
    device: str = "cuda"
    dtype: str = "bfloat16"
    save_interval: int = 1000  # Save checkpoint every N iterations

class RetrainConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    model_path: str
    category: str
    subcategory: str
    additional_iters: int = 10000
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None

class CheckParamsRequest(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    model_path: str

# API Routes

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Osmix API Service",
        "version": "1.0.0",
        "endpoints": {
            "categories": "/api/categories",
            "train": "/api/train",
            "check_params": "/api/check_params",
            "retrain": "/api/retrain",
            "models": "/api/models"
        }
    }

@app.get("/api/categories")
async def get_categories():
    """Get all available categories and subcategories"""
    return {
        "categories": CATEGORIES,
        "status": "success"
    }

@app.get("/api/categories/{category}")
async def get_category_subcategories(category: str):
    """Get subcategories for a specific category"""
    if category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    return {
        "category": category,
        "subcategories": CATEGORIES[category],
        "status": "success"
    }

@app.get("/api/categories/{category}/{subcategory}/status")
async def get_subcategory_status(category: str, subcategory: str):
    """Check if a subcategory has training script available"""
    script_path = SYSTEM_DIR / category / subcategory / "train_osmix.py"
    has_script = script_path.exists()
    
    return {
        "category": category,
        "subcategory": subcategory,
        "available": has_script,
        "status": "available" if has_script else "coming_soon"
    }

@app.post("/api/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start training with custom configuration"""
    # Validate category and subcategory
    if config.category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: {config.category}")
    
    if config.subcategory not in CATEGORIES[config.category]:
        raise HTTPException(status_code=400, detail=f"Invalid subcategory: {config.subcategory}")
    
    # Check if training script exists
    script_path = SYSTEM_DIR / config.category / config.subcategory / "train_osmix.py"
    if not script_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Training script not available for {config.category}/{config.subcategory}. Coming soon!"
        )
    
    # Check for dataset files
    dataset_path = DATASETS_DIR / config.category / config.subcategory
    train_bin = dataset_path / "train.bin"
    val_bin = dataset_path / "val.bin"
    
    if not train_bin.exists() or not val_bin.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset files not found. Please ensure train.bin and val.bin exist in {dataset_path}"
        )
    
    # Prepare training command
    training_id = f"{config.model_name}_{config.category}_{config.subcategory}"
    
    # Add background task to run training
    background_tasks.add_task(
        run_training,
        script_path,
        config,
        training_id
    )
    
    return {
        "status": "started",
        "training_id": training_id,
        "message": f"Training started for {config.model_name}",
        "config": config.dict()
    }

def run_training(script_path: Path, config: TrainingConfig, training_id: str):
    """Run training in background"""
    try:
        # Create training command with all parameters
        cmd = [
            sys.executable,
            str(script_path),
            "--model_name", config.model_name,
            "--output_format", config.output_format,
            "--batch_size", str(config.batch_size),
            "--block_size", str(config.block_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--learning_rate", str(config.learning_rate),
            "--max_iters", str(config.max_iters),
            "--n_layer", str(config.n_layer),
            "--n_head", str(config.n_head),
            "--n_embd", str(config.n_embd),
            "--dropout", str(config.dropout),
            "--device", config.device,
            "--dtype", config.dtype,
            "--save_interval", str(config.save_interval)
        ]
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save training log
        log_path = PRODUCT_DIR / "models" / f"{training_id}_log.txt"
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
            
    except Exception as e:
        print(f"Training error: {e}")

@app.post("/api/check_params")
async def check_params(request: CheckParamsRequest):
    """Check model parameters"""
    model_path = Path(request.model_path)
    
    # If relative path, try to resolve from product/models
    if not model_path.is_absolute():
        possible_paths = [
            PRODUCT_DIR / "models" / request.model_path,
            BASE_DIR / request.model_path,
            Path(request.model_path)
        ]
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
    
    try:
        import torch
        
        # Try to load checkpoint - handle different formats
        file_ext = model_path.suffix.lower()
        total_params = 0
        model_args = {}
        
        if file_ext == '.safetensors':
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                # For safetensors, we only have state dict
                total_params = sum(p.numel() for p in state_dict.values())
                model_args = {"format": "safetensors", "note": "Metadata not available in safetensors format"}
            except ImportError:
                raise HTTPException(status_code=500, detail="safetensors library not installed")
        elif file_ext == '.onnx':
            try:
                import onnx
                onnx_model = onnx.load(str(model_path))
                # Count parameters from ONNX model
                total_params = sum(param.data_type.size * param.dims[0] if len(param.dims) > 0 else 0 
                                 for param in onnx_model.graph.initializer)
                model_args = {"format": "onnx", "note": "Limited metadata available in ONNX format"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading ONNX model: {str(e)}")
        else:
            # Load checkpoint (ckpt, pth, pt, etc.)
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Extract parameters
            if "model_args" in checkpoint:
                model_args = checkpoint["model_args"]
            elif "config" in checkpoint:
                model_args = checkpoint["config"]
            else:
                model_args = {}
            
            # Get model size
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                total_params = sum(p.numel() for p in state_dict.values())
            else:
                total_params = 0
        
        return {
            "status": "success",
            "model_path": str(model_path.relative_to(BASE_DIR)),
            "parameters": model_args,
            "total_parameters": total_params,
            "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
            "format": file_ext[1:] if file_ext else "unknown"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking parameters: {str(e)}")

@app.post("/api/retrain")
async def retrain_model(config: RetrainConfig, background_tasks: BackgroundTasks):
    """Retrain an existing model"""
    model_path = Path(config.model_path)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Check if training script exists
    script_path = SYSTEM_DIR / config.category / config.subcategory / "train_osmix.py"
    if not script_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training script not available for {config.category}/{config.subcategory}"
        )
    
    # Load existing model config
    try:
        import torch
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Prepare retrain config
        retrain_id = f"retrain_{model_path.stem}"
        
        background_tasks.add_task(
            run_retraining,
            script_path,
            model_path,
            config,
            retrain_id
        )
        
        return {
            "status": "started",
            "retrain_id": retrain_id,
            "message": f"Retraining started for {model_path.name}",
            "additional_iters": config.additional_iters
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def run_retraining(script_path: Path, model_path: Path, config: RetrainConfig, retrain_id: str):
    """Run retraining in background"""
    try:
        # Load checkpoint to get original config
        import torch
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Prepare retrain command
        cmd = [
            sys.executable,
            str(script_path),
            "--init_from", str(model_path),
            "--max_iters", str(config.additional_iters)
        ]
        
        if config.learning_rate:
            cmd.extend(["--learning_rate", str(config.learning_rate)])
        if config.batch_size:
            cmd.extend(["--batch_size", str(config.batch_size)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save retrain log
        log_path = PRODUCT_DIR / "models" / f"{retrain_id}_log.txt"
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
            
    except Exception as e:
        print(f"Retraining error: {e}")

@app.get("/api/models")
async def list_models():
    """List all trained models"""
    models_dir = PRODUCT_DIR / "models"
    
    if not models_dir.exists():
        return {
            "status": "success",
            "models": []
        }
    
    models = []
    # Supported formats
    supported_formats = [".ckpt", ".pth", ".pt", ".safetensors", ".onnx", ".ggml", ".gguf"]
    
    for model_file in models_dir.glob("*.*"):
        if model_file.suffix.lower() in supported_formats:
            # Skip checkpoint files (those with _iter_ in name)
            if "_iter_" not in model_file.stem:
                models.append({
                    "name": model_file.name,
                    "path": str(model_file.relative_to(BASE_DIR)),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    "format": model_file.suffix[1:].lower()
                })
    
    # Also list checkpoints
    checkpoints = []
    for checkpoint_dir in models_dir.glob("*_checkpoints"):
        for checkpoint_file in checkpoint_dir.glob("*.*"):
            if checkpoint_file.suffix.lower() in supported_formats:
                checkpoints.append({
                    "name": checkpoint_file.name,
                    "path": str(checkpoint_file.relative_to(BASE_DIR)),
                    "size_mb": round(checkpoint_file.stat().st_size / (1024 * 1024), 2),
                    "format": checkpoint_file.suffix[1:].lower(),
                    "is_checkpoint": True,
                    "model_name": checkpoint_dir.stem.replace("_checkpoints", "")
                })
    
    return {
        "status": "success",
        "models": models,
        "checkpoints": checkpoints
    }

@app.get("/api/datasets/{category}/{subcategory}")
async def check_dataset(category: str, subcategory: str):
    """Check if dataset files exist for a subcategory"""
    dataset_path = DATASETS_DIR / category / subcategory
    train_bin = dataset_path / "train.bin"
    val_bin = dataset_path / "val.bin"
    
    return {
        "category": category,
        "subcategory": subcategory,
        "dataset_path": str(dataset_path),
        "train_bin_exists": train_bin.exists(),
        "val_bin_exists": val_bin.exists(),
        "ready": train_bin.exists() and val_bin.exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)