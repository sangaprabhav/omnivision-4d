
"""
Download all required model checkpoints
Run once before deployment or building Docker image
"""
import os
import sys
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

def download_sam2(checkpoint_dir: str = "/app/models"):
    """Download SAM-2 checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / "sam2_hiera_large.pt"
    
    if checkpoint_path.exists():
        print(f"SAM-2 already exists at {checkpoint_path}")
        return
    
    print("Downloading SAM-2 Large checkpoint...")
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    urllib.request.urlretrieve(url, checkpoint_path)
    print(f"✅ SAM-2 downloaded to {checkpoint_path}")

def download_depth(checkpoint_dir: str = "/app/models"):
    """Download ZoeDepth from HuggingFace"""
    depth_path = Path(checkpoint_dir) / "zoedepth"
    
    if depth_path.exists():
        print(f"ZoeDepth already exists at {depth_path}")
        return
    
    print("Downloading ZoeDepth...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    snapshot_download(
        repo_id="Intel/zoedepth-nyu-kitti",
        local_dir=str(depth_path),
        local_dir_use_symlinks=False
    )
    print(f"✅ ZoeDepth downloaded to {depth_path}")

def download_cosmos(checkpoint_dir: str = "/app/models"):
    """Download Cosmos-Reason2 (requires HF token)"""
    cosmos_path = Path(checkpoint_dir) / "cosmos-reason2-8b"
    
    if cosmos_path.exists():
        print(f"Cosmos already exists at {cosmos_path}")
        return
    
    print("Downloading Cosmos-Reason2-8B...")
    print("NOTE: This requires HuggingFace authentication and ~50GB storage")
    
    try:
        snapshot_download(
            repo_id="nvidia/Cosmos-Reason2-8B",
            local_dir=str(cosmos_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Cosmos downloaded to {cosmos_path}")
    except Exception as e:
        print(f"❌ Failed to download Cosmos: {e}")
        print("Make sure you've run 'huggingface-cli login' first")

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/models"
    
    print(f"Downloading models to {checkpoint_dir}...")
    print("=" * 60)
    
    download_sam2(checkpoint_dir)
    download_depth(checkpoint_dir)
    download_cosmos(checkpoint_dir)
    
    print("=" * 60)
    print("All downloads complete!")