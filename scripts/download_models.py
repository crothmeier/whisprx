#!/usr/bin/env python3
"""Download and setup models for WhispRX"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import urllib.request
import tarfile
import json


def create_models_dir():
    """Create models directory structure"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    subdirs = ["whisper", "llm", "tts/piper"]
    for subdir in subdirs:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"✓ Created models directory: {models_dir.absolute()}")
    return models_dir


def download_whisper(models_dir: Path, model_size: str = "base"):
    """Download Whisper model via faster-whisper

    Args:
        models_dir: Models directory path
        model_size: Model size (tiny, base, small, medium, large-v3)
    """
    print(f"\n📥 Downloading Whisper model ({model_size})...")

    whisper_dir = models_dir / "whisper"

    try:
        # faster-whisper will auto-download on first use
        # We'll just create a marker file with the model name
        model_file = whisper_dir / f"{model_size}.json"

        model_info = {
            "model_name": f"openai/whisper-{model_size}",
            "model_size": model_size,
            "backend": "faster-whisper"
        }

        with open(model_file, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"✓ Whisper model configuration saved")
        print(f"  Model will be auto-downloaded on first use")
        print(f"  Model: {model_size}")
        print(f"  Path: {whisper_dir / model_size}")

        return str(whisper_dir / model_size)

    except Exception as e:
        print(f"✗ Failed to setup Whisper: {e}")
        return None


def download_llm(models_dir: Path, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Setup LLM model info

    Args:
        models_dir: Models directory path
        model_name: HuggingFace model name
    """
    print(f"\n📥 Setting up LLM model...")

    llm_dir = models_dir / "llm"

    try:
        # vLLM will auto-download from HuggingFace
        # Save model configuration
        model_file = llm_dir / "config.json"

        model_info = {
            "model_name": model_name,
            "backend": "vllm",
            "note": "Model will be auto-downloaded by vLLM on first use"
        }

        with open(model_file, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"✓ LLM model configuration saved")
        print(f"  Model will be auto-downloaded by vLLM on first use")
        print(f"  Model: {model_name}")

        return model_name

    except Exception as e:
        print(f"✗ Failed to setup LLM: {e}")
        return None


def download_piper_tts(models_dir: Path, voice: str = "en_US-lessac-medium"):
    """Download Piper TTS model

    Args:
        models_dir: Models directory path
        voice: Voice name (e.g., en_US-lessac-medium)
    """
    print(f"\n📥 Downloading Piper TTS model ({voice})...")

    piper_dir = models_dir / "tts" / "piper"
    voice_dir = piper_dir / voice

    try:
        voice_dir.mkdir(parents=True, exist_ok=True)

        base_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0"

        # Download voice model (.onnx) and config (.json)
        files = [
            f"{voice}.onnx",
            f"{voice}.onnx.json",
        ]

        for filename in files:
            url = f"{base_url}/{filename}"
            output_path = voice_dir / filename.replace(f"{voice}.", "")

            print(f"  Downloading {filename}...")

            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ⚠ Failed to download {filename}: {e}")
                print(f"    You may need to download manually from:")
                print(f"    {url}")

        print(f"✓ Piper TTS setup complete")
        print(f"  Voice: {voice}")
        print(f"  Path: {voice_dir}")

        # Return path to the .onnx file
        onnx_file = voice_dir / f"{voice}.onnx"
        if onnx_file.exists():
            return str(onnx_file)
        else:
            return str(voice_dir / "model.onnx")  # Fallback name

    except Exception as e:
        print(f"✗ Failed to download Piper TTS: {e}")
        return None


def install_piper_binary():
    """Install Piper binary if not available"""
    print("\n🔧 Checking for Piper binary...")

    try:
        result = subprocess.run(
            ["piper", "--version"],
            capture_output=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ Piper is already installed")
            return True

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print("⚠ Piper not found")
    print("\nTo install Piper:")
    print("  1. Download from: https://github.com/rhasspy/piper/releases")
    print("  2. Or install via package manager:")
    print("     - Debian/Ubuntu: Download .deb from releases")
    print("     - Or build from source")
    print("\n  Alternatively, the system will fall back to espeak-ng")

    return False


def create_env_file(models_dir: Path, whisper_path: str, llm_name: str, tts_path: str):
    """Create .env file with model paths

    Args:
        models_dir: Models directory
        whisper_path: Path to Whisper model
        llm_name: LLM model name
        tts_path: Path to TTS model
    """
    env_file = Path(".env")

    env_content = f"""# WhispRX Model Configuration
# Auto-generated by download_models.py

# Whisper STT Model
WHISPER_MODEL_PATH={whisper_path}

# LLM Model (HuggingFace name or local path)
LLM_MODEL_NAME={llm_name}

# TTS Model (Piper .onnx file)
TTS_MODEL_PATH={tts_path}

# Optional: CUDA device
# CUDA_VISIBLE_DEVICES=0
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"\n✓ Created .env file: {env_file.absolute()}")
    print("\n  You can now start the server with:")
    print("    cd services/whisprx_api")
    print("    uvicorn main:app --reload")


def main():
    parser = argparse.ArgumentParser(description="Download WhispRX models")
    parser.add_argument(
        "--whisper-size",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--llm-model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="LLM model name from HuggingFace (default: microsoft/Phi-3-mini-4k-instruct)"
    )
    parser.add_argument(
        "--tts-voice",
        default="en_US-lessac-medium",
        help="Piper TTS voice (default: en_US-lessac-medium)"
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper download"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM setup"
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS download"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WhispRX Model Download Script")
    print("=" * 60)

    # Create models directory
    models_dir = create_models_dir()

    # Download models
    whisper_path = None
    llm_name = None
    tts_path = None

    if not args.skip_whisper:
        whisper_path = download_whisper(models_dir, args.whisper_size)

    if not args.skip_llm:
        llm_name = download_llm(models_dir, args.llm_model)

    if not args.skip_tts:
        install_piper_binary()
        tts_path = download_piper_tts(models_dir, args.tts_voice)

    # Create .env file
    if whisper_path or llm_name or tts_path:
        create_env_file(
            models_dir,
            whisper_path or "tiny",  # Default to tiny if not downloaded
            llm_name or "microsoft/Phi-3-mini-4k-instruct",
            tts_path or "",
        )

    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Install dependencies:")
    print("     pip install -r requirements.txt")
    print("\n  2. Start the server:")
    print("     cd services/whisprx_api")
    print("     python main.py")
    print("\n  3. Run the client:")
    print("     python packages/whisprx_sdk/client_demo.py")


if __name__ == "__main__":
    main()
