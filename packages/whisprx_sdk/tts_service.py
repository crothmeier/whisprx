#!/usr/bin/env python3
"""TTS Service using Piper for fast, local text-to-speech synthesis"""
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional
import numpy as np


class TTSService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        use_piper: bool = True,
    ):
        """Initialize TTS service

        Args:
            model_path: Path to TTS model (Piper .onnx file)
            sample_rate: Output sample rate in Hz
            use_piper: Whether to use Piper (if False, uses espeak as fallback)
        """
        self.sample_rate = sample_rate
        self.use_piper = use_piper
        self.model_path = model_path

        # Check if Piper is available
        if use_piper:
            try:
                result = subprocess.run(
                    ["piper", "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"✓ Found Piper TTS")
                    if model_path and Path(model_path).exists():
                        print(f"✓ Using model: {model_path}")
                    else:
                        print(f"⚠ Model path not found, will use Piper defaults")
                else:
                    print("⚠ Piper not available, falling back to espeak")
                    self.use_piper = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("⚠ Piper not found, falling back to espeak")
                self.use_piper = False

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio waveform

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio bytes (16-bit signed int, mono)
        """
        if not text or not text.strip():
            return b""

        try:
            if self.use_piper and self.model_path:
                return self._synthesize_piper(text)
            else:
                return self._synthesize_espeak(text)
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return b""

    def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using Piper

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio bytes
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            # Build Piper command
            cmd = ["piper"]

            if self.model_path:
                cmd.extend(["--model", self.model_path])

            cmd.extend([
                "--output_file", tmp_path,
                "--length_scale", "1.0",  # Normal speed
                "--noise_scale", "0.667",  # Default quality
                "--noise_w", "0.8",  # Default quality
            ])

            # Run Piper
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Piper failed: {result.stderr.decode()}")

            # Read WAV file and extract PCM data
            with wave.open(tmp_path, 'rb') as wav:
                if wav.getnchannels() != 1:
                    raise ValueError("Expected mono audio")

                # Read all frames
                pcm_data = wav.readframes(wav.getnframes())

                # Resample if needed (simple decimation/interpolation)
                if wav.getframerate() != self.sample_rate:
                    pcm_data = self._resample_audio(
                        pcm_data,
                        wav.getframerate(),
                        self.sample_rate
                    )

            return pcm_data

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def _synthesize_espeak(self, text: str) -> bytes:
        """Synthesize using espeak as fallback

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio bytes
        """
        try:
            # espeak can output raw PCM directly
            cmd = [
                "espeak-ng",
                "-v", "en-us",
                "-s", "150",  # Speed (words per minute)
                "-p", "50",   # Pitch
                "--stdout",
                text
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"espeak failed: {result.stderr.decode()}")

            # espeak outputs WAV format, extract PCM
            # Skip WAV header (44 bytes typically)
            wav_data = result.stdout
            if len(wav_data) > 44:
                return wav_data[44:]  # Skip header
            return wav_data

        except FileNotFoundError:
            print("⚠ espeak-ng not found. Install with: apt-get install espeak-ng")
            return b""

    def _resample_audio(self, pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Simple audio resampling using numpy

        Args:
            pcm_data: Input PCM data
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled PCM data
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(pcm_data, dtype=np.int16)

        # Simple linear interpolation resampling
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)

        # Create time arrays
        old_time = np.linspace(0, duration, len(audio))
        new_time = np.linspace(0, duration, new_length)

        # Interpolate
        resampled = np.interp(new_time, old_time, audio.astype(np.float32))

        # Convert back to int16
        return resampled.astype(np.int16).tobytes()


# Quick test
if __name__ == "__main__":
    tts = TTSService()
    audio = tts.synthesize("Hello, this is a test of the text to speech system.")

    if audio:
        print(f"Generated {len(audio)} bytes of audio")

        # Save to file for testing
        with wave.open("test_output.wav", "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)
            wav.writeframes(audio)
        print("Saved to test_output.wav")
    else:
        print("Failed to generate audio")
