import os
import tempfile
import subprocess
import logging
import numpy as np
import soundfile as sf
import json
import whisper
from utils.torch_fix import run_async

# Configure logging
logger = logging.getLogger(__name__)

# Cache for loaded models
_whisper_model = None

def extract_audio(video_path):
    """
    Extract audio from a video file using FFmpeg
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to the extracted audio file (.wav)
    """
    logger.info(f"Extracting audio from video: {video_path}")
    
    # Create a temporary file for the audio
    fd, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    try:
        # First try: standard audio extraction
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # Sample rate 16000Hz (standard for speech recognition)
                "-ac", "1",  # Mono
                audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            # Check if the output file exists and has content
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info("Audio extracted successfully")
                return audio_path
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Standard audio extraction failed: {e.stderr}")
            
        # Second try: Alternative format settings
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-f", "wav",  # Force WAV format
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info("Audio extracted successfully with alternative settings")
                return audio_path
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Alternative audio extraction failed: {e.stderr}")
            
        # Third try: Use more robust settings with explicit format detection
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-v", "quiet",
                "-stats",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info("Audio extracted successfully with robust settings")
                return audio_path
                
        except subprocess.CalledProcessError as e:
            logger.error(f"All extraction methods failed: {e.stderr}")
            
        # If we still don't have valid audio, create a silent audio file
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.warning("Creating silent audio as fallback")
            create_silent_audio(audio_path)
            return audio_path
            
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        # Create silent audio as fallback
        create_silent_audio(audio_path)
        
    return audio_path

def create_silent_audio(output_path, duration=10, sample_rate=16000):
    """Create a silent audio file for fallback cases"""
    try:
        # Generate silent audio (zeros)
        samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
        
        # Write to WAV file
        sf.write(output_path, samples, sample_rate)
        logger.info(f"Created silent audio file: {output_path}")
    except Exception as e:
        logger.error(f"Error creating silent audio: {str(e)}")

def get_whisper_model():
    """Get or load the Whisper model (with caching)"""
    global _whisper_model
    
    if _whisper_model is None:
        try:
            logger.info("Loading Whisper model (base)")
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            return None
            
    return _whisper_model

def transcribe_audio(audio_path):
    """
    Transcribe audio to text using Whisper
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing the transcription with timestamps
    """
    logger.info(f"Transcribing audio: {audio_path}")
    
    try:
        # First try: Using Whisper directly
        model = get_whisper_model()
        
        if model:
            try:
                # Load audio
                audio_data, sample_rate = sf.read(audio_path)
                
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # If stereo, convert to mono
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Transcribe using our async wrapper
                result = model.transcribe(
                    audio_data,
                    verbose=False,
                    condition_on_previous_text=True,
                    initial_prompt="This is a transcription of a YouTube video."
                )
                
                if result['text'] and len(result['text']) >= 10:
                    logger.info("Audio transcription successful using Whisper")
                    return {
                        'full_text': result['text'],
                        'segments': [
                            {
                                'text': seg['text'],
                                'start': seg['start'],
                                'end': seg['end']
                            }
                            for seg in result.get('segments', [])
                        ]
                    }
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {str(e)}")
        
        # Second try: Command-line Whisper
        try:
            logger.info("Trying command-line Whisper...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                whisper_cmd = [
                    "whisper", audio_path,
                    "--model", "base",
                    "--output_format", "json",
                    "--output_dir", tmp_dir
                ]
                
                subprocess.run(whisper_cmd, check=True, capture_output=True)
                json_path = os.path.join(tmp_dir, f"{os.path.basename(audio_path)}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        result = json.load(f)
                    
                    logger.info("Audio transcription successful using command-line Whisper")
                    return {
                        'full_text': result.get('text', ''),
                        'segments': result.get('segments', [])
                    }
        except Exception as e:
            logger.warning(f"Command-line Whisper failed: {str(e)}")
        
        # Third try: Use FFmpeg to extract text with subtitle format 
        try:
            logger.info("Trying FFmpeg subtitle extraction...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                srt_path = os.path.join(tmp_dir, "subtitles.srt")
                
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-f", "srt",
                    srt_path
                ]
                
                subprocess.run(ffmpeg_cmd, capture_output=True)
                
                if os.path.exists(srt_path) and os.path.getsize(srt_path) > 0:
                    # Parse SRT file
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Simple SRT parsing
                    segments = []
                    current_time = ""
                    current_text = ""
                    full_text = ""
                    
                    for line in lines:
                        line = line.strip()
                        if '-->' in line:
                            current_time = line
                        elif line and not line.isdigit():
                            current_text += line + " "
                        elif not line and current_text:
                            start_time = float(current_time.split('-->')[0].strip().replace(',', '.').split(':')[-1])
                            end_time = float(current_time.split('-->')[1].strip().replace(',', '.').split(':')[-1])
                            
                            segments.append({
                                'text': current_text.strip(),
                                'start': start_time,
                                'end': end_time
                            })
                            
                            full_text += current_text.strip() + " "
                            current_text = ""
                    
                    logger.info("Audio transcription extracted from subtitles")
                    return {
                        'full_text': full_text.strip(),
                        'segments': segments
                    }
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {str(e)}")
        
        # Fallback: Return empty transcription with a note
        logger.warning("All transcription methods failed. Returning minimal transcription.")
        return {
            'full_text': "Transcription failed. The blog will be generated based on visual content only.",
            'segments': [{'text': "No transcription available", 'start': 0, 'end': 0}]
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return {
            'full_text': f"Error during transcription: {str(e)}",
            'segments': [{'text': "No transcription available", 'start': 0, 'end': 0}]
        } 