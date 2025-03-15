import os
import cv2
import numpy as np
import logging
import torch
from PIL import Image
import tempfile
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for models
_image_captioner = None
_feature_extractor = None

def extract_frames(video_path, interval=5):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        interval: Interval in seconds between frames
        
    Returns:
        (frames, timestamps) tuple containing frame images and their timestamps
    """
    logger.info(f"Extracting frames from video: {video_path} at {interval}s intervals")
    
    frames = []
    timestamps = []
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return [], []
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning(f"Invalid FPS detected: {fps}, using default of 25")
            fps = 25
            
        # Calculate frame interval
        frame_interval = int(fps * interval)
        logger.info(f"Video FPS: {fps}, extracting every {frame_interval} frames")
        
        # Read frames
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Save frame at specified intervals
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(frame_count / fps)
                
            frame_count += 1
            
            # Log progress for long videos
            if frame_count % 1000 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # If no frames were extracted, create a test frame
        if not frames:
            logger.warning("No frames extracted, creating test frame")
            test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            cv2.putText(test_frame, "No frames could be extracted", (320, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(test_frame)
            timestamps.append(0.0)
            
        return frames, timestamps
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return [], []

def get_image_hash(image):
    """Generate perceptual hash of image for deduplication"""
    # Convert to grayscale and resize to 8x8
    img = Image.fromarray(image).resize((8, 8), Image.Resampling.LANCZOS).convert('L')
    
    # Convert to numpy array
    pixels = np.array(img.getdata(), dtype=np.float32).reshape((8, 8))
    
    # Apply DCT transform
    dct = cv2.dct(pixels)
    
    # Take the lowest frequency components
    hash_value = dct[:8, :8].flatten()
    return hash_value

def load_feature_extractor():
    """Load and initialize the feature extractor for image similarity"""
    from transformers import AutoFeatureExtractor, AutoModel
    
    try:
        # Initialize the model and processor
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224",
            use_fast=True
        )
        model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        
        def extract_features(image):
            """Extract features from image using Vision Transformer"""
            pil_image = Image.fromarray(image)
            inputs = feature_extractor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.pooler_output.cpu().numpy()[0]
        
        return extract_features
        
    except Exception as e:
        logger.error(f"Error loading feature extractor: {str(e)}")
        return None

def deduplicate_frames(frames, timestamps, similarity_threshold=0.85):
    """
    Remove similar frames based on perceptual hashing and feature similarity
    
    Args:
        frames: List of frame images
        timestamps: List of frame timestamps
        similarity_threshold: Threshold for considering frames similar (0-1)
        
    Returns:
        (unique_frames, unique_timestamps) tuple containing deduplicated frames and timestamps
    """
    if not frames:
        logger.warning("No frames to deduplicate")
        return [], []
        
    logger.info(f"Deduplicating {len(frames)} frames with threshold {similarity_threshold}")
    
    try:
        # First-level filtering using perceptual hashing
        hashes = [get_image_hash(frame) for frame in frames]
        
        # If available, use feature extraction for better similarity detection
        extract_features = load_feature_extractor()
        
        if extract_features:
            # Extract visual features from each frame
            try:
                features = [extract_features(frame) for frame in frames]
                features = np.array(features)
                
                # Compute similarity matrix
                similarity_matrix = cosine_similarity(features)
                
                # Find frames to remove (similar to others)
                to_remove = set()
                
                for i in range(len(frames)):
                    if i in to_remove:
                        continue
                    for j in range(i + 1, len(frames)):
                        if j in to_remove:
                            continue
                        if similarity_matrix[i, j] > similarity_threshold:
                            to_remove.add(j)
                
                # Keep only unique frames
                unique_frames = [f for i, f in enumerate(frames) if i not in to_remove]
                unique_timestamps = [t for i, t in enumerate(timestamps) if i not in to_remove]
                
                logger.info(f"Deduplicated frames: {len(frames)} -> {len(unique_frames)} (feature-based)")
                return unique_frames, unique_timestamps
                
            except Exception as e:
                logger.warning(f"Feature-based deduplication failed: {str(e)}")
        
        # Fallback to simple hash-based deduplication
        # Compute pairwise hash distances
        distances = np.zeros((len(frames), len(frames)))
        for i in range(len(frames)):
            for j in range(i+1, len(frames)):
                distances[i, j] = distances[j, i] = np.linalg.norm(hashes[i] - hashes[j])
        
        # Normalize distances
        max_dist = np.max(distances) if np.max(distances) > 0 else 1
        distances /= max_dist
        
        # Use a simple threshold for similarity
        similar_threshold = 0.2  # Lower means more similar
        
        # Find frames to remove
        to_remove = set()
        
        for i in range(len(frames)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(frames)):
                if j in to_remove:
                    continue
                if distances[i, j] < similar_threshold:
                    to_remove.add(j)
        
        # Keep only unique frames
        unique_frames = [f for i, f in enumerate(frames) if i not in to_remove]
        unique_timestamps = [t for i, t in enumerate(timestamps) if i not in to_remove]
        
        logger.info(f"Deduplicated frames: {len(frames)} -> {len(unique_frames)} (hash-based)")
        return unique_frames, unique_timestamps
        
    except Exception as e:
        logger.error(f"Error during frame deduplication: {str(e)}")
        return frames, timestamps  # Return original if deduplication fails

def load_image_captioner():
    """Load and initialize the image captioning model"""
    global _image_captioner
    
    if _image_captioner is None:
        try:
            logger.info("Loading image captioning model...")
            _image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            logger.info("Image captioning model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading image captioning model: {str(e)}")
            return None
            
    return _image_captioner

def generate_captions(frames):
    """
    Generate captions for a list of frames
    
    Args:
        frames: List of frame images
        
    Returns:
        List of text captions for each frame
    """
    if not frames:
        logger.warning("No frames to caption")
        return []
        
    logger.info(f"Generating captions for {len(frames)} frames")
    
    captions = []
    captioner = load_image_captioner()
    
    if not captioner:
        # Fallback to simple scene descriptions
        logger.warning("Using fallback caption generator")
        for i, frame in enumerate(frames):
            captions.append(f"Scene {i+1} from the video")
        return captions
    
    try:
        for i, frame in enumerate(frames):
            try:
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Generate caption
                caption_result = captioner(pil_image)
                
                if caption_result and isinstance(caption_result, list) and len(caption_result) > 0:
                    caption = caption_result[0].get('generated_text', '')
                    captions.append(caption)
                else:
                    captions.append(f"Scene {i+1} from the video")
                    
            except Exception as frame_error:
                logger.warning(f"Error captioning frame {i}: {str(frame_error)}")
                captions.append(f"Scene {i+1} from the video")
                
    except Exception as e:
        logger.error(f"Error during image captioning: {str(e)}")
        # Fallback to simple captions
        for i in range(len(frames)):
            captions.append(f"Scene {i+1} from the video")
    
    logger.info(f"Generated {len(captions)} captions")
    return captions 