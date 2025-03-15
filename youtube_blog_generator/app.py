import os
import sys
import asyncio
import streamlit as st
import tempfile
import logging
import time
import random
from datetime import datetime
import torch
from dotenv import load_dotenv
from anthropic import Anthropic
import nest_asyncio

# Load environment variables from .env file
load_dotenv()

# Access the API key
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")

# Validate API key
if not CLAUDE_API_KEY:
    print("Warning: CLAUDE_API_KEY not found in environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
from utils.video_downloader import download_youtube_video
from utils.audio_processor import extract_audio, transcribe_audio
from utils.image_processor import extract_frames, deduplicate_frames, generate_captions
from utils.blog_generator import BlogGenerator, generate_blog_from_ai
from utils.torch_fix import fix_torch_for_python312

# Main Streamlit configuration
st.set_page_config(
    page_title="YouTube to Blog Generator",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply PyTorch fixes for Python 3.11+
fix_torch_for_python312()

# Apply nest_asyncio to allow nested event loops
try:
    nest_asyncio.apply()
except RuntimeError:
    pass

# Initialize asyncio properly
def init_asyncio():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Call this at the start of your app
init_asyncio()

# Session state initialization
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'transcription' not in st.session_state:
    st.session_state.transcription = {}
if 'captions' not in st.session_state:
    st.session_state.captions = []
if 'blog_data' not in st.session_state:
    st.session_state.blog_data = {}

# Title and description
st.title("🎥 YouTube to Blog Generator")
st.markdown("""
Convert YouTube videos into professional blog posts with AI. 
This tool extracts audio, video frames, and generates high-quality content automatically.
""")

# Sidebar configuration
st.sidebar.header("Settings")

# Blog generation settings
use_claude = st.sidebar.checkbox("Use Claude AI for enhanced blog generation", 
                                value=bool(CLAUDE_API_KEY))
blog_style = st.sidebar.selectbox(
    "Blog Style",
    ["Professional", "Casual", "Technical", "Educational"],
    index=0
)

export_format = st.sidebar.selectbox(
    "Export Format",
    ["HTML", "PDF", "Markdown"],
    index=0
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    frame_interval = st.slider("Frame extraction interval (seconds)", 1, 30, 5)
    deduplication_threshold = st.slider("Image similarity threshold", 0.5, 1.0, 0.75)
    num_images = st.slider("Maximum images to include", 5, 30, 15)

# YouTube URL input
st.header("Step 1: Enter YouTube URL")
youtube_url = st.text_input("YouTube video URL")

# Process button
process_clicked = st.button("Process Video")

# Create processing stages
if process_clicked and youtube_url:
    # Reset session state
    st.session_state.processed_video = False
    st.session_state.video_path = None
    st.session_state.frames = []
    st.session_state.timestamps = []
    st.session_state.transcription = {}
    st.session_state.captions = []
    st.session_state.blog_data = {}
    
    try:
        with st.status("Processing your video...", expanded=True) as status:
            # Step 1: Download video
            st.write("Downloading YouTube video...")
            video_path = download_youtube_video(youtube_url)
            st.session_state.video_path = video_path
            status.update(label="Video downloaded successfully!", state="running")
            
            # Step 2: Extract audio and transcribe
            st.write("Extracting and transcribing audio...")
            audio_path = extract_audio(video_path)
            transcription = transcribe_audio(audio_path)
            st.session_state.transcription = transcription
            status.update(label="Audio transcription complete!", state="running")
            
            # Step 3: Extract frames
            st.write("Extracting video frames...")
            frames, timestamps = extract_frames(video_path, interval=frame_interval)
            status.update(label="Frame extraction complete!", state="running")
            
            # Step 4: Deduplicate frames
            st.write("Processing and deduplicating frames...")
            unique_frames, unique_timestamps = deduplicate_frames(
                frames, timestamps, similarity_threshold=deduplication_threshold
            )
            
            # Limit the number of frames if needed
            if len(unique_frames) > num_images:
                # Evenly sample frames
                indices = [int(i) for i in numpy.linspace(0, len(unique_frames)-1, num_images)]
                unique_frames = [unique_frames[i] for i in indices]
                unique_timestamps = [unique_timestamps[i] for i in indices]
            
            st.session_state.frames = unique_frames
            st.session_state.timestamps = unique_timestamps
            status.update(label="Frame processing complete!", state="running")
            
            # Step 5: Generate captions for frames
            st.write("Generating image captions...")
            captions = generate_captions(unique_frames)
            st.session_state.captions = captions
            status.update(label="Caption generation complete!", state="running")
            
            # Step 6: Generate blog
            st.write("Generating blog content...")
            if use_claude and CLAUDE_API_KEY:
                client = Anthropic(api_key=CLAUDE_API_KEY)
                blog_data = generate_blog_from_ai(
                    transcription,
                    unique_frames,
                    unique_timestamps,
                    captions,
                    client=client,
                    model_name=CLAUDE_MODEL
                )
            else:
                blog_data = generate_blog_from_ai(transcription)
            st.session_state.blog_data = blog_data
            status.update(label="Blog generation complete!", state="complete")
            
            st.session_state.processed_video = True
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        logger.exception("Error in video processing")
        
        # Cleanup
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            try:
                os.remove(st.session_state.video_path)
            except:
                pass

# Display results if processing is complete
if st.session_state.processed_video:
    st.header("Step 2: Blog Preview")
    
    # Show transcription
    with st.expander("Video Transcription", expanded=False):
        st.write(st.session_state.transcription.get('full_text', 'No transcription available'))
    
    # Show images with captions
    with st.expander("Extracted Images", expanded=False):
        cols = st.columns(3)
        for i, (frame, caption) in enumerate(zip(st.session_state.frames, st.session_state.captions)):
            with cols[i % 3]:
                st.image(frame, caption=caption)
    
    # Show blog preview
    st.subheader("Blog Content Preview")
    st.markdown(f"## {st.session_state.blog_data.get('title', 'Blog Post')}")
    st.markdown(f"*{st.session_state.blog_data.get('meta_description', '')}*")
    
    # Show some of the blog content
    blog_content = st.session_state.blog_data.get('content', '')
    if export_format.lower() == 'html':
        st.components.v1.html(blog_content, height=500, scrolling=True)
    else:
        st.markdown(blog_content)
    
    # Export options
    st.header("Step 3: Export Blog")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as " + export_format):
            with st.spinner(f"Exporting as {export_format}..."):
                try:
                    content, mime_type, file_ext = export_blog(
                        st.session_state.blog_data,
                        st.session_state.frames,
                        export_format.lower()
                    )
                    
                    # Create download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"blog_{timestamp}.{file_ext}"
                    
                    st.download_button(
                        f"Download {export_format}",
                        content,
                        filename,
                        mime_type
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("Start Over"):
            # Reset session state
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    # Cleanup on exit
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        try:
            os.remove(st.session_state.video_path)
        except:
            pass 

def get_anthropic_client():
    """Initialize and return the Anthropic client"""
    try:
        client = Anthropic(api_key=CLAUDE_API_KEY)
        return client
    except Exception as e:
        print(f"Error initializing Anthropic client: {str(e)}")
        return None 

if __name__ == "__main__":
    main() 