import streamlit as st
# Must be the first Streamlit command
st.set_page_config(
    page_title="Video to Blog Converter",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yt_dlp
import cv2
import os
from PIL import Image
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from transformers.pipelines import pipeline
import torch
import tempfile
from pathlib import Path
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import jinja2
import markdown
from datetime import datetime
import warnings
import torchvision
import shutil
from dotenv import load_dotenv
import json
import time
import requests
from anthropic import Anthropic
import re
import whisper
import subprocess
import soundfile as sf
import zipfile
from bs4 import BeautifulSoup
import asyncio
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from pytube import YouTube

# Workaround for PyTorch compatibility with Python 3.12
def fix_torch_classes():
    class PathFix:
        def __init__(self):
            self._path = []
            
        @property
        def _path(self):
            return self._path_value
            
        @_path.setter
        def _path(self, value):
            self._path_value = value
    
    sys.modules['torch'].__path__ = PathFix()

# Apply the fix before importing torch
fix_torch_classes()

try:
    import pdfkit
except ImportError:
    pass

try:
    from weasyprint import HTML
except ImportError:
    pass

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    pass

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    pass


torchvision.disable_beta_transforms_warning()


warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


load_dotenv()


image_captioner = None
image_feature_extractor = None
feature_model = None
whisper_model = None


CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")


@st.cache_resource
def load_models():
    global image_captioner, image_feature_extractor, feature_model, whisper_model
    try:
        image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        
        feature_extractor, model = initialize_model()
        
        def extract_features(image):
            inputs = feature_extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            return {"pooler_output": outputs.pooler_output.cpu().numpy()}
        
        image_feature_extractor = extract_features
        feature_model = model
        
        whisper_model = whisper.load_model("base")
        
        return image_captioner, image_feature_extractor, whisper_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def initialize_model():
    try:
        # Initialize the model and processor with explicit fast processing
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224",
            use_fast=True
        )
        model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        return feature_extractor, model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None, None

def get_image_hash(image):
    """Generate perceptual hash of image for deduplication"""
    img = Image.fromarray(image).resize((8, 8), Image.Resampling.LANCZOS).convert('L')
    pixels = np.array(img.getdata(), dtype=np.float32).reshape((8, 8))
    dct = cv2.dct(pixels)
    hash_value = dct[:8, :8].flatten()
    return hash_value

def deduplicate_frames(frames, timestamps, similarity_threshold=0.85):
    """Remove similar frames based on perceptual hashing and feature similarity"""
    if not frames or image_feature_extractor is None:
        return frames, timestamps
    
    try:
        features = []
        for frame in frames:
            try:
                pil_image = Image.fromarray(frame)
                feature = image_feature_extractor(pil_image)
                features.append(feature['pooler_output'][0])
            except Exception as e:
                st.warning(f"Error extracting features: {str(e)}")
                return frames, timestamps
        
        features = np.array(features)
        
        similarity_matrix = cosine_similarity(features)
        
        to_remove = set()
        
        for i in range(len(frames)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(frames)):
                if j in to_remove:
                    continue
                if similarity_matrix[i, j] > similarity_threshold:
                    to_remove.add(j)
        
        unique_frames = [f for i, f in enumerate(frames) if i not in to_remove]
        unique_timestamps = [t for i, t in enumerate(timestamps) if i not in to_remove]
        
        return unique_frames, unique_timestamps
    except Exception as e:
        st.warning(f"Frame deduplication failed: {str(e)}")
        return frames, timestamps

def extract_blog_content(content):
    """Extract blog content from Claude's response when JSON parsing fails"""
    try:
        title_match = re.search(r'title["\s:]+([^"}\n]+)', content)
        title = title_match.group(1) if title_match else "Video Analysis"
        
        meta_match = re.search(r'meta_description["\s:]+([^"}\n]+)', content)
        meta_description = meta_match.group(1) if meta_match else "Analysis of video content"
        
        content_text = ""
        in_content = False
        for line in content.split('\n'):
            if '<h1>' in line or '<p>' in line:
                in_content = True
            if in_content:
                content_text += line + '\n'
            if '</body>' in line or '"}' in line:
                in_content = False
        
        if not content_text:
            content_start = content.find('"content":')
            if content_start > 0:
                content_text = content[content_start+10:]
                content_text = content_text.strip('"}').strip()
        
        return {
            "title": title.strip('"\' '),
            "meta_description": meta_description.strip('"\' '),
            "content": content_text.strip()
        }
    except Exception as e:
        st.error(f"Error extracting blog content: {str(e)}")
        return {
            "title": "Video Analysis",
            "meta_description": "Analysis of video content",
            "content": content
        }

def generate_blog_with_claude(transcription, frames, timestamps, captions):
    """Generate a high-quality blog based on video content with intelligent image placement"""
    try:
        client = Anthropic(api_key=CLAUDE_API_KEY)
        
        # First, analyze the transcript for key topics and structure
        analysis_prompt = f"""
        Analyze this video transcript and identify:
        1. Main topics and themes
        2. Key points and insights
        3. Supporting examples or data
        4. Technical concepts explained
        5. Practical applications discussed
        6. Natural content segments
        7. Important quotes or statements
        8. Industry trends mentioned
        9. Expert opinions or citations
        10. Action items or takeaways

        Transcript:
        {transcription['full_text']}

        Provide a detailed analysis that will help structure a comprehensive blog post.
        """
        
        analysis_response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            temperature=0.3,
            system="You are an expert content analyst specializing in video content analysis.",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Create intelligent image mapping
        visual_context = {}
        for i, (timestamp, caption) in enumerate(zip(timestamps, captions)):
            relevant_segments = [
                seg for seg in transcription['segments']
                if abs(seg['start'] - timestamp) < 15  # Increased window to 15 seconds
            ]
            if relevant_segments:
                visual_context[timestamp] = {
                    'image_index': i,
                'caption': caption,
                    'context': relevant_segments[0]['text'],
                    'timestamp': timestamp
                }
        
        # Generate the comprehensive blog
        blog_prompt = f"""
        Create a comprehensive, engaging blog post based on this video content.
        Use the content analysis and available images to create a well-structured article.

        Content Analysis:
        {analysis_response.content[0].text}

        Available Images and Context:
        {json.dumps(visual_context, indent=2)}

        Full Transcript:
        {transcription['full_text']}

        Requirements:

        1. Content Structure:
           - Create an SEO-optimized headline that captures the main topic
           - Write a compelling introduction (2-3 paragraphs) that hooks the reader
           - Include an executive summary (300-500 words)
           - Create a clear table of contents
           - Organize content into 5-7 logical sections based on the video's flow
           - Add descriptive subheadings for each section
           - Include a strong conclusion with actionable takeaways
           - End with a compelling call to action

        2. Content Enhancement:
           - Integrate direct quotes from the video for authenticity
           - Add relevant industry statistics and data points
           - Include expert insights from the video
           - Provide practical examples and case studies
           - Add implementation tips and best practices
           - Address common questions and challenges
           - Include future implications and trends

        3. Visual Integration:
           - Place images strategically to support the content
           - Use this EXACT format for images:
           <img src="image_N.jpg" alt="detailed description" style="max-width: 100%; height: auto; margin: 20px 0;" />
           - Ensure each image relates to the surrounding text
           - Add descriptive alt text for SEO
           - Place images at natural breaks in content

        4. Engagement Elements:
           - Add thought-provoking questions throughout
           - Include highlighted key quotes
           - Use bullet points for lists and takeaways
           - Create info boxes for important concepts
           - Add "Tweet This" quotes
           - Include expert tips boxes
           - Use examples and analogies for complex concepts

        5. SEO Optimization:
           - Use semantic keywords naturally
           - Optimize header hierarchy
           - Include internal linking suggestions
           - Add meta description
           - Use LSI keywords
           - Optimize for featured snippets

        Format the response as a JSON object with:
        {
            "title": "SEO-optimized title",
            "meta_description": "Compelling 150-160 character description",
            "content": "Full HTML blog content with strategically placed images"
        }

        Make the content comprehensive, engaging, and highly valuable to readers.
        Focus on maintaining the speaker's voice while adding professional polish.
        """

        # Generate blog with Claude
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0.7,
            system="You are an expert blog writer specializing in creating engaging, comprehensive content from video transcripts.",
            messages=[{"role": "user", "content": blog_prompt}]
        )

        content = response.content[0].text

        # Parse response and handle potential JSON issues
        try:
            json_str = content[content.find('{'):content.rfind('}')+1]
            blog_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            blog_data = extract_blog_content(content)

        # Add schema markup and enhance content
        keywords = extract_keywords(blog_data['content'])
        blog_data['content'] = add_schema_markup(
            blog_data['content'],
            blog_data['title'],
            blog_data['meta_description'],
            keywords
        )

        return blog_data

    except Exception as e:
        st.error(f"Error generating blog with Claude: {str(e)}")
        st.info("Falling back to offline blog generation...")
        return generate_enhanced_blog(transcription['full_text'], captions, timestamps)

def load_blog_templates():
    """Load Jinja2 templates for different blog styles"""
    templates = {
        "Modern": """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="description" content="{{ meta_description }}">
            <title>{{ title }}</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
                img { max-width: 100%; height: auto; border-radius: 8px; margin: 20px 0; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                .caption { font-style: italic; color: #666; }
                .math { font-family: 'Computer Modern', serif; }
            </style>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body>
            {{ content|safe }}
        </body>
        </html>
        """,
        "Minimal": """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="description" content="{{ meta_description }}">
            <title>{{ title }}</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.5; max-width: 650px; margin: 2rem auto; padding: 0 1rem; }
                img { max-width: 100%; }
                h1, h2 { font-weight: 600; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body>
            {{ content|safe }}
        </body>
        </html>
        """,
        "Academic": """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="description" content="{{ meta_description }}">
            <title>{{ title }}</title>
            <style>
                body { font-family: 'Times New Roman', serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
                img { max-width: 100%; }
                .figure { text-align: center; margin: 20px 0; }
                .caption { font-style: italic; }
                .abstract { font-style: italic; margin: 20px 0; }
                .math { font-family: 'Computer Modern', serif; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body>
            {{ content|safe }}
        </body>
        </html>
        """
    }
    return {name: jinja2.Template(template) for name, template in templates.items()}

def export_blog(blog_data, format_type, template_name, images):
    """Export blog in various formats"""
    if format_type == 'html':
        templates = load_blog_templates()
        html_content = templates[template_name].render(
            title=blog_data['title'],
            content=blog_data['content'],
            meta_description=blog_data['meta_description']
        )
        return html_content, 'text/html'
        
    elif format_type == 'pdf':
        try:
            templates = load_blog_templates()
            html_content = templates[template_name].render(
                title=blog_data['title'],
                content=blog_data['content'],
                meta_description=blog_data['meta_description']
            )
            
            with tempfile.TemporaryDirectory() as pdf_dir:
                html_path = os.path.join(pdf_dir, "blog.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                for i, img_path in enumerate(images):
                    img_filename = f"image_{i}.jpg"
                    shutil.copy2(img_path, os.path.join(pdf_dir, img_filename))
                
                pdf_path = os.path.join(pdf_dir, "blog.pdf")
                
                pdf_generated = False
                
                try:
                    import pdfkit
                    st.info("Generating PDF with pdfkit...")
                    
                    if os.name == 'nt':
                        wkhtmltopdf_paths = [
                            r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                            r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                            os.environ.get('WKHTMLTOPDF_PATH', '')
                        ]
                        
                        config = None
                        for path in wkhtmltopdf_paths:
                            if os.path.exists(path):
                                config = pdfkit.configuration(wkhtmltopdf=path)
                                break
                                
                        if not config:
                            st.warning("wkhtmltopdf not found in common locations. Trying without configuration...")
                    else:
                        config = None
                    
                    pdf_options = {
                        'page-size': 'A4',
                        'margin-top': '20mm',
                        'margin-right': '20mm',
                        'margin-bottom': '20mm',
                        'margin-left': '20mm',
                        'encoding': 'UTF-8',
                        'no-outline': None,
                        'enable-local-file-access': None
                    }
                    
                    pdfkit.from_file(
                        html_path,
                        pdf_path,
                        options=pdf_options,
                        configuration=config
                    )
                    
                    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                        pdf_generated = True
                        st.success("PDF generated successfully with pdfkit")
                    else:
                        st.warning("pdfkit did not generate a valid PDF file")
                        
                except Exception as pdfkit_error:
                    st.warning(f"pdfkit error: {str(pdfkit_error)}")
                
                if not pdf_generated:
                    try:
                        from playwright.sync_api import sync_playwright
                        st.info("Trying final PDF generation method with Playwright...")
                        
                        with sync_playwright() as p:
                            browser = p.chromium.launch()
                            page = browser.new_page()
                            page.goto(f"file://{os.path.abspath(html_path)}")
                            page.wait_for_load_state("networkidle")
                            page.pdf(path=pdf_path, format="A4", margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"})
                            browser.close()
                        
                        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                            pdf_generated = True
                            st.success("PDF generated successfully with Playwright")
                        else:
                            st.warning("Playwright did not generate a valid PDF file")
                            
                    except Exception as playwright_error:
                        st.warning(f"Playwright error: {str(playwright_error)}")
                
                if not pdf_generated:
                    try:
                        from reportlab.lib.pagesizes import A4
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
                        from reportlab.lib.styles import getSampleStyleSheet
                        
                        st.info("Generating simple PDF with ReportLab...")
                        
                        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                        styles = getSampleStyleSheet()
                        
                        content_elements = []
                        
                        content_elements.append(Paragraph(blog_data['title'], styles['Title']))
                        content_elements.append(Spacer(1, 12))
                        
                        content_elements.append(Paragraph(blog_data['meta_description'], styles['Italic']))
                        content_elements.append(Spacer(1, 12))
                        
                        soup = BeautifulSoup(blog_data['content'], 'html.parser')
                        
                        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'img']):
                            if element.name == 'h1':
                                content_elements.append(Paragraph(element.text, styles['Heading1']))
                            elif element.name == 'h2':
                                content_elements.append(Paragraph(element.text, styles['Heading2']))
                            elif element.name == 'h3':
                                content_elements.append(Paragraph(element.text, styles['Heading3']))
                            elif element.name == 'p':
                                content_elements.append(Paragraph(element.text, styles['Normal']))
                            elif element.name == 'img':
                                img_src = element.get('src', '')
                                if img_src.startswith('image_'):
                                    img_idx = int(img_src.split('_')[1].split('.')[0])
                                    if img_idx < len(images):
                                        try:
                                            img = RLImage(images[img_idx], width=400, height=300)
                                            content_elements.append(img)
                                        except Exception as img_error:
                                            st.warning(f"Could not add image {img_idx}: {str(img_error)}")
                            
                            content_elements.append(Spacer(1, 6))
                        
                        doc.build(content_elements)
                        
                        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                            pdf_generated = True
                            st.success("Simple PDF generated successfully with ReportLab")
                        else:
                            st.warning("ReportLab did not generate a valid PDF file")
                            
                    except Exception as reportlab_error:
                        st.warning(f"ReportLab error: {str(reportlab_error)}")
                
                if pdf_generated:
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_content = pdf_file.read()
                    return pdf_content, 'application/pdf'
                else:
                    st.error("All PDF generation methods failed. Falling back to HTML format.")
                    return html_content, 'text/html'
                
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            st.info("Falling back to HTML format...")
            templates = load_blog_templates()
            html_content = templates[template_name].render(
                title=blog_data['title'],
                content=blog_data['content'],
                meta_description=blog_data['meta_description']
            )
            return html_content, 'text/html'
            
    elif format_type == 'md':
        with tempfile.TemporaryDirectory() as md_export_dir:
            image_dir = os.path.dirname(images[0]) if images else ""
            export_image_paths = []
            
            for i, img_path in enumerate(images):
                export_img_path = os.path.join(md_export_dir, f"image_{i}.jpg")
                shutil.copy2(img_path, export_img_path)
                export_image_paths.append(export_img_path)
            
            md_content = f"# {blog_data['title']}\n\n"
            md_content += f"_{blog_data['meta_description']}_\n\n"
            content = blog_data['content']
            
            for i, img in enumerate(images):
                img_filename = f"image_{i}.jpg"
                content = content.replace(
                    f'<img src="image_{i}.jpg"',
                    f'![Image {i}]({img_filename})'
                )
                content = content.replace(
                    f'<img src="{img_filename}"',
                    f'![Image {i}]({img_filename})'
                )
            
            content = content.replace('<h1>', '# ').replace('</h1>', '\n\n')
            content = content.replace('<h2>', '## ').replace('</h2>', '\n\n')
            content = content.replace('<h3>', '### ').replace('</h3>', '\n\n')
            content = content.replace('<p>', '').replace('</p>', '\n\n')
            content = content.replace('<br>', '\n')
            content = content.replace('<strong>', '**').replace('</strong>', '**')
            content = content.replace('<em>', '*').replace('</em>', '*')
            content = content.replace('<ul>', '').replace('</ul>', '\n')
            content = content.replace('<li>', '- ').replace('</li>', '\n')
            
            md_content += f"## Table of Contents\n\n"
            
            headings = re.findall(r'<h[2-3][^>]*>(.*?)</h[2-3]>', blog_data['content'])
            for heading in headings:
                anchor = heading.lower().replace(' ', '-').replace('.', '').replace(',', '')
                md_content += f"- [{heading}](#{anchor})\n"
            
            md_content += "\n\n" + content
            
            md_content += f"\n\n---\n\n"
            md_content += f"*Keywords: {', '.join(extract_keywords(blog_data['content']))}\n\n"
            md_content += f"*Published: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            zip_path = os.path.join(tempfile.gettempdir(), f"blog_export_{int(time.time())}.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                md_file_path = os.path.join(md_export_dir, "blog.md")
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                zipf.write(md_file_path, arcname="blog.md")
                
                for i, img_path in enumerate(export_image_paths):
                    zipf.write(img_path, arcname=f"image_{i}.jpg")
            
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
            
            os.remove(zip_path)
            
            return zip_content, 'application/zip'

def download_youtube_video(url):
    """Enhanced YouTube video download function with multiple fallbacks"""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    def sanitize_filename(title):
        """Clean filename of invalid characters"""
        return "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    def try_yt_dlp(url):
        """Attempt download with yt-dlp"""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'nocheckcertificate': True,
                'ignoreerrors': True,
                'no_warnings': False,
                'quiet': False,
                'extract_audio': True,
                'audio_format': 'mp3',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                },
                'socket_timeout': 30,
                'retries': 10
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    title = info.get('title', 'video')
                    filename = os.path.join(temp_dir, f"{sanitize_filename(title)}.mp3")
                    if os.path.exists(filename):
                        return filename
        except Exception as e:
            st.warning(f"yt-dlp attempt failed: {str(e)}")
        return None

    def try_pytube(url):
        """Attempt download with pytube"""
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            if stream:
                output_file = stream.download(output_path=temp_dir)
                base, _ = os.path.splitext(output_file)
                new_file = base + '.mp3'
                os.rename(output_file, new_file)
                return new_file
        except Exception as e:
            st.warning(f"pytube attempt failed: {str(e)}")
        return None

    def try_direct_download(url):
        """Attempt direct download using requests"""
        try:
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': '*/*'
            }
            response = session.get(url, headers=headers, stream=True)
            if response.ok:
                output_path = os.path.join(temp_dir, 'direct_download.mp3')
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return output_path
        except Exception as e:
            st.warning(f"Direct download attempt failed: {str(e)}")
        return None

    def try_ffmpeg_download(url):
        """Attempt download using ffmpeg"""
        try:
            output_path = os.path.join(temp_dir, 'ffmpeg_download.mp3')
            command = [
                'ffmpeg', '-y',
                '-http_persistent', '0',
                '-timeout', '30',
                '-i', url,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ar', '44100',
                '-ab', '192k',
                '-f', 'mp3',
                output_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            if os.path.exists(output_path):
                return output_path
        except Exception as e:
            st.warning(f"FFmpeg download attempt failed: {str(e)}")
        return None

    # Try all methods with retries
    methods = [try_yt_dlp, try_pytube, try_direct_download, try_ffmpeg_download]
    max_retries = 3
    
    for method in methods:
        for attempt in range(max_retries):
            try:
                st.info(f"Attempting download using {method.__name__} (attempt {attempt + 1}/{max_retries})")
                result = method(url)
                if result and os.path.exists(result):
                    st.success(f"Successfully downloaded using {method.__name__}")
                    return result
                time.sleep(1)  # Add delay between attempts
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed with {method.__name__}: {str(e)}")
                time.sleep(2)  # Add longer delay after failure
    
    raise Exception("Failed to download video after trying all methods")

def extract_frames(video_path, interval=5):
    """Extract frames from video at given interval (seconds)"""
    try:
        frames = []
        timestamps = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(frame_count / fps)
                
            frame_count += 1
            
        cap.release()
        return frames, timestamps
    except Exception as e:
        raise Exception(f"Failed to extract frames: {str(e)}")

def caption_image(frame):
    """Generate caption for an image using the image_captioner"""
    try:
        image = Image.fromarray(frame)
        caption = image_captioner(image)[0]['generated_text']
        return caption
    except Exception as e:
        return f"Failed to generate caption: {str(e)}"

def extract_audio_and_transcribe(video_path):
    """Extract audio from video and transcribe it using ffmpeg"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with st.status("Extracting audio from video..."):
                audio_path = os.path.join(temp_dir, "temp_audio.wav")
                
                try:
                    # Use ffmpeg directly
                    st.info("Extracting audio with ffmpeg...")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le", 
                        "-ar", "16000", "-ac", "1",
                        audio_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    st.success("Audio extraction successful")
                except Exception as ffmpeg_error:
                    raise Exception(f"Audio extraction failed: {str(ffmpeg_error)}")

            with st.status("Transcribing audio..."):
                transcription = transcribe_with_fallbacks(audio_path)
                return transcription

    except Exception as e:
        raise Exception(f"Failed to process audio: {str(e)}")

def transcribe_with_fallbacks(audio_path):
    """Transcribe audio using multiple fallback methods"""
    try:
        # First try Whisper model
        audio_data, sample_rate = sf.read(audio_path)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        result = whisper_model.transcribe(
            audio_data,
            verbose=False,
            condition_on_previous_text=True,
            initial_prompt="This is a transcription of a YouTube video."
        )
        
        if result['text'] and len(result['text']) >= 10:
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
    except Exception as whisper_error:
        st.warning(f"Primary transcription failed: {str(whisper_error)}")

    # Try faster-whisper
    try:
        st.info("Attempting faster-whisper transcription...")
        from faster_whisper import WhisperModel
        
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            initial_prompt="This is a transcription of a YouTube video."
        )
        
        full_text = ""
        segment_list = []
        
        for segment in segments:
            segment_list.append({
                'text': segment.text,
                'start': segment.start,
                'end': segment.end
            })
            full_text += segment.text + " "
        
        return {
            'full_text': full_text.strip(),
            'segments': segment_list
        }
    except Exception as faster_error:
        st.warning(f"faster-whisper failed: {str(faster_error)}")

    # Final fallback to command-line whisper
    try:
        st.info("Trying command-line whisper...")
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
                return {
                    'full_text': result.get('text', ''),
                    'segments': result.get('segments', [])
                }
    except Exception as cmd_error:
        st.error(f"All transcription methods failed: {str(cmd_error)}")

    # Ultimate fallback with empty result
    return {
        'full_text': "Transcription failed. The blog will be generated based on visual content only.",
        'segments': [{'text': "No transcription available", 'start': 0, 'end': 0}]
    }

def generate_enhanced_blog(transcript_text, captions, timestamps):
    """Generate a blog from transcript with support for images"""
    try:
        # Extract main topics from transcript
        sentences = transcript_text.split('.')
        title = "Understanding " + sentences[0].strip()
        
        # Create sections based on content
        content = f"<h1>{title}</h1>\n\n"
        
        # Add introduction
        content += "<h2>Introduction</h2>\n\n"
        content += f"<p>{'. '.join(sentences[:3])}</p>\n\n"
        
        # Split content into sections
        section_length = len(sentences) // 4  # Create 4 main sections
        
        sections = [
            "Key Concepts",
            "Detailed Analysis",
            "Important Insights",
            "Practical Applications"
        ]
        
        # Create sections with content and relevant images
        for i, section_title in enumerate(sections):
            content += f"<h2>{section_title}</h2>\n\n"
            
            start_idx = i * section_length
            end_idx = (i + 1) * section_length
            section_text = '. '.join(sentences[start_idx:end_idx])
            
            # Find relevant images for this section
            section_start_time = (timestamps[0] + (timestamps[-1] - timestamps[0]) * (i / len(sections)))
            section_end_time = (timestamps[0] + (timestamps[-1] - timestamps[0]) * ((i + 1) / len(sections)))
            
            relevant_images = [
                (idx, caption) for idx, (timestamp, caption) in enumerate(zip(timestamps, captions))
                if section_start_time <= timestamp <= section_end_time
            ]
            
            # Add content with images
            paragraphs = section_text.split('. ')
            for j, paragraph in enumerate(paragraphs):
                if j > 0 and j % 2 == 0 and relevant_images:  # Add image every few paragraphs
                    img_idx, caption = relevant_images.pop(0)
                    content += f'<img src="image_{img_idx}.jpg" alt="{caption}" style="max-width: 100%; height: auto; margin: 20px 0;" />\n\n'
                content += f"<p>{paragraph}.</p>\n\n"
        
        # Add conclusion
        content += "<h2>Conclusion</h2>\n\n"
        content += f"<p>{'. '.join(sentences[-3:])}</p>\n\n"
    
        return {
            "title": title,
            "content": content,
            "meta_description": '. '.join(sentences[:2])
        }
    except Exception as e:
        st.error(f"Error in enhanced blog generation: {str(e)}")
        return {
            "title": "Video Content Analysis",
            "content": f"<h1>Video Content Analysis</h1>\n\n<p>{transcript_text}</p>",
            "meta_description": "Analysis of video content"
        }

def analyze_blog(content):
    """Simple blog content analyzer"""
    word_count = len(content.split())
    paragraph_count = content.count('</p>')
    image_count = content.count('<img')
    
    return {
        "statistics": {
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "image_count": image_count,
        },
        "seo_score": "Good" if word_count > 300 and paragraph_count > 5 else "Needs improvement",
        "readability": "Good" if word_count/paragraph_count < 100 else "Needs shorter paragraphs"
    }

def extract_keywords(content):
    """Extract relevant keywords from content for SEO"""
    text = re.sub(r'<[^>]+>', '', content)
    
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
    
    stop_words = set([
        'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for',
        'with', 'as', 'this', 'on', 'are', 'be', 'by', 'an', 'was', 'but',
        'not', 'you', 'from', 'at', 'have', 'has', 'had', 'were', 'they',
        'their', 'what', 'which', 'who', 'when', 'where', 'how', 'why',
        'can', 'all', 'we', 'our', 'us', 'your', 'these', 'those', 'been',
        'would', 'could', 'should', 'will', 'may', 'might', 'must', 'one',
        'two', 'three', 'many', 'some', 'any', 'every', 'also', 'very'
    ])
    
    filtered_words = [word for word in words if word not in stop_words]
    
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return [word for word, count in top_keywords]

def add_schema_markup(content, title, meta_description, keywords):
    """Add schema.org markup for SEO"""
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": meta_description,
        "keywords": ", ".join(keywords),
        "datePublished": datetime.now().strftime("%Y-%m-%d"),
        "author": {
            "@type": "Person",
            "name": "Video to Blog Converter"
        }
    }
    
    schema_script = f'<script type="application/ld+json">{json.dumps(schema)}</script>\n\n'
    
    toc = '<div class="table-of-contents">\n<h3>Table of Contents</h3>\n<ul>\n'
    
    headings = re.findall(r'<h2[^>]*>(.*?)</h2>', content)
    for i, heading in enumerate(headings):
        anchor_id = f"section-{i+1}"
        content = content.replace(
            f'<h2>{heading}</h2>',
            f'<h2 id="{anchor_id}">{heading}</h2>',
            1
        )
        toc += f'<li><a href="#{anchor_id}">{heading}</a></li>\n'
    
    toc += '</ul>\n</div>\n\n'
    
    first_para_end = content.find('</p>')
    first_h1_end = content.find('</h1>')
    
    insert_pos = max(first_para_end, first_h1_end)
    if insert_pos > 0:
        insert_pos += 4
        enhanced_content = content[:insert_pos] + '\n\n' + toc + content[insert_pos:]
    else:
        enhanced_content = toc + content
    
    return schema_script + enhanced_content

def check_api_keys():
    if not CLAUDE_API_KEY:
        st.warning("⚠️ Claude API key not found. Please set it in your environment variables.")
        st.info("You can still use the app without Claude AI, but blog generation will use the offline mode.")
    return bool(CLAUDE_API_KEY)

def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def create_pdf(content, output_path, title="Blog Post"):
    """Convert content to PDF using ReportLab"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        
        # Build PDF content
        elements = []
        
        # Add title
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 20))
        
        # Add content paragraphs
        for paragraph in content.split('\n'):
            if paragraph.strip():
                elements.append(Paragraph(paragraph, styles['Normal']))
                elements.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(elements)
        return True
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return False

def generate_blog_post(video_transcript, video_title):
    """Generate blog post and PDF from video transcript"""
    try:
        # Format the blog content
        blog_content = f"""
# {video_title}

{video_transcript}

---
Generated using AI
        """
        
        # Create PDF file path
        pdf_path = os.path.join("temp", f"{video_title.replace(' ', '_')}.pdf")
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Generate PDF
        if create_pdf(blog_content, pdf_path, video_title):
            # Display success message
            st.success("Blog post and PDF generated successfully!")
            
            # Add download button for PDF
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=f"{video_title.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
            
            # Display blog preview
            st.markdown("### Blog Post Preview")
            st.markdown(blog_content)
            
            return blog_content
        else:
            st.error("Failed to generate PDF")
            return None
            
    except Exception as e:
        st.error(f"Error generating blog post: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("🎥 Video to Blog Converter")
    
    st.markdown("""
    Convert YouTube videos into well-structured blog posts with AI-powered content generation.
    
    ### Features:
    - 🤖 AI-powered video transcription
    - 📷 Intelligent frame extraction
    - ✍️ High-quality blog generation
    - 📊 SEO optimization
    - 📱 Multiple export formats
    """)
    
    st.sidebar.header("Settings")
    
    use_claude = st.sidebar.checkbox(
        "Use Claude for Blog Generation",
        value=True,
        help="Generate high-quality blogs using Claude AI"
    )
    
    template_style = st.sidebar.selectbox(
        "Blog Style",
        ["Modern", "Minimal", "Academic"]
    )
    
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["HTML", "PDF", "MD"]
    )
    
    similarity_threshold = st.sidebar.slider(
        "Image Similarity Threshold",
        0.5, 1.0, 0.75,
        help="Higher values will keep more similar images"
    )
    
    st.markdown("### Enter YouTube URL")
    st.markdown("Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    url = st.text_input("YouTube URL:")
    
    if st.button("Generate Blog") and url:
        with st.spinner("Processing video..."):
            try:
                image_captioner, image_feature_extractor, whisper_model = load_models()
                if not all([image_captioner, whisper_model]):
                    st.error("Failed to load required models. Please check your internet connection and try again.")
                    return
                
                video_path = download_youtube_video(url)
                
                transcription = extract_audio_and_transcribe(video_path)
                
                frames, timestamps = extract_frames(video_path)
                
                if not frames:
                    st.error("No frames could be extracted from the video. Please try another URL.")
                    return
                
                unique_frames, unique_timestamps = deduplicate_frames(frames, timestamps, similarity_threshold)
                
                captions = [caption_image(frame) for frame in unique_frames]
                
                if use_claude:
                    try:
                        st.info("Generating blog using Claude AI...")
                        blog_data = generate_blog_with_claude(
                            transcription,
                            unique_frames,
                            unique_timestamps,
                            captions
                        )
                    except Exception as e:
                        st.warning(f"Claude API failed: {str(e)}. Falling back to offline mode.")
                        blog_data = generate_enhanced_blog(transcription['full_text'], captions, timestamps)
                else:
                    st.info("Generating blog from video transcript...")
                    blog_data = generate_enhanced_blog(transcription['full_text'], captions, timestamps)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    image_files = []
                    for i, frame in enumerate(unique_frames):
                        img_path = os.path.join(temp_dir, f"image_{i}.jpg")
                        if len(frame.shape) == 2:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        else:
                            frame_rgb = frame
                        
                        Image.fromarray(frame_rgb).save(img_path, quality=95, optimize=True)
                        image_files.append(img_path)
                        
                        if i < 5:
                            st.sidebar.image(frame_rgb, caption=f"Frame {i+1}", use_container_width=True)
                    
                    format_type = export_format.lower()
                    if format_type == "md":
                        mime_type = "application/zip"
                        file_ext = "zip"
                    elif format_type == "pdf":
                        mime_type = "application/pdf"
                        file_ext = "pdf"
                    else:
                        mime_type = "text/html"
                        file_ext = "html"
                    
                    exported_content, actual_mime_type = export_blog(
                        blog_data,
                        format_type,
                        template_style,
                        image_files
                    )
                    
                    mime_type = actual_mime_type
                    
                    if exported_content:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"blog_{timestamp}.{file_ext}"
                        
                        if format_type == "md":
                            st.info("Markdown export includes a ZIP file with the blog post and all images.")
                        elif format_type == "pdf" and mime_type == "text/html":
                            st.warning("PDF generation failed. Falling back to HTML format.")
                            filename = f"blog_{timestamp}.html"
                        
                        st.download_button(
                            f"Download {export_format}",
                            exported_content,
                            filename,
                            mime_type
                        )
                
                st.subheader("Blog Preview")
                if mime_type == 'text/html':
                    st.components.v1.html(exported_content, height=600, scrolling=True)
                else:
                    preview_content = blog_data['content']
                    preview_content = re.sub(r'<[^>]+>', ' ', preview_content)
                    st.text_area("Content Preview", preview_content[:1000] + "...", height=300)
                
                analysis = analyze_blog(blog_data['content'])
                st.subheader("Blog Analysis")
                st.json(analysis)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
            finally:
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.remove(video_path)

if __name__ == "__main__":
    # Setup proper async handling
    loop = init_async()
    
    try:
        # Your main application code
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
