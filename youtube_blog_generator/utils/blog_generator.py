import json
import re
import time
import logging
from datetime import datetime
import random
import html
from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlogGenerator:
    def __init__(self, use_advanced_features=True):
        """
        Initialize the blog generator with configuration options.
        
        Args:
            use_advanced_features: Whether to use advanced formatting and SEO features
        """
        self.use_advanced_features = use_advanced_features
        self.blog_templates = self._load_blog_templates()
        
    def _load_blog_templates(self) -> Dict[str, str]:
        """Load templates for different blog styles"""
        return {
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
            </head>
            <body>
                {{ content|safe }}
            </body>
            </html>
            """
        }
    
    def generate_blog_with_ai(self, 
                             transcription: Dict[str, Any], 
                             frames: List[Any], 
                             timestamps: List[float], 
                             captions: List[str],
                             client: Any,
                             model_name: str = "claude-3-5-sonnet-20240620") -> Dict[str, str]:
        """
        Generate a high-quality blog based on video content using an AI service.
        
        Args:
            transcription: Dictionary containing full_text and segments
            frames: List of video frames
            timestamps: List of timestamps for each frame
            captions: List of captions for each frame
            client: API client for AI service
            model_name: Name of the model to use
            
        Returns:
            Dict containing title, meta_description, and content
        """
        try:
            logger.info("Starting AI-powered blog generation")
            
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
                model=model_name,
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
                    if abs(seg['start'] - timestamp) < 15  # 15 second window
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

            # Generate blog with AI
            response = client.messages.create(
                model=model_name,
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
                blog_data = self._extract_blog_content(content)

            # Add schema markup and enhance content
            keywords = self._extract_keywords(blog_data['content'])
            blog_data['content'] = self._add_schema_markup(
                blog_data['content'],
                blog_data['title'],
                blog_data['meta_description'],
                keywords
            )

            logger.info("AI blog generation completed successfully")
            return blog_data

        except Exception as e:
            logger.error(f"Error with Claude API: {str(e)}")
            logger.info("Falling back to offline blog generation...")
            return self.generate_enhanced_blog(transcription['full_text'], captions, timestamps)
    
    def generate_enhanced_blog(self, 
                              transcript_text: str, 
                              captions: List[str], 
                              timestamps: List[float]) -> Dict[str, str]:
        """
        Generate a well-structured blog from transcript with support for images.
        This is used as a fallback when AI generation fails.
        
        Args:
            transcript_text: Full transcript text
            captions: List of image captions
            timestamps: List of timestamp for each caption/image
            
        Returns:
            Dict containing title, meta_description, and content
        """
        try:
            logger.info("Generating enhanced blog from transcript")
            # Extract main topics from transcript
            sentences = transcript_text.split('.')
            title = "Understanding " + sentences[0].strip()
            
            # Create sections based on content
            content = f"<h1>{title}</h1>\n\n"
            
            # Add introduction
            content += "<h2>Introduction</h2>\n\n"
            content += f"<p>{'. '.join(sentences[:3])}</p>\n\n"
            
            # Split content into sections
            section_length = max(1, len(sentences) // 4)  # Create 4 main sections
            
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
                end_idx = min((i + 1) * section_length, len(sentences))
                section_text = '. '.join(sentences[start_idx:end_idx])
                
                # Find relevant images for this section
                if timestamps and len(timestamps) > 0:
                    section_start_time = (timestamps[0] + (timestamps[-1] - timestamps[0]) * (i / len(sections)))
                    section_end_time = (timestamps[0] + (timestamps[-1] - timestamps[0]) * ((i + 1) / len(sections)))
                    
                    relevant_images = [
                        (idx, caption) for idx, (timestamp, caption) in enumerate(zip(timestamps, captions))
                        if section_start_time <= timestamp <= section_end_time
                    ]
                else:
                    relevant_images = []
                
                # Add content with images
                paragraphs = section_text.split('. ')
                for j, paragraph in enumerate(paragraphs):
                    if j > 0 and j % 2 == 0 and relevant_images:  # Add image every few paragraphs
                        img_idx, caption = relevant_images.pop(0)
                        content += f'<img src="image_{img_idx}.jpg" alt="{caption}" style="max-width: 100%; height: auto; margin: 20px 0;" />\n\n'
                    if paragraph.strip():
                        content += f"<p>{paragraph}.</p>\n\n"
            
            # Add conclusion
            content += "<h2>Conclusion</h2>\n\n"
            content += f"<p>{'. '.join(sentences[-3:])}</p>\n\n"
            
            # Add table of contents if advanced features are enabled
            if self.use_advanced_features:
                content = self._add_table_of_contents(content)
                
            # Extract meta description
            meta_description = '. '.join(sentences[:2])
            if len(meta_description) > 160:
                meta_description = meta_description[:157] + "..."
        
            return {
                "title": title,
                "content": content,
                "meta_description": meta_description
            }
        except Exception as e:
            logger.error(f"Error in enhanced blog generation: {str(e)}")
            return {
                "title": "Video Content Analysis",
                "content": f"<h1>Video Content Analysis</h1>\n\n<p>{transcript_text}</p>",
                "meta_description": "Analysis of video content"
            }
    
    def _extract_blog_content(self, content: str) -> Dict[str, str]:
        """
        Extract blog content from an AI response when JSON parsing fails.
        
        Args:
            content: Raw response content
            
        Returns:
            Dict containing title, meta_description, and content
        """
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
            logger.error(f"Error extracting blog content: {str(e)}")
            return {
                "title": "Video Analysis",
                "meta_description": "Analysis of video content",
                "content": content
            }

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract relevant keywords from content for SEO.
        
        Args:
            content: HTML content
            
        Returns:
            List of keywords
        """
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

    def _add_schema_markup(self, 
                          content: str, 
                          title: str, 
                          meta_description: str, 
                          keywords: List[str]) -> str:
        """
        Add schema.org markup for SEO.
        
        Args:
            content: HTML content
            title: Blog title
            meta_description: Meta description
            keywords: List of keywords
            
        Returns:
            Enhanced HTML with schema markup
        """
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
        
        # Create a table of contents
        content = self._add_table_of_contents(content)
        
        return schema_script + content

    def _add_table_of_contents(self, content: str) -> str:
        """
        Add a table of contents to the blog.
        
        Args:
            content: HTML content
            
        Returns:
            Content with table of contents
        """
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
        
        return enhanced_content
    
    def analyze_blog(self, content: str) -> Dict[str, Any]:
        """
        Analyze blog content for statistics and quality metrics.
        
        Args:
            content: HTML content
            
        Returns:
            Dict with stats, SEO score, and readability score
        """
        word_count = len(re.sub(r'<[^>]+>', '', content).split())
        paragraph_count = content.count('</p>')
        image_count = content.count('<img')
        heading_count = content.count('<h2') + content.count('<h3')
        
        # Calculate average paragraph length
        avg_paragraph_length = word_count / max(1, paragraph_count)
        
        # Calculate keyword density
        keywords = self._extract_keywords(content)
        keyword_density = {}
        text_content = re.sub(r'<[^>]+>', '', content).lower()
        for keyword in keywords:
            count = text_content.count(keyword)
            density = count / max(1, word_count) * 100
            keyword_density[keyword] = round(density, 2)
        
        # Determine SEO score
        seo_score = "Good" if (
            word_count > 1000 and 
            paragraph_count > 10 and
            heading_count > 4 and
            image_count > 2
        ) else "Needs improvement"
        
        # Determine readability
        readability = "Good" if avg_paragraph_length < 100 else "Needs shorter paragraphs"
        
        return {
            "statistics": {
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "image_count": image_count,
                "heading_count": heading_count,
                "avg_paragraph_length": round(avg_paragraph_length, 2)
            },
            "keyword_density": keyword_density,
            "seo_score": seo_score,
            "readability": readability
        }
    
    def export_html(self, 
                   blog_data: Dict[str, str], 
                   template_name: str = "Modern") -> str:
        """
        Export blog as HTML using a template.
        
        Args:
            blog_data: Dict with title, content, and meta_description
            template_name: Name of template to use
            
        Returns:
            HTML content
        """
        template = self.blog_templates.get(template_name, self.blog_templates["Modern"])
        
        html_content = template.replace("{{ title }}", blog_data['title'])
        html_content = html_content.replace("{{ content|safe }}", blog_data['content'])
        html_content = html_content.replace("{{ meta_description }}", blog_data['meta_description'])
        
        return html_content
    
    def export_markdown(self, 
                       blog_data: Dict[str, str], 
                       image_count: int) -> str:
        """
        Export blog as markdown.
        
        Args:
            blog_data: Dict with title, content, and meta_description
            image_count: Number of images in the blog
            
        Returns:
            Markdown content
        """
        content = blog_data['content']
        md_content = f"# {blog_data['title']}\n\n"
        md_content += f"_{blog_data['meta_description']}_\n\n"
        
        # Convert HTML to Markdown
        for i in range(image_count):
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
        
        # Extract headings for TOC
        md_content += f"## Table of Contents\n\n"
        headings = re.findall(r'<h[2-3][^>]*>(.*?)</h[2-3]>', blog_data['content'])
        for heading in headings:
            anchor = heading.lower().replace(' ', '-').replace('.', '').replace(',', '')
            md_content += f"- [{heading}](#{anchor})\n"
        
        md_content += "\n\n" + content
        
        md_content += f"\n\n---\n\n"
        md_content += f"*Keywords: {', '.join(self._extract_keywords(blog_data['content']))}\n\n"
        md_content += f"*Published: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        return md_content

def generate_blog_from_ai(transcript, frames=None, timestamps=None, captions=None, client=None, model_name=None):
    """
    Standalone function to generate a blog from a video transcript using AI.
    
    Args:
        transcript: Dictionary containing full_text and segments
        frames: List of video frames (optional)
        timestamps: List of timestamps for each frame (optional)
        captions: List of captions for each frame (optional)
        client: AI client (optional)
        model_name: Name of the AI model to use (optional)
        
    Returns:
        Dict containing title, meta_description, and content
    """
    generator = BlogGenerator()
    
    if client and frames and timestamps and captions:
        return generator.generate_blog_with_ai(
            transcript, frames, timestamps, captions, client, model_name
        )
    else:
        # Use offline generation if AI client not provided
        transcript_text = transcript['full_text'] if isinstance(transcript, dict) else transcript
        return generator.generate_enhanced_blog(
            transcript_text, 
            captions or [], 
            timestamps or []
        )

def generate_simple_blog(title, transcript, image_paths=None):
    """
    Generate a simple blog from a title and transcript.
    
    Args:
        title: Blog title
        transcript: Transcript text
        image_paths: List of paths to images (optional)
        
    Returns:
        HTML content
    """
    content = f"<h1>{html.escape(title)}</h1>\n\n"
    
    # Split transcript into paragraphs
    paragraphs = transcript.split('\n\n')
    
    # Add introduction
    if paragraphs:
        content += "<h2>Introduction</h2>\n\n"
        content += f"<p>{html.escape(paragraphs[0])}</p>\n\n"
    
    # Add main content with images
    if len(paragraphs) > 1:
        content += "<h2>Content</h2>\n\n"
        
        for i, paragraph in enumerate(paragraphs[1:]):
            content += f"<p>{html.escape(paragraph)}</p>\n\n"
            
            # Add an image after some paragraphs if available
            if image_paths and i < len(image_paths) and (i % 2 == 0):
                img_path = os.path.basename(image_paths[i])
                content += f'<img src="{img_path}" alt="Content visual {i+1}" style="max-width: 100%; height: auto; margin: 20px 0;" />\n\n'
    
    # Add conclusion
    if paragraphs:
        content += "<h2>Conclusion</h2>\n\n"
        content += f"<p>{html.escape(paragraphs[-1])}</p>\n\n"
    
    # Create blog data
    blog_data = {
        "title": title,
        "content": content,
        "meta_description": paragraphs[0][:160] if paragraphs else "Blog post from video content"
    }
    
    # Generate HTML
    generator = BlogGenerator()
    return generator.export_html(blog_data)

if __name__ == "__main__":
    # Example usage
    test_transcript = {
        "full_text": "This is a sample transcript. It contains information about a topic. This is the second sentence. Here's more content for testing purposes. This would normally be much longer in a real video.",
        "segments": [
            {"text": "This is a sample transcript.", "start": 0, "end": 3},
            {"text": "It contains information about a topic.", "start": 3, "end": 6},
            {"text": "This is the second sentence.", "start": 6, "end": 9},
            {"text": "Here's more content for testing purposes.", "start": 9, "end": 12},
            {"text": "This would normally be much longer in a real video.", "start": 12, "end": 15}
        ]
    }
    
    # Test offline blog generation
    generator = BlogGenerator()
    blog = generator.generate_enhanced_blog(
        test_transcript["full_text"], 
        ["Test caption 1", "Test caption 2"], 
        [3.0, 9.0]
    )
    
    print("Generated blog title:", blog["title"])
    print("Generated blog meta description:", blog["meta_description"])
    print("Content length:", len(blog["content"]))
    
    # Export as HTML
    html_output = generator.export_html(blog)
    print("HTML length:", len(html_output))
    
    # Export as Markdown
    md_output = generator.export_markdown(blog, 2)
    print("Markdown length:", len(md_output)) 