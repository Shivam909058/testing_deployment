import os
import argparse
import logging
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

# Third-party imports
import streamlit as st
from langchain.llms import Anthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class YouTubeBlogGenerator:
    """Generate high-quality blogs from YouTube videos using LangChain and Claude"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube Blog Generator.
        
        Args:
            api_key: Claude API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key is required. Set it as an argument or in the CLAUDE_API_KEY environment variable.")
        
        # Initialize Claude LLM
        self.llm = Anthropic(
            model="claude-3-5-sonnet-20240620",
            anthropic_api_key=self.api_key,
            temperature=0.7,
            max_tokens_to_sample=4000
        )
        
        logger.info("YouTube Blog Generator initialized")
    
    def extract_video_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a YouTube video using LangChain's YoutubeLoader.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video metadata and transcript
        """
        logger.info(f"Extracting content from YouTube video: {url}")
        
        try:
            # Load video transcript
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True,
                language=["en", "en-US"]
            )
            documents = loader.load()
            
            # Extract video metadata
            video_info = {
                "title": documents[0].metadata.get("title", "Untitled Video"),
                "author": documents[0].metadata.get("author", "Unknown"),
                "publish_date": documents[0].metadata.get("publish_date", "Unknown"),
                "description": documents[0].metadata.get("description", ""),
                "view_count": documents[0].metadata.get("view_count", 0),
                "thumbnail_url": documents[0].metadata.get("thumbnail_url", ""),
                "transcript": documents[0].page_content
            }
            
            logger.info(f"Successfully extracted content from video: {video_info['title']}")
            return video_info
            
        except Exception as e:
            logger.error(f"Error extracting video content: {str(e)}")
            raise Exception(f"Failed to extract video content: {str(e)}")
    
    def summarize_transcript(self, transcript: str) -> str:
        """
        Generate a concise summary of the video transcript.
        
        Args:
            transcript: Video transcript text
            
        Returns:
            Summarized transcript
        """
        logger.info("Summarizing video transcript")
        
        try:
            # Split transcript into chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=400
            )
            texts = text_splitter.split_text(transcript)
            docs = [Document(page_content=t) for t in texts]
            
            # Create summarization chain
            chain = load_summarize_chain(
                self.llm, 
                chain_type="map_reduce",
                verbose=False
            )
            
            # Generate summary
            summary = chain.run(docs)
            logger.info("Transcript summarization completed")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing transcript: {str(e)}")
            # Return a portion of the transcript if summarization fails
            return transcript[:2000] + "..."
    
    def analyze_video_content(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze video content to extract key topics, themes, and insights.
        
        Args:
            video_info: Dictionary with video metadata and transcript
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing video content")
        
        try:
            # Create analysis prompt
            analysis_prompt = PromptTemplate(
                input_variables=["title", "transcript"],
                template="""
                Analyze this YouTube video content and identify:
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

                Video Title: {title}

                Transcript:
                {transcript}

                Provide a detailed analysis in JSON format with these sections.
                """
            )
            
            # Create analysis chain
            analysis_chain = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt,
                verbose=False
            )
            
            # Generate analysis
            analysis_result = analysis_chain.run(
                title=video_info["title"],
                transcript=video_info["transcript"][:10000]  # Use first 10k chars for analysis
            )
            
            # Try to parse JSON response
            try:
                # Extract JSON if it's embedded in text
                json_match = re.search(r'```json\s*(.*?)\s*```', analysis_result, re.DOTALL)
                if json_match:
                    analysis_result = json_match.group(1)
                
                # Clean up any non-JSON text
                json_start = analysis_result.find('{')
                json_end = analysis_result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    analysis_result = analysis_result[json_start:json_end]
                
                analysis_data = json.loads(analysis_result)
                logger.info("Successfully parsed analysis JSON")
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse analysis as JSON, using raw text")
                analysis_data = {"raw_analysis": analysis_result}
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            return {"error": str(e)}
    
    def generate_blog_post(self, video_info: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a high-quality blog post based on video content and analysis.
        
        Args:
            video_info: Dictionary with video metadata and transcript
            analysis: Dictionary with content analysis
            
        Returns:
            Dictionary with blog title, meta description, and content
        """
        logger.info("Generating blog post")
        
        try:
            # Create blog generation prompt
            blog_prompt = PromptTemplate(
                input_variables=["title", "transcript", "analysis", "summary"],
                template="""
                You are an expert content writer specializing in creating comprehensive, SEO-optimized blog posts from YouTube videos.
                
                Create a high-quality, engaging blog post based on this YouTube video content.
                
                Video Title: {title}
                
                Video Summary:
                {summary}
                
                Content Analysis:
                {analysis}
                
                Full Transcript:
                {transcript}
                
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
                
                3. SEO Optimization:
                   - Use semantic keywords naturally
                   - Optimize header hierarchy (H1, H2, H3)
                   - Include internal linking suggestions
                   - Add meta description
                   - Use LSI keywords
                   - Optimize for featured snippets
                   - Include a meta description (150-160 characters)
                
                4. Engagement Elements:
                   - Add thought-provoking questions throughout
                   - Include highlighted key quotes
                   - Use bullet points for lists and takeaways
                   - Create info boxes for important concepts
                   - Add "Tweet This" quotes
                   - Include expert tips boxes
                   - Use examples and analogies for complex concepts
                
                Format the response as a JSON object with:
                {{
                    "title": "SEO-optimized title",
                    "meta_description": "Compelling 150-160 character description",
                    "content": "Full HTML blog content"
                }}
                
                Make the content comprehensive, engaging, and highly valuable to readers.
                Focus on maintaining the speaker's voice while adding professional polish.
                """
            )
            
            # Create summary if not already available
            summary = self.summarize_transcript(video_info["transcript"]) if "summary" not in video_info else video_info["summary"]
            
            # Create blog generation chain
            blog_chain = LLMChain(
                llm=self.llm,
                prompt=blog_prompt,
                verbose=False
            )
            
            # Generate blog post
            blog_result = blog_chain.run(
                title=video_info["title"],
                transcript=video_info["transcript"][:15000],  # Use first 15k chars of transcript
                analysis=json.dumps(analysis, indent=2),
                summary=summary
            )
            
            # Try to parse JSON response
            try:
                # Extract JSON if it's embedded in text
                json_match = re.search(r'```json\s*(.*?)\s*```', blog_result, re.DOTALL)
                if json_match:
                    blog_result = json_match.group(1)
                
                # Clean up any non-JSON text
                json_start = blog_result.find('{')
                json_end = blog_result.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    blog_result = blog_result[json_start:json_end]
                
                blog_data = json.loads(blog_result)
                logger.info("Successfully parsed blog JSON")
            except json.JSONDecodeError:
                # If JSON parsing fails, extract content manually
                logger.warning("Failed to parse blog as JSON, extracting content manually")
                blog_data = self.extract_blog_content(blog_result)
            
            # Add schema markup
            blog_data["content"] = self.add_schema_markup(
                blog_data["content"],
                blog_data["title"],
                blog_data["meta_description"]
            )
            
            logger.info("Blog post generation completed")
            return blog_data
            
        except Exception as e:
            logger.error(f"Error generating blog post: {str(e)}")
            return {
                "title": video_info["title"],
                "meta_description": "Analysis of video content",
                "content": f"<h1>{video_info['title']}</h1>\n\n<p>Error generating blog content: {str(e)}</p>"
            }
    
    def extract_blog_content(self, content: str) -> Dict[str, str]:
        """
        Extract blog content from LLM response when JSON parsing fails.
        
        Args:
            content: Raw response content
            
        Returns:
            Dictionary with blog data
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
    
    def add_schema_markup(self, content: str, title: str, meta_description: str) -> str:
        """
        Add schema.org markup for SEO.
        
        Args:
            content: HTML content
            title: Blog title
            meta_description: Meta description
            
        Returns:
            Enhanced HTML with schema markup
        """
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": meta_description,
            "datePublished": datetime.now().strftime("%Y-%m-%d"),
            "author": {
                "@type": "Person",
                "name": "YouTube Blog Generator"
            }
        }
        
        schema_script = f'<script type="application/ld+json">{json.dumps(schema)}</script>\n\n'
        
        # Add table of contents
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
    
    def generate_html(self, blog_data: Dict[str, str]) -> str:
        """
        Generate HTML from blog data.
        
        Args:
            blog_data: Dictionary with blog data
            
        Returns:
            HTML content
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="description" content="{meta_description}">
            <title>{title}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; margin-bottom: 20px; }}
                h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
                h3 {{ color: #7f8c8d; margin-top: 25px; }}
                p {{ margin-bottom: 15px; }}
                .table-of-contents {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .table-of-contents ul {{ padding-left: 20px; }}
                .table-of-contents h3 {{ margin-top: 0; }}
                blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; margin-left: 0; color: #555; }}
                code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                ul, ol {{ padding-left: 25px; margin-bottom: 15px; }}
                .highlight {{ background-color: #ffffd0; padding: 10px; border-radius: 5px; margin: 15px 0; }}
                .tip-box {{ background-color: #e1f5fe; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .tweet-this {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
        
        html_content = html_template.format(
            title=blog_data['title'],
            content=blog_data['content'],
            meta_description=blog_data['meta_description']
        )
        
        return html_content
    
    def process_youtube_url(self, url: str) -> Dict[str, Any]:
        """
        Process a YouTube URL to generate a blog post.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with blog data and HTML content
        """
        logger.info(f"Processing YouTube URL: {url}")
        
        # Extract video content
        video_info = self.extract_video_content(url)
        
        # Analyze video content
        analysis = self.analyze_video_content(video_info)
        
        # Generate blog post
        blog_data = self.generate_blog_post(video_info, analysis)
        
        # Generate HTML
        html_content = self.generate_html(blog_data)
        
        return {
            "video_info": video_info,
            "analysis": analysis,
            "blog_data": blog_data,
            "html_content": html_content
        }

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="YouTube Blog Generator",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for API key
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # App header
    st.title("📝 YouTube Blog Generator")
    
    st.markdown("""
    Generate high-quality, SEO-optimized blog posts from YouTube videos.
    
    This tool uses AI to understand the video content and create comprehensive blog posts.
    """)
    
    # API key input section
    st.header("🔑 API Key Authentication")
    
    with st.expander("Enter your Claude API Key", expanded=not st.session_state.api_key_valid):
        api_key = st.text_input(
            "Claude API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Claude API key from Anthropic. Your key is not stored on our servers."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Validate Key"):
                if api_key:
                    try:
                        # Test the API key with a simple request
                        test_llm = Anthropic(
                            model="claude-3-5-sonnet-20240620",
                            anthropic_api_key=api_key,
                            temperature=0.7,
                            max_tokens_to_sample=10
                        )
                        
                        # Try a simple completion to validate the key
                        test_prompt = PromptTemplate(
                            input_variables=[],
                            template="Say 'API key is valid' in 5 words or less."
                        )
                        test_chain = LLMChain(llm=test_llm, prompt=test_prompt)
                        test_chain.run({})
                        
                        # If we get here, the key is valid
                        st.session_state.api_key = api_key
                        st.session_state.api_key_valid = True
                        st.success("✅ API key validated successfully!")
                    except Exception as e:
                        st.session_state.api_key_valid = False
                        st.error(f"❌ Invalid API key: {str(e)}")
                else:
                    st.session_state.api_key_valid = False
                    st.error("❌ Please enter an API key")
        
        with col2:
            st.markdown("""
            **Don't have a Claude API key?** 
            [Sign up for Anthropic Claude API access](https://www.anthropic.com/api)
            """)
    
    # Main application - only show if API key is valid
    if st.session_state.api_key_valid:
        st.header("🎬 Generate Blog from YouTube Video")
        
        # YouTube URL input
        url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter the full URL of the YouTube video you want to convert to a blog post"
        )
        
        # Blog style options
        col1, col2 = st.columns(2)
        with col1:
            blog_style = st.selectbox(
                "Blog Style",
                ["Professional", "Casual", "Technical", "Educational"],
                help="Select the writing style for your blog"
            )
        
        with col2:
            blog_length = st.selectbox(
                "Blog Length",
                ["Standard (1000-1500 words)", "Detailed (1500-2500 words)", "Comprehensive (2500+ words)"],
                help="Select the desired length for your blog post"
            )
        
        # Generate blog button
        if st.button("Generate Blog", type="primary", disabled=st.session_state.processing):
            if not url:
                st.error("Please enter a YouTube URL")
            else:
                st.session_state.processing = True
                
                try:
                    with st.spinner("Processing video content..."):
                        # Initialize blog generator
                        generator = YouTubeBlogGenerator(api_key=st.session_state.api_key)
                        
                        # Process YouTube URL
                        result = generator.process_youtube_url(url)
                        st.session_state.result = result
                        
                    st.session_state.processing = False
                    st.success("✅ Blog generated successfully!")
                    
                except Exception as e:
                    st.session_state.processing = False
                    st.error(f"❌ An error occurred: {str(e)}")
                    logger.error(f"Error in main app: {str(e)}", exc_info=True)
        
        # Display results if available
        if st.session_state.result:
            result = st.session_state.result
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Blog Preview", "Video Analysis", "Export Options"])
            
            with tab1:
                st.subheader(result['blog_data']['title'])
                st.markdown(f"*{result['blog_data']['meta_description']}*")
                st.components.v1.html(result['html_content'], height=600, scrolling=True)
            
            with tab2:
                st.subheader("Video Information")
                st.write(f"**Title:** {result['video_info']['title']}")
                st.write(f"**Author:** {result['video_info']['author']}")
                st.write(f"**Published:** {result['video_info']['publish_date']}")
                
                # Display analysis if available
                if 'main_topics' in result['analysis']:
                    st.subheader("Main Topics")
                    for topic in result['analysis'].get('main_topics', []):
                        st.markdown(f"- {topic}")
                
                # Display transcript preview
                st.subheader("Transcript Preview")
                st.text_area(
                    "Video Transcript (Preview)",
                    result['video_info']['transcript'][:1000] + "...",
                    height=200
                )
            
            with tab3:
                st.subheader("Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # HTML download
                    st.download_button(
                        label="Download as HTML",
                        data=result['html_content'],
                        file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        help="Download the blog post as an HTML file"
                    )
                
                with col2:
                    # JSON download
                    blog_json = json.dumps({
                        "title": result['blog_data']['title'],
                        "meta_description": result['blog_data']['meta_description'],
                        "content": result['blog_data']['content'],
                        "video_info": {
                            "title": result['video_info']['title'],
                            "author": result['video_info']['author'],
                            "publish_date": result['video_info']['publish_date'],
                            "url": url
                        }
                    }, indent=2)
                    
                    st.download_button(
                        label="Download as JSON",
                        data=blog_json,
                        file_name=f"blog_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download the blog data as a JSON file"
                    )
                
                # Markdown conversion option
                st.subheader("Convert to Markdown")
                if st.button("Generate Markdown"):
                    # Convert HTML to Markdown
                    try:
                        import html2text
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = False
                        h.body_width = 0  # No wrapping
                        
                        markdown_content = h.handle(result['blog_data']['content'])
                        
                        st.download_button(
                            label="Download Markdown",
                            data=markdown_content,
                            file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    except ImportError:
                        st.error("html2text library not available. Please install it to enable Markdown conversion.")
    else:
        # Show message if API key is not validated
        st.info("👆 Please enter and validate your Claude API key to use the application")
        
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
        YouTube Blog Generator | Powered by Claude AI | Created with Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

# Command-line interface
def cli():
    parser = argparse.ArgumentParser(description="Generate a blog post from a YouTube video")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", help="Output file path (default: blog_output.html)")
    parser.add_argument("--api-key", help="Claude API key (default: uses CLAUDE_API_KEY environment variable)")
    args = parser.parse_args()
    
    try:
        # Initialize blog generator
        generator = YouTubeBlogGenerator(api_key=args.api_key)
        
        # Process YouTube URL
        result = generator.process_youtube_url(args.url)
        
        # Save output
        output_path = args.output or "blog_output.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["html_content"])
        
        print(f"Blog post generated and saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error in CLI: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    
    # Check if running as script or Streamlit app
    if st._is_running_with_streamlit:
        main()
    else:
        sys.exit(cli())