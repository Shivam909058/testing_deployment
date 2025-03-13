#!/usr/bin/env python3
"""
YouTube to Blog Converter
-------------------------
This script converts YouTube videos into SEO-optimized, authentic blog posts.

################ IT NEEDS FURTHER IMPROVEMENT #################

# IT IS JUST STARTING PHASE TO GET STARTED AND GET BLOG SCORES 


"""



"""

WHY WE ARE NOT USING FINE TUNNING TRANSFORMER MODELS FOR THIS TASK?
-->> 

so the main thing is that most of the large language models are already trained on large datasets and they are already good at generating text and understanding the context of the text.
so we don't need to fine-tune them for this task. we can directly use them to generate the blog post. we just need optimize the internal functionalties of the model to get the best results.

soo still it is just a starting phase to get started and get the blog scores.


"""

import os
import re
import time
import argparse
from urllib.parse import urlparse, parse_qs
import subprocess
import json
import os
import concurrent.futures
import hashlib
import textstat
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
from moviepy.editor import AudioFileClip
from dotenv import load_dotenv
from moviepy import *
import yt_dlp
load_dotenv()
openai = OpenAI(api_key="")
nltk.download("stopwords", quiet=True)

class YouTubeToBlog:
    def __init__(self, video_url, output_dir="output", temp_dir="temp", cache_dir="cache"):
        self.video_url = video_url
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.cache_dir = cache_dir
        self.video_id = self._extract_video_id(video_url)
        self.video_title = ""
        self.video_author = ""
        self.transcript = ""
        self.blog_content = ""
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
    
    def _extract_video_id(self, url):
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            if parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            if parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _get_cache_path(self, prefix, extension="txt"):
        return os.path.join(self.cache_dir, f"{prefix}_{self.video_id}.{extension}")
    
    def _check_cache(self, cache_type):
        cache_path = self._get_cache_path(cache_type)
        if os.path.exists(cache_path):
            print(f"Found cached {cache_type} for video {self.video_id}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def _save_to_cache(self, cache_type, content):
        cache_path = self._get_cache_path(cache_type)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved {cache_type} to cache")
    
    def download_video_audio(self):
        print(f"Downloading audio from YouTube video: {self.video_url}")
        try:
            
            info_cmd = [
                "yt-dlp", 
                "--dump-json",
                "--no-playlist",
                self.video_url
            ]
            
            try:
                result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
                video_info = json.loads(result.stdout)
                self.video_title = video_info.get('title', f"video_{self.video_id}")
                self.video_author = video_info.get('uploader', "Unknown Creator")
                print(f"Video title: {self.video_title}")
                print(f"Video author: {self.video_author}")
            except Exception as e:
                print(f"Warning: Could not get video info: {str(e)}")
                self.video_title = f"video_{self.video_id}"
                self.video_author = "Unknown Creator"
            
            audio_path = os.path.join(self.temp_dir, f"{self.video_id}.webm")
            download_cmd = [
                "yt-dlp",
                "-f", "bestaudio",  # Best audio format
                "-o", audio_path,
                "--no-playlist",
                self.video_url
            ]
            
            subprocess.run(download_cmd, check=True)
            
            print(f"Audio downloaded: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading video: {str(e)}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            raise Exception(f"Failed to download video: {str(e)}")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise
    
    def _transcribe_chunk(self, chunk_path):
        try:
            with open(chunk_path, "rb") as chunk_file:
                chunk_transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=chunk_file
                )
            return chunk_transcript.text
        except Exception as e:
            print(f"Error transcribing chunk {chunk_path}: {str(e)}")
            return ""
    
    def transcribe_audio(self, audio_file_path):
        cached_transcript = self._check_cache("transcript")
        if cached_transcript:
            self.transcript = cached_transcript
            return cached_transcript
        
        print("Transcribing audio...")
        try:
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            if file_size_mb > 24:  # Leave some margin
                print(f"Audio file is {file_size_mb:.1f}MB, which exceeds OpenAI's limit.")
                print("Splitting audio into chunks for parallel transcription...")
                
                chunks_dir = os.path.join(self.temp_dir, "chunks")
                os.makedirs(chunks_dir, exist_ok=True)
                
                audio = AudioFileClip(audio_file_path)
                duration = audio.duration
                chunk_size = 5 * 60  # 5 minutes in seconds
                
                chunk_paths = []
                for i, start_time in enumerate(range(0, int(duration), chunk_size)):
                    end_time = min(start_time + chunk_size, duration)
                    print(f"Preparing chunk {i+1}: {start_time/60:.1f}min to {end_time/60:.1f}min")
                    
                    chunk = audio.subclip(start_time, end_time)
                    chunk_path = os.path.join(chunks_dir, f"chunk_{i}.mp3")
                    chunk.write_audiofile(chunk_path, verbose=False, logger=None)
                    chunk_paths.append(chunk_path)
                
                print(f"Transcribing {len(chunk_paths)} chunks in parallel...")
                chunk_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chunk_paths))) as executor:
                    future_to_chunk = {executor.submit(self._transcribe_chunk, chunk_path): i 
                                      for i, chunk_path in enumerate(chunk_paths)}
                    
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            chunk_text = future.result()
                            chunk_results.append((chunk_idx, chunk_text))
                            print(f"Chunk {chunk_idx+1}/{len(chunk_paths)} transcribed")
                        except Exception as e:
                            print(f"Chunk {chunk_idx+1} failed: {str(e)}")
                
                chunk_results.sort(key=lambda x: x[0])
                full_transcript = " ".join(text for _, text in chunk_results)
                
                for chunk_path in chunk_paths:
                    os.remove(chunk_path)
                os.rmdir(chunks_dir)
                
                self.transcript = full_transcript.strip()
            else:
                with open(audio_file_path, "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                self.transcript = transcript.text
            
            print("Transcription complete!")
            
            transcript_path = os.path.join(self.output_dir, f"{self.video_id}_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(self.transcript)
            
            self._save_to_cache("transcript", self.transcript)
            
            return self.transcript
            
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            raise
    
    def generate_blog_post(self):
        print("Generating comprehensive blog post...")
        try:
            # First, analyze the transcript for key topics
            topic_analysis_prompt = f"""
            Analyze this video transcript and identify:
            1. Main topics and themes
            2. Key points and insights
            3. Supporting examples or data
            4. Technical concepts explained
            5. Practical applications discussed

            Transcript:
            {self.transcript}

            Return the analysis as a structured list.
            """

            # Get topic analysis
            topic_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content analyst."},
                    {"role": "user", "content": topic_analysis_prompt}
                ],
                temperature=0.3
            )
            
            topic_analysis = topic_response.choices[0].message.content

            # Now generate the comprehensive blog
            blog_prompt = f"""
            Create a comprehensive, long-form blog post (2000-3000 words) based on this video transcript.
            
            Video Title: {self.video_title}
            Video Creator: {self.video_author}
            
            Topic Analysis:
            {topic_analysis}

            Requirements for the blog:

            1. Structure:
               - Compelling headline that includes main keyword
               - Executive summary (300-500 words)
               - Table of contents
               - 5-7 main sections with descriptive subheadings
               - Conclusion with key takeaways
               - Call to action

            2. Content Enhancement:
               - Add relevant industry statistics
               - Include expert insights
               - Provide practical examples
               - Add implementation tips
               - Include best practices
               - Address common questions/challenges

            3. SEO Optimization:
               - Use semantic keywords naturally
               - Include LSI keywords
               - Optimize header hierarchy
               - Add meta description
               - Include internal linking suggestions

            4. Engagement Elements:
               - Add thought-provoking questions
               - Include actionable tips
               - Use bullet points and lists
               - Add highlight boxes for key insights
               - Include suggested tweet quotes

            Format using proper markdown:
            - # for main title
            - ## for section headings
            - ### for subsections
            - > for quotations and highlights
            - - for bullet points
            - *** for section breaks

            Make the content comprehensive, engaging, and highly valuable to readers.
            
            Original Transcript:
            {self.transcript}
            """

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert blog writer specializing in comprehensive, long-form content."},
                    {"role": "user", "content": blog_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )

            # Get the initial blog content
            initial_content = response.choices[0].message.content

            # Generate additional sections for depth
            enhancement_prompt = f"""
            Review this blog post and enhance it with:
            1. Additional real-world examples
            2. Industry trends and statistics
            3. Expert opinions and quotes
            4. Practical implementation steps
            5. Common pitfalls and solutions
            6. Future implications

            Current Blog:
            {initial_content}
            """

            # Get enhancements
            enhancement_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content enhancer."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )

            # Combine and format the content
            enhanced_content = initial_content + "\n\n" + enhancement_response.choices[0].message.content

            self.blog_content = enhanced_content

            # Save the blog post
            safe_title = re.sub(r'[^\w\-_\. ]', '_', self.video_title)[:50]
            blog_path = os.path.join(self.output_dir, f"{safe_title}.md")
            
            # Add metadata and formatting
            final_content = f"""---
title: {self.video_title}
author: {self.video_author}
date: {time.strftime('%Y-%m-%d')}
category: Video Summary
tags: [video, transcript, blog]
---

{self.blog_content}

---
*This blog post was generated from a video by {self.video_author}*
"""

            with open(blog_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            self._save_to_cache("blog", final_content)
            
            print(f"Enhanced blog post generated and saved to: {blog_path}")
            return blog_path

        except Exception as e:
            print(f"Error generating blog post: {str(e)}")
            raise
    
    def cleanup(self):
        try:
            audio_path = os.path.join(self.temp_dir, f"{self.video_id}.webm")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print("Temporary files cleaned up.")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {str(e)}")
    
    def convert(self):
        try:
            cached_blog = self._check_cache("blog")
            if cached_blog:
                self.blog_content = cached_blog
                safe_title = re.sub(r'[^\w\-_\. ]', '_', self.video_title or f"video_{self.video_id}")[:50]
                blog_path = os.path.join(self.output_dir, f"{safe_title}.md")
                with open(blog_path, "w", encoding="utf-8") as f:
                    f.write(self.blog_content)
                print(f"Using cached blog post: {blog_path}")
                return blog_path
            
            audio_path = self.download_video_audio()
            
            self.transcribe_audio(audio_path)
            
            blog_path = self.generate_blog_post()
            
            self.cleanup()
            
            print(f"Conversion complete! Blog post saved to: {blog_path}")
            return blog_path
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            self.cleanup()
            raise

# Add blog scoring functionality from main.py
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def readability_score(text):
    try:
        return textstat.flesch_reading_ease(text)
    except Exception as e:
        print(f"Error computing readability score: {e}")
        return 0

def seo_score(text):
    words = text.split()
    word_count = len(words)
    
    if word_count < 300:
        score = 30  # Penalize short content
    elif 300 <= word_count <= 1500:
        score = 70
    else:
        score = 90  # Longer blogs tend to rank better

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = vectorizer.get_feature_names_out()
        keyword_density = len(feature_array) / word_count if word_count > 0 else 0
    except Exception as e:
        print(f"Error computing TF-IDF: {e}")
        keyword_density = 0

    if 0.01 < keyword_density < 0.05:
        score += 10
    else:
        score -= 10

    return min(max(score, 0), 100)

def call_openai_chat(prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}  # Request JSON format directly
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                return None

def parse_json_response(response_text):
    try:
        
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if code_block_match:
            json_text = code_block_match.group(1)
            return json.loads(json_text)
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON parsing error: {e}. Response was: {response_text}")
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        json_match = re.search(json_pattern, response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        return {}

def ai_detection(text):
    
    prompt = (
        "You are an expert in content analysis. Analyze the following text "
        "and determine if it is AI-generated or human-written. Provide your answer as a JSON object with two keys: "
        "'score' (an integer between 0 and 100, where 0 means entirely human and 100 means entirely AI) and "
        "'explanation' (a brief explanation of your assessment).\n\n"
        f"Text:\n{text}\n"
    )
    response_text = call_openai_chat(prompt)
    result = parse_json_response(response_text) if response_text else {}
    # Fallback: if no valid score, return a default value (e.g., 50)
    if "score" not in result:
        result["score"] = 50
        result["explanation"] = "Could not parse AI detection score."
    return result

def grammar_score(text):
    
    prompt = (
        "You are an expert in language quality assessment. Rate the grammar and clarity of the following text "
        "on a scale from 0 to 100 (where 100 indicates perfect grammar and clarity) and provide a brief explanation. "
        "Respond as a JSON object with keys 'score' and 'explanation'.\n\n"
        f"Text:\n{text}\n"
    )
    response_text = call_openai_chat(prompt)
    result = parse_json_response(response_text) if response_text else {}
    if "score" not in result:
        result["score"] = 50
        result["explanation"] = "Could not parse grammar score."
    return result

def plagiarism_check(text):
    
    return 0  
def structure_score(text):
    headings = len(re.findall(r"^#{1,6}\s.+$", text, re.MULTILINE)) 
    html_headings = len(re.findall(r"<h[1-6]>", text))  
    bullet_points = len(re.findall(r"[\*\-]\s", text)) 
    paragraphs = text.count("\n\n")  

    base_score = 50
    structure_bonus = min((headings + html_headings) * 5 + bullet_points * 3 + paragraphs * 2, 50)
    return min(base_score + structure_bonus, 100)

def analyze_keyword_density(text):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10,  
            ngram_range=(1, 2)  
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        keywords = {
            feature_names[i]: round(scores[i] * 100, 2)
            for i in scores.argsort()[-5:][::-1]  
        }
        
        return keywords
    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return {}

def calculate_engagement_score(text):
    
    prompt = (
        "Analyze this text for reader engagement potential. Consider:\n"
        "1. Use of hooks and compelling openings\n"
        "2. Story elements and narrative flow\n"
        "3. Call-to-actions and reader interaction\n"
        "4. Use of examples and illustrations\n"
        "Provide a score (0-100) and brief explanation as JSON.\n\n"
        f"Text:\n{text[:1000]}..."  
    )
    response = call_openai_chat(prompt)
    result = parse_json_response(response) if response else {}
    return result

def analyze_content_depth(text):
    
    prompt = (
        "Analyze this content for depth and expertise. Consider:\n"
        "1. Topic coverage completeness\n"
        "2. Use of technical terms and explanations\n"
        "3. Presence of supporting evidence\n"
        "4. Balance of basic and advanced concepts\n"
        "Provide a score (0-100) and explanation as JSON.\n\n"
        f"Text:\n{text[:1500]}..."  
    )
    response = call_openai_chat(prompt)
    result = parse_json_response(response) if response else {}
    return result

def analyze_headers(text):
    
    markdown_headers = re.findall(r'^#{1,6}\s.+$', text, re.MULTILINE)
    html_headers = re.findall(r'<h[1-6]>.*?</h[1-6]>', text, re.IGNORECASE)
    
    return {
        "Header Count": len(markdown_headers) + len(html_headers),
        "Has H1": bool(re.search(r'^#\s|<h1>', text, re.MULTILINE|re.IGNORECASE)),
        "Header Hierarchy": "Good" if len(markdown_headers) + len(html_headers) >= 3 else "Needs Improvement"
    }

def check_meta_description(text):
    
    first_para = text.split('\n\n')[0] if text else ""
    return {
        "Length": len(first_para),
        "Quality": "Good" if 120 <= len(first_para) <= 160 else "Needs Adjustment"
    }

def calculate_final_score(readability, seo, quality, content):

    try:
        score = 70
        
        if content["Word Count"] < 300:
            score -= 15
        elif content["Word Count"] > 1500:
            score += 10
        
        if 60 <= readability["Flesch Reading Ease"] <= 70:
            score += 10
        
        if isinstance(seo["Basic SEO Score"], (int, float)):
            score += (seo["Basic SEO Score"] - 50) * 0.2
        
        if isinstance(quality["Grammar Score"], dict) and "score" in quality["Grammar Score"]:
            score += (quality["Grammar Score"]["score"] - 50) * 0.3
        
        if isinstance(quality["AI Detection"], dict) and "score" in quality["AI Detection"]:
            if quality["AI Detection"]["score"] > 80:
                score -= 15
        
        return max(min(round(score, 2), 100), 0)  
    except Exception as e:
        print(f"Error calculating final score: {e}")
        return 50

def analyze_blog(text):

    text = clean_text(text)
    word_count = len(text.split())
    
    content_metrics = {
        "Word Count": word_count,
        "Estimated Read Time": f"{round(word_count/200, 1)} minutes",  
        "Paragraphs": len(text.split('\n\n')),
    }
    
    readability_metrics = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Grade Level": textstat.coleman_liau_index(text),
        "Text Complexity": textstat.text_standard(text),
        "Sentence Count": textstat.sentence_count(text)
    }
    
    seo_metrics = {
        "Basic SEO Score": seo_score(text),
        "Keyword Density": analyze_keyword_density(text),
        "Meta Description Length": check_meta_description(text),
        "Header Structure": analyze_headers(text)
    }
    
    quality_metrics = {
        "Grammar Score": grammar_score(text),
        "AI Detection": ai_detection(text),
        "Engagement Score": calculate_engagement_score(text),
        "Content Depth": analyze_content_depth(text)
    }
    
    final_score = calculate_final_score(
        readability_metrics,
        seo_metrics,
        quality_metrics,
        content_metrics
    )
    
    return {
        "Content Metrics": content_metrics,
        "Readability Analysis": readability_metrics,
        "SEO Analysis": seo_metrics,
        "Quality Metrics": quality_metrics,
        "Final Score": final_score
    }

def read_blog_from_file(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if file_path.lower().endswith(('.html', '.htm')):
            print("Detected HTML file, parsing content...")
            soup = BeautifulSoup(content, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            content = soup.get_text(separator=' ', strip=True)
            print(f"Extracted {len(content)} characters from HTML file")
        else:
            content = clean_text(content)
            
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Convert YouTube videos to blog posts and analyze blog quality")
    parser.add_argument("--url", help="YouTube video URL", required=False)
    parser.add_argument("--file", help="Path to blog file (.md, .html, .txt)", required=False)
    parser.add_argument("--output", default="output", help="Output directory for blog posts")
    parser.add_argument("--temp", default="temp", help="Temporary directory for downloaded files")
    parser.add_argument("--cache", default="cache", help="Cache directory for transcripts and blogs")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze the blog, don't convert")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            return
        
        print(f"Reading from file: {args.file}")
        blog_content = read_blog_from_file(args.file)
        
        if not blog_content:
            print("Error: Could not read blog content from file.")
            return
        
        print(f"Analyzing content with {len(blog_content)} characters and approximately {len(blog_content.split())} words")
        
        analysis = analyze_blog(blog_content)
        
        print("\n=== Blog Analysis Results ===")
        for key, value in analysis.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval}")
            else:
                print(f"{key}: {value}")
        
        analysis_file = os.path.join(args.output, f"{os.path.basename(args.file)}_analysis.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis saved to: {analysis_file}")
        return
    
    if args.analyze_only and not args.file:
        print("Error: --analyze-only requires --file to be specified.")
        return
    
    video_url = args.url
    if not video_url:
        video_url = input("Please enter a YouTube video URL: ")
    
    converter = YouTubeToBlog(video_url, args.output, args.temp, args.cache)
    blog_path = converter.convert()
    
    if not args.analyze_only:
        print("\nAnalyzing generated blog...")
        blog_content = read_blog_from_file(blog_path)
        analysis = analyze_blog(blog_content)
        
        print("\n=== Blog Analysis Results ===")
        for key, value in analysis.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval}")
            else:
                print(f"{key}: {value}")
        
        analysis_file = os.path.join(args.output, f"{os.path.basename(blog_path)}_analysis.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()




