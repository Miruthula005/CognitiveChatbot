from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for
import json
import random
import io
import time
import logging
import warnings
import base64
import requests
from youtube_search import YoutubeSearch
from gtts import gTTS
from groq import Groq
from pathlib import Path
from PIL import Image
import re
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from TTS.api import TTS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter("ignore", FutureWarning)

# -------------------------------
# Load Questionnaire
# -------------------------------
try:
    with open("data/questionnaire.json") as f:
        data = json.load(f)
except FileNotFoundError:
    data = {
        "Scale": {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5},
        "Visual": [],
        "Auditory": [],
        "Reading/Writing": [],
        "Kinesthetic": []
    }

scale = data["Scale"]
user_scores = {"Visual": 0, "Auditory": 0, "Reading/Writing": 0, "Kinesthetic": 0}
question_set = []
current_index = 0

# -------------------------------
# Initialize Groq Client
# -------------------------------
# get key from environment (set via .env or system env vars)
GROQ_API_KEY = os.getenv("API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ API key not found. Please set API_KEY in .env or environment.")


GROQ_MODEL = "llama-3.3-70b-versatile"

try:
    client = Groq(api_key=GROQ_API_KEY)
    # Test connection
    test_response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hi"}],
        model=GROQ_MODEL,
        max_tokens=10
    )
    print("✓ Groq API connected successfully!")
except Exception as e:
    print(f"❌ Error connecting to Groq: {e}")
    client = None


# Initialize embeddings and load vector store
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore_path = r"C:\Users\braga\Documents\College\Project\Cognitive_Chatbot\ml_book_vectorstore"
    
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # Add this parameter
        )
        print("✓ Vector store loaded successfully!")
    else:
        print("❌ Vector store path not found!")
        vectorstore = None
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    vectorstore = None

def get_relevant_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from vector store"""
    try:
        if vectorstore is None:
            return ""
            
        # Get relevant documents
        docs = vectorstore.similarity_search(query, k=k)
        
        # Combine context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def generate_llm_response(prompt: str, max_new_tokens: int = 250) -> str:
    """Generate response using Groq API with RAG"""
    if client is None:
        return "⚠️ Groq API not connected."

    try:
        # Get relevant context
        context = get_relevant_context(prompt)
        
        # Create augmented prompt
        augmented_prompt = f"""Use the following ML textbook excerpts as context for your response:

{context}

Based on this context and your knowledge, {prompt}

Respond in a clear, structured way covering:
1. Definition/Explanation
2. Key Points
3. Example or Application
"""

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert ML tutor. Use the provided context to give accurate, textbook-based answers."},
                {"role": "user", "content": augmented_prompt}
            ],
            model=GROQ_MODEL,
            max_tokens=max_new_tokens,
            temperature=0.7
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        text = f"❌ Error: {e}"
    
    # Ensure all sections exist
    for sec in ["Definition:", "Example:", "Advantages:", "Disadvantages:", "Next topic to study"]:
        if sec not in text:
            text += f"\n{sec} N/A"
    return text

def select_questions():
    """Select random questions for each learning style."""
    global question_set, current_index, user_scores
    question_set = []
    user_scores = {k: 0 for k in user_scores}
    current_index = 0
    for style in ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]:
        if data.get(style):
            selected = random.sample(data[style], min(2, len(data[style])))
            for q in selected:
                question_set.append({"question": q["question"], "style": style})
    random.shuffle(question_set)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/next_question")
def next_question():
    current_index = int(request.args.get('index', 0))
    if current_index < len(QUESTIONS):
        return jsonify({
            "question": QUESTIONS[current_index],
            "index": current_index + 1,
            "total": len(QUESTIONS)
        })
    else:
        return jsonify({"done": True})

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.get_json()
    if not hasattr(submit_answer, 'responses'):
        submit_answer.responses = []
    
    answer = data.get("answer")
    submit_answer.responses.append(SCALE[answer])
    
    print(f"Question {len(submit_answer.responses)}: {answer} -> {SCALE[answer]}")
    
    if len(submit_answer.responses) == len(QUESTIONS):
        try:
            features = np.array(submit_answer.responses).reshape(1, -1)
            features_scaled = svm_scaler.transform(features)
            prediction = svm_model.predict(features_scaled)[0]
            probabilities = svm_model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction] * 100  # Convert to percentage
            
            style_map = {0: "Auditory", 1: "Reading/Writing", 2: "Visual"}
            predicted_style = style_map.get(prediction)
            
            if predicted_style is None:
                print(f"Warning: Invalid prediction value: {prediction}")
                predicted_style = "Reading/Writing"  # Default fallback
            
            print(f"Prediction value: {prediction}")
            print(f"Mapped style: {predicted_style}")
            print(f"Confidence: {confidence:.2f}%")
            
            submit_answer.responses = []  # Reset for next user
            return jsonify({"done": True, "style": predicted_style, "confidence": round(confidence, 2)})
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            submit_answer.responses = []  # Reset on error
            return jsonify({"error": "Prediction failed"}), 500
    
    return jsonify({"status": "ok"})

@app.route("/after_result")
def after_result():
    style = request.args.get('style')
    confidence = request.args.get('confidence')
    print(f"Rendering after_result with style: {style}, confidence: {confidence}")  # Debug log
    return render_template("after_result.html", style=style, confidence=confidence)

@app.route("/chatbot")
def chatbot_page():
    style = request.args.get("style", "Unknown")
    return render_template("chatbot.html", style=style)

@app.route("/select_visual_mode")
def select_visual_mode():
    return redirect(url_for("chatbot_page", style="Visual"))

@app.route("/get_video", methods=["POST"])
def get_video():
    data_req = request.get_json()
    query = data_req.get("query", "")
    if not query:
        return jsonify({"error": "No query provided."}), 400
    
    try:
        # Enhance query with ML context for better relevance
        enhanced_query = f"{query} machine learning tutorial explanation"
        
        # Get multiple results to filter best one
        results = YoutubeSearch(enhanced_query, max_results=5).to_dict()
        
        if results:
            # Filter and rank results by relevance
            relevant_videos = []
            ml_keywords = ['machine learning', 'ml', 'data science', 'algorithm', 
                          'neural network', 'deep learning', 'ai', 'artificial intelligence',
                          'tutorial', 'explained', 'introduction']
            
            for video in results:
                title = video.get("title", "").lower()
                channel = video.get("channel", "").lower()
                
                # Calculate relevance score
                relevance_score = 0
                for keyword in ml_keywords:
                    if keyword in title or keyword in channel:
                        relevance_score += 1
                
                # Prefer educational channels and tutorials
                if any(word in title for word in ['tutorial', 'explained', 'introduction', 'guide']):
                    relevance_score += 2
                if any(word in channel for word in ['academy', 'education', 'learning', 'tech']):
                    relevance_score += 1
                
                # Avoid very short videos (likely not educational)
                duration = video.get("duration", "")
                if duration and len(duration) > 4:  # at least 5+ minutes
                    relevance_score += 1
                
                video['relevance_score'] = relevance_score
                relevant_videos.append(video)
            
            # Sort by relevance score
            relevant_videos.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Get the most relevant video
            if relevant_videos:
                best_video = relevant_videos[0]
                video_id = best_video.get("id") or best_video.get("url_suffix", "").split("v=")[-1]
                title = best_video.get("title", "Untitled Video")
                channel = best_video.get("channel", "Unknown Channel")
                duration = best_video.get("duration", "")
                views = best_video.get("views", "")
                
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                return jsonify({
                    "title": title,
                    "url": video_url,
                    "channel": channel,
                    "duration": duration,
                    "views": views
                })
            else:
                return jsonify({"error": "No relevant videos found."})
        else:
            return jsonify({"error": "No results found."})
            
    except Exception as e:
        print(f"Error fetching video: {e}")
        return jsonify({"error": str(e)})

@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    data_req = request.get_json()
    text = data_req.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return send_file(mp3_fp, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500

# Initialize Coqui TTS
try:
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    print("✓ Coqui TTS model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Coqui TTS model: {e}")
    tts = None

def generate_speech(text: str) -> str:
    """Generate speech using Coqui TTS"""
    try:
        output_path = "static/speech.wav"
        # Clean and prepare text for TTS
        cleaned_text = text.replace('\n', ' ').strip()
        # Limit text length to avoid tensor size issues
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500] + "..."
            
        tts.tts_to_file(text=cleaned_text, file_path=output_path)
        return url_for('static', filename='speech.wav')
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def handle_visual_response(query: str, visual_type: str) -> dict:
    """Handle visual learning mode responses"""
    try:
        if visual_type == "video":
            video_data = search_youtube_video(query)
            if video_data:
                return {
                    "type": "video",
                    "video_data": video_data,
                    "success": True
                }
            return {"error": "No relevant videos found", "success": False}
        else:
            # Image mode
            mermaid_code = generate_mermaid_code(query)
            explanation = generate_concept_explanation(query)
            
            if not mermaid_code:
                return {"error": "Failed to generate diagram"}, 500
                
            mermaid_bytes = mermaid_code.encode('utf-8')
            base64_code = base64.urlsafe_b64encode(mermaid_bytes).decode('utf-8')
            
            image_url = f"https://mermaid.ink/img/{base64_code}"
            response = requests.get(image_url)
            
            if response.status_code == 200:
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                return {
                    "image": f"data:image/png;base64,{image_base64}",
                    "code": mermaid_code,
                    "explanation": explanation or "No explanation available."
                }
    except Exception as e:
        print(f"Error in visual response: {e}")
        return {"error": str(e)}, 500

def handle_auditory_response(query: str) -> dict:
    """Handle auditory learning mode responses"""
    try:
        prompt = """You are an expert ML teacher. Explain the concept in exactly 2-4 sentences using these rules:
1. Use simple, conversational language
2. Include one brief analogy
3. Focus on the most important point
4. Avoid technical jargon

Question: {query}"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            model=GROQ_MODEL,
            max_tokens=150,
            temperature=0.3
        ).choices[0].message.content.strip()
        
        audio_path = generate_speech(response)
        
        return {
            "response": response,
            "audio": audio_path,
            "success": True if audio_path else False
        }
    except Exception as e:
        print(f"Error in auditory response: {e}")
        return {"error": str(e)}, 500

def handle_reading_response(query: str) -> dict:
    """Handle reading/writing learning mode responses"""
    try:
        prompt = """You are an expert ML teacher. Explain the following concept in a clear, structured way:

Definition:
- What is it?
- Key purpose

Advantages:
- Key benefits
- Best use cases

Disadvantages:
- Limitations
- Challenges

Question: {0}"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ML expert. Use clear explanations with examples."},
                {"role": "user", "content": prompt.format(query)}
            ],
            model=GROQ_MODEL,
            max_tokens=500,
            temperature=0.3
        ).choices[0].message.content.strip()
        
        return {"response": response}
            
    except Exception as e:
        print(f"Error in reading response: {e}")
        return {"error": str(e)}, 500

def handle_kinesthetic_response(query: str) -> dict:
    """Handle kinesthetic learning mode responses with practical examples and resources"""
    try:
        # Generate practical, hands-on explanation
        prompt = """You are an expert ML teacher for kinesthetic learners. Explain the concept in a practical, hands-on way:

1. Start with a real-world analogy or scenario (2-3 sentences)
2. Describe a practical example or experiment they can try (2-3 sentences)
3. Explain the key takeaway in simple terms (1-2 sentences)

Keep it conversational and action-oriented. Focus on "doing" and "experiencing" rather than theory.

Question: {query}"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an ML expert focused on practical, hands-on learning."},
                {"role": "user", "content": prompt.format(query=query)}
            ],
            model=GROQ_MODEL,
            max_tokens=300,
            temperature=0.5
        ).choices[0].message.content.strip()
        
        # Search for relevant educational resources
        resource = search_learning_resource(query)
        
        return {
            "response": response,
            "resource": resource,
            "success": True
        }
    except Exception as e:
        print(f"Error in kinesthetic response: {e}")
        return {"error": str(e)}, 500


def search_learning_resource(query: str) -> dict:
    """Search for educational resources, prioritizing GeeksforGeeks"""
    try:
        # GeeksforGeeks base URL structure
        gfg_base_url = "https://www.geeksforgeeks.org/"
        
        # Common ML topics mapping to GeeksforGeeks URLs
        gfg_topic_map = {
            'supervised learning': 'supervised-unsupervised-learning',
            'unsupervised learning': 'supervised-unsupervised-learning',
            'neural network': 'introduction-to-artificial-neutral-networks',
            'cnn': 'introduction-convolution-neural-network',
            'convolutional neural network': 'introduction-convolution-neural-network',
            'rnn': 'introduction-to-recurrent-neural-network',
            'recurrent neural network': 'introduction-to-recurrent-neural-network',
            'decision tree': 'decision-tree-introduction-example',
            'random forest': 'random-forest-algorithm-in-machine-learning',
            'svm': 'support-vector-machine-algorithm',
            'support vector machine': 'support-vector-machine-algorithm',
            'k means': 'k-means-clustering-introduction',
            'kmeans': 'k-means-clustering-introduction',
            'linear regression': 'ml-linear-regression',
            'logistic regression': 'understanding-logistic-regression',
            'knn': 'k-nearest-neighbours',
            'k nearest neighbor': 'k-nearest-neighbours',
            'naive bayes': 'naive-bayes-classifiers',
            'gradient descent': 'gradient-descent-algorithm-and-its-variants',
            'backpropagation': 'backpropagation-in-neural-network',
            'overfitting': 'underfitting-and-overfitting-in-machine-learning',
            'underfitting': 'underfitting-and-overfitting-in-machine-learning',
            'regularization': 'regularization-in-machine-learning',
            'cross validation': 'cross-validation-machine-learning',
            'pca': 'principal-component-analysis-pca',
            'principal component analysis': 'principal-component-analysis-pca',
            'dimensionality reduction': 'dimensionality-reduction',
            'ensemble learning': 'ensemble-methods-in-machine-learning',
            'boosting': 'boosting-in-machine-learning-boosting-and-adaboost',
            'bagging': 'bagging-in-machine-learning',
            'lstm': 'deep-learning-introduction-to-long-short-term-memory',
            'gru': 'gated-recurrent-unit-networks',
            'autoencoder': 'auto-encoders',
            'gan': 'generative-adversarial-network-gan',
            'generative adversarial network': 'generative-adversarial-network-gan',
            'transformer': 'transformer-neural-network',
            'attention mechanism': 'attention-mechanism',
            'reinforcement learning': 'what-is-reinforcement-learning',
            'q learning': 'q-learning-in-python',
            'deep learning': 'introduction-deep-learning',
            'machine learning': 'machine-learning',
            'activation function': 'activation-functions-neural-networks',
            'loss function': 'loss-functions-in-machine-learning',
            'optimizer': 'optimization-techniques-for-gradient-descent',
            'batch normalization': 'batch-normalization-ml',
            'dropout': 'dropout-in-neural-networks',
            'transfer learning': 'ml-introduction-to-transfer-learning',
            'data preprocessing': 'data-preprocessing-machine-learning-python',
            'feature engineering': 'feature-engineering',
            'feature selection': 'feature-selection-techniques-in-machine-learning',
            'confusion matrix': 'confusion-matrix-machine-learning',
            'precision recall': 'precision-and-recall-in-machine-learning',
            'f1 score': 'f1-score-in-machine-learning',
            'roc curve': 'roc-curve-in-machine-learning',
            'bias variance': 'bias-variance-tradeoff-machine-learning',
        }
        
        # Normalize query for matching
        query_lower = query.lower().strip();
        
        # Try to find direct match in mapping
        gfg_article = None
        for key, article in gfg_topic_map.items():
            if key in query_lower or query_lower in key:
                gfg_article = article
                break
        
        # If direct match found, return GeeksforGeeks URL
        if gfg_article:
            return {
                "title": f"GeeksforGeeks: {query}",
                "url": f"{gfg_base_url}{gfg_article}/",
                "description": f"Comprehensive tutorial and hands-on examples for {query} on GeeksforGeeks"
            }
        
        # If no direct match, use Groq to generate GeeksforGeeks-focused recommendation
        resource_prompt = f"""Given the topic "{query}" in machine learning, create a GeeksforGeeks URL.

GeeksforGeeks URLs follow this pattern: https://www.geeksforgeeks.org/[topic-name-with-hyphens]/

Convert the topic to a likely GeeksforGeeks article URL. Use hyphens between words, keep it lowercase, and add relevant ML context.

Examples:
- "neural networks" → https://www.geeksforgeeks.org/introduction-to-artificial-neutral-networks/
- "gradient descent" → https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/
- "decision trees" → https://www.geeksforgeeks.org/decision-tree-introduction-example/

Provide ONLY:
TITLE: GeeksforGeeks: [topic name]
URL: https://www.geeksforgeeks.org/[article-slug]/
DESCRIPTION: [One sentence about what they'll learn]"""

        resource_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at creating GeeksforGeeks article URLs for ML topics. Always return valid GeeksforGeeks URLs."},
                {"role": "user", "content": resource_prompt}
            ],
            model=GROQ_MODEL,
            max_tokens=150,
            temperature=0.2
        ).choices[0].message.content.strip()
        
        # Parse the response
        lines = resource_response.split('\n')
        title = url = description = ""
        
        for line in lines:
            if line.startswith('TITLE:'):
                title = line.replace('TITLE:', '').strip()
            elif line.startswith('URL:'):
                url = line.replace('URL:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
        
        # Validate it's a GeeksforGeeks URL
        if url and 'geeksforgeeks.org' in url.lower():
            return {
                "title": title or f"GeeksforGeeks: {query}",
                "url": url,
                "description": description or f"Learn about {query} with practical examples and explanations"
            }
        else:
            # Fallback: construct a GeeksforGeeks search URL
            search_query = query.replace(' ', '+')
            return {
                "title": f"GeeksforGeeks: {query}",
                "url": f"https://www.geeksforgeeks.org/?s={search_query}",
                "description": f"Search GeeksforGeeks for tutorials on {query}"
            }
            
    except Exception as e:
        print(f"Error searching for resources: {e}")
        # Always fallback to GeeksforGeeks search
        search_query = query.replace(' ', '+')
        return {
            "title": f"GeeksforGeeks: {query}",
            "url": f"https://www.geeksforgeeks.org/?s={search_query}",
            "description": f"Search GeeksforGeeks for tutorials on {query}"
        }

# Update the llm_response route to use these functions
@app.route("/llm_response", methods=["POST"])
def llm_response():
    data_req = request.get_json()
    query = data_req.get("query", "").strip()
    style = data_req.get("style", "default").strip()
    visual_type = data_req.get("visual_type", "image").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        if style.lower() == "visual":
            result = handle_visual_response(query, visual_type)
        elif style.lower() == "auditory":
            result = handle_auditory_response(query)
        elif style.lower() == "reading/writing":
            result = handle_reading_response(query)
        elif style.lower() == "kinesthetic":
            result = handle_kinesthetic_response(query)
        else:
            return jsonify({"error": "Invalid learning style"}), 400

        # Check if result contains an error status code
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        print(f"Error in llm_response: {e}")
        return jsonify({"error": str(e)}), 500

# Remove these configurations
VISUAL_MODE_ENABLED = True
VISUAL_MODE_CONFIG = {
    'diagram_types': {...},
    'keywords': {...}
}

# Keep only the essential image generation code
def generate_mermaid_code(prompt: str) -> str:
    """Generate Mermaid diagram code using Groq"""
    system_msg = """You are an expert at creating Mermaid mindmap diagrams.
Follow this exact format:

mindmap
    root((Main Topic))
        key1
            subkey1
            subkey2
        key2
            subkey3
            subkey4

Rules:
- Start with 'mindmap' 
- If necessary u could use more than 2-3 nodes in the mindmap
- Use 4 spaces for indentation
- Use double parentheses only for root node
- Keep node text simple and short
- No special characters except ()
- No empty lines between nodes"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Create a mindmap diagram explaining this ML concept: {prompt}"}
            ],
            model=GROQ_MODEL,
            max_tokens=400,
            temperature=0.2  # Lower temperature for more consistent output
        )
        
        # Clean and format the generated code
        generated_code = completion.choices[0].message.content.strip()
        
        # Ensure it starts with 'mindmap'
        if not generated_code.startswith('mindmap'):
            generated_code = 'mindmap\n' + generated_code
            
        # Clean up the code
        lines = generated_code.split('\n')
        formatted_lines = []
        for line in lines:
            # Preserve indentation
            indent = len(line) - len(line.lstrip())
            # Clean content (allow only alphanumeric and parentheses)
            content = ''.join(c for c in line.strip() if c.isalnum() or c in ' ()')
            if content:
                formatted_lines.append(' ' * indent + content)
        
        return '\n'.join(formatted_lines)
        
    except Exception as e:
        print(f"Error generating diagram: {e}")
        # Return a simple valid fallback diagram
        return """mindmap
    root((Topic))
        concept1
            detail1
            detail2
        concept2
            detail3
            detail4"""

def generate_concept_explanation(prompt: str) -> str:
    """Generate a brief explanation of the concept using Groq"""
    system_msg = """You are an ML expert. Provide a brief, clear explanation of the concept in 2-3 sentences.
Focus on the core idea and its importance. Keep it simple and concise."""

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Explain this ML concept briefly: {prompt}"}
            ],
            model=GROQ_MODEL,
            max_tokens=150,  # Keep explanation concise
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return None

@app.route("/generate_image", methods=["POST"])
def get_generated_image():
    data_req = request.get_json()
    prompt = data_req.get("prompt", "").strip()
    
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400
        
    try:
        # Generate both mindmap and explanation
        mermaid_code = generate_mermaid_code(prompt)
        explanation = generate_concept_explanation(prompt)
        
        if not mermaid_code:
            return jsonify({"error": "Failed to generate diagram code"}), 500
            
        # Convert to base64 for mermaid.ink
        mermaid_bytes = mermaid_code.encode('utf-8')
        base64_code = base64.urlsafe_b64encode(mermaid_bytes).decode('utf-8')
        
        # Get image from mermaid.ink
        image_url = f"https://mermaid.ink/img/{base64_code}"
        response = requests.get(image_url)
        
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return jsonify({
                "image": f"data:image/png;base64,{image_base64}",
                "code": mermaid_code,
                "explanation": explanation or "No explanation available."
            })
        else:
            return jsonify({"error": "Failed to render diagram"}), 500
            
    except Exception as e:
        print(f"Error in image generation: {e}")
        return jsonify({"error": str(e)}, 500)

def extract_terms(mermaid_code: str) -> list:
    """Extract all terms from the Mermaid mindmap code"""
    terms = []
    lines = mermaid_code.split('\n')
    for line in lines:
        # Skip the mindmap keyword line
        if line.strip() == 'mindmap':
            continue
        # Extract term (remove special characters and indentation)
        term = line.strip().replace('root((', '').replace('))', '').strip()
        if term:
            terms.append(term)
    return terms

def generate_term_explanations(terms: list) -> str:
    """Generate explanations for the terms using Groq"""
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": """You are an ML expert. Format each term explanation as:
Term: Explanation (1-2 sentences)
Keep explanations clear and concise."""},
                {"role": "user", "content": f"Explain these ML terms briefly (one by one): {', '.join(terms)}"}
            ],
            model=GROQ_MODEL,
            max_tokens=500,
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating explanations: {e}")
        return "Failed to generate explanations."

# Visual Mode Configuration
VISUAL_MODE_ENABLED = True
MERMAID_API_URL = "https://mermaid.ink/img/"
DEFAULT_IMAGE_WIDTH = 800  # pixels

# Diagram type keywords for auto-detection
DIAGRAM_KEYWORDS = {
    'flowchart': ['how does', 'algorithm', 'process', 'step', 'procedure', 'workflow'],
    'mindmap': ['what is', 'explain', 'concept', 'define', 'describe', 'overview'],
    'comparison': ['difference', 'compare', 'versus', 'vs', 'contrast', 'distinguish'],
    'sequence': ['sequence', 'interaction', 'communication', 'flow between'],
}

# Load the SVM model and scaler
try:
    model_path = r"C:\Users\braga\Documents\College\Project\Cognitive_Chatbot\saved_model\Learning Style\svm_LS_model.pkl"
    scaler_path = r"C:\Users\braga\Documents\College\Project\Cognitive_Chatbot\saved_model\Learning Style\svm_scaler.pkl"
    
    svm_model = joblib.load(model_path)
    svm_scaler = joblib.load(scaler_path)
    print("✓ SVM model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading SVM model: {e}")
    svm_model = None
    svm_scaler = None

# Update QUESTIONS list to match dataset exactly
QUESTIONS = [
    "I learn better by reading what the teacher writes on the board.",  # Visual 1
    "I learn better by reading instructions than by listening to instructions.",  # Visual 2
    "I understand better when I read instructions.",  # Visual 3
    "I learn better by reading than by listening to someone.",  # Visual 4
    "I learn more by reading textbooks than by listening to lectures.",  # Visual 5
    "When the teacher tells me the instructions, I understand better.",  # Auditory 1
    "I learn better in class when listening to the teacher than reading the textbook.",  # Auditory 2
    "I understand things better in class when the teacher gives a lecture.",  # Auditory 3
    "I learn better in class when I listen to someone.",  # Auditory 4
    "I remember things I have heard in class better than things I have read.",  # Auditory 5
    "I prefer to learn by doing something in class.",  # Kinesthetic 1
    "When I do things in class, I learn better.",  # Kinesthetic 2
    "I enjoy learning in class by doing experiments.",  # Kinesthetic 3
    "I understand things better in class when I participate in role-playing.",  # Kinesthetic 4
    "I learn best in class when I can participate in related activities."  # Kinesthetic 5
]

# Keep the existing SCALE dictionary
SCALE = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

DIAGRAM_TEMPLATES = {
    'supervised': """mindmap
  root((Supervised Learning))
    Training
      Data Preparation
      Model Training
    Types
      Classification
      Regression""",
    
    'neural': """mindmap
  root((Neural Networks))
    Architecture
      Input Layer
      Hidden Layers
    Training
      Forward Pass
      Backpropagation""",
    
    'cnn': """mindmap
  root((CNN))
    Layers
      Convolutional
      Pooling
    Features
      Patterns
      Filters"""
}

def validate_mermaid_code(code: str) -> bool:
    """Validate Mermaid code structure"""
    lines = code.split('\n')
    if not lines[0].strip() == 'mindmap':
        return False
    
    # Check each line has proper indentation and content
    for line in lines[1:]:
        if line.strip():  # Skip empty lines
            # Check indentation is multiple of 2
            spaces = len(line) - len(line.lstrip())
            if spaces % 2 != 0:
                return False
            # Check content is valid
            content = line.strip()
            if not all(c.isalnum() or c in ' ()_-' for c in content):
                return False
    return True

# Add this after loading the model to test predictions
def test_predictions():
    # Test case for each learning style
    test_cases = {
        "Visual": [5,5,5,5,5, 1,1,1,1,1, 1,1,1,1,1],  # High visual scores
        "Auditory": [1,1,1,1,1, 5,5,5,5,5, 1,1,1,1,1],  # High auditory scores
        "Reading/Writing": [1,1,1,1,1, 1,1,1,1,1, 5,5,5,5,5]  # High kinesthetic scores
    }
    
    for style, scores in test_cases.items():
        features = np.array(scores).reshape(1, -1)
        features_scaled = svm_scaler.transform(features)
        prediction = svm_model.predict(features_scaled)[0]
        print(f"Test case {style}: Predicted {prediction}")

# Call this after model loading
test_predictions()

if __name__ == "__main__":
    app.run(debug=True)
