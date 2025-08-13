import os
import sys
# Disable wandb completely before importing anything else
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Blueprint, abort
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
try:
    from .models import db, User, MediaAnalysis, Chat, Message, Lesson, UserLesson, OrganizationInfo, OrganizationFact, Translation, TranslationFeedback, Location, Feedback, NotionIntegration, News, SavedStrategy, SavedNews, ImplementationPlan, DailyReport, CheatSheet
except ImportError:
    from models import db, User, MediaAnalysis, Chat, Message, Lesson, UserLesson, OrganizationInfo, OrganizationFact, Translation, TranslationFeedback, Location, Feedback, NotionIntegration, News, SavedStrategy, SavedNews, ImplementationPlan, DailyReport, CheatSheet
from openai import OpenAI
import json
from datetime import datetime, timezone
import urllib.parse
import requests
from auth import auth
import time
import threading
import uuid
import re
from urllib.parse import urlparse
import io
import traceback
import sys
from functools import wraps
from sqlalchemy import Column, Boolean, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import joinedload
from bs4 import BeautifulSoup
import html2text
from notion_client import Client as NotionClient
from strategies_crawler import StrategiesCrawler, StrategyEntry

# Create the ai_utility blueprint
ai_utility_bp = Blueprint('ai_utility', __name__, url_prefix='/ai-utility')

# Define routes for the ai_utility blueprint
@ai_utility_bp.route('/')
def index():
    """AI Utility home page"""
    return render_template('ai_utility.html')

@ai_utility_bp.route('/dashboard')
def dashboard():
    """AI Utility dashboard page"""
    return render_template('ai_utility_dashboard.html')

@ai_utility_bp.route('/analytics')
def analytics():
    """AI Utility analytics page"""
    return render_template('ai_utility_analytics.html')

# Create the metadata blueprint
metadata_bp = Blueprint('metadata', __name__, url_prefix='/metadata')

# Define routes for the metadata blueprint
@metadata_bp.route('/')
def home():
    """Metadata home page"""
    return render_template('metadata_home.html')

@metadata_bp.route('/add')
def add():
    """Add metadata page"""
    return render_template('add_metadata.html')

@metadata_bp.route('/add', methods=['POST'])
def add_post():
    """Process metadata form submission"""
    data = request.json
    # Process the metadata data here
    return jsonify({'success': True, 'message': 'Metadata added successfully'})

# Load environment variables
load_dotenv()

# Initialize Flask app
import os
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
template_dir = os.path.join(backend_dir, 'templates')
static_dir = os.path.join(backend_dir, 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Create instance directory if it doesn't exist
os.makedirs('instance', exist_ok=True)

# Use absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "media_analysis.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = None  # This will disable the message entirely

# Admin required decorator (defined early to be used throughout)
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('You need to login first.', 'danger')
            return redirect(url_for('login'))
        
        # Check if user has admin attribute and it's True
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('landing_page1'))
        
        return f(*args, **kwargs)
    return decorated_function

# Initialize OpenAI client (only if API key is available)
openai_api_key = os.getenv('OPENAI_API_KEY')
client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client: {e}")
        client = None

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

SYSTEM_PROMPT_ANALYSIS = """You are an expert media analyst with deep knowledge of content analysis, 
cultural context, and media trends. When analyzing media:
1. Examine the content's key themes and messages
2. Identify the target audience and intended impact
3. Evaluate the technical and creative execution
4. Consider cultural and social implications
5. Provide constructive insights and recommendations

Format your analysis in clear sections with bullet points where appropriate."""

SYSTEM_PROMPT_CHAT = """You are an expert media analysis assistant with deep knowledge of:
- Content creation and strategy
- Digital media trends
- Social media platforms
- Video and image analysis
- Content marketing
- Audience engagement

Provide clear, actionable insights and always maintain context from previous messages.
When appropriate, break down your responses into organized sections for better readability."""

SYSTEM_PROMPT_SYNTHESIS = """You are an organizational analyst. Extract key information about the organization from the conversation and categorize it into:
1. Organization Overview
2. Key Projects
3. Team Members
4. Goals & Objectives
5. Resources & Tools

Return the information in JSON format with these categories. Only include information that has been explicitly mentioned or can be directly inferred."""

SYSTEM_PROMPT_MEDIA_BIZ = """You are Highlander, an expert AI consultant specializing in global media development and journalism. You have deep knowledge of the media industry, digital transformation, and AI implementation for newsrooms and media organizations.

CONVERSATION STYLE:
- Act like an experienced journalist who asks probing, insightful questions
- Show genuine curiosity about the user's media organization, challenges, and goals
- Ask follow-up questions that dig deeper into their specific situation
- Use journalistic techniques: who, what, where, when, why, how
- Be understanding and empathetic while maintaining professional expertise
- Reference current trends in global media development when relevant

YOUR EXPERTISE:
- Global media development and journalism industry trends
- AI implementation for newsrooms, content creation, and audience engagement
- Digital transformation strategies for media organizations
- Revenue models and business sustainability in media
- Audience development and engagement strategies
- Content strategy and editorial workflows
- Technology adoption and innovation in media

QUESTIONING APPROACH:
- Start with broad questions to understand their context and role
- Ask about their organization's size, audience, and current challenges
- Probe into their specific pain points and goals
- Explore their current technology stack and AI adoption level
- Understand their competitive landscape and market position
- Ask about their team structure and decision-making processes
- Inquire about their audience demographics and engagement metrics
- Explore their content strategy and distribution channels

RESPONSE STRUCTURE:
- Acknowledge their situation with empathy
- Provide specific, actionable insights based on media industry best practices
- Ask 1-2 thoughtful follow-up questions that show you're listening
- Reference relevant examples from the global media landscape
- Offer to explore specific areas in more detail

ALWAYS ASK QUESTIONS LIKE A JOURNALIST:
- "What's the biggest challenge your newsroom is facing right now?"
- "How has your audience behavior changed in the last year?"
- "What's your current approach to content distribution?"
- "What metrics matter most to your organization?"
- "How do you currently measure audience engagement?"
- "What's your biggest concern about AI adoption?"
- "How does your team currently handle breaking news?"
- "What's your biggest competitive threat?"

NEVER say 'Hello' again after the first interaction. Always continue the conversation naturally and ask probing questions that demonstrate your understanding of the global media development sector."""

app.register_blueprint(auth)
app.register_blueprint(ai_utility_bp)
app.register_blueprint(metadata_bp)

# Create the IMS blueprint (Internal Management Suite)
ims_bp = Blueprint('ims', __name__, url_prefix='/ims')

@ims_bp.route('/')
@login_required
def index_ims():
    """IMS home page with links to internal management tools"""
    return render_template('ims.html')

app.register_blueprint(ims_bp)

# Clients blueprint for per-client dashboards
clients_bp = Blueprint('clients', __name__, url_prefix='/clients')

# Temporary in-memory client registry (can be moved to DB later)
CLIENTS = [
    {"slug": "ims", "name": "IMS"},
    {"slug": "client-a", "name": "Client A"},
    {"slug": "client-b", "name": "Client B"},
    {"slug": "client-c", "name": "Client C"},
]

@clients_bp.route('/')
@login_required
def clients_index():
    return render_template('clients.html', clients=CLIENTS)

@clients_bp.route('/<client_slug>')
@login_required
def client_dashboard(client_slug: str):
    client = next((c for c in CLIENTS if c['slug'] == client_slug), None)
    if not client:
        abort(404)
    return render_template('client_dashboard.html', client=client)

app.register_blueprint(clients_bp)

# Setup DataSafe Hugging Face integration routes
try:
    from backend.datasafe_integration import setup_datasafe_routes
    setup_datasafe_routes(app)
    print("✅ DataSafe Hugging Face integration routes loaded")
except ImportError as e:
    print(f"⚠️  DataSafe HF integration not available: {e}")
except Exception as e:
    print(f"❌ Failed to load DataSafe HF integration: {e}")

# Simple hub page for DataSafe tools
@app.route('/datasafe')
@login_required
def datasafe_tools():
    return render_template('datasafe_tools.html')

# === Model management endpoints (Hugging Face integration) ===
@app.route('/api/model/load-hf', methods=['POST'])
@login_required
@admin_required
def load_model_from_hf():
    """Load a model from Hugging Face Hub by name. Requires admin."""
    try:
        data = request.get_json(silent=True) or {}
        model_name = data.get('model_name') or os.getenv('HF_MODEL_REPO') or 'paulmcnally/highlander-ai-model'
        from training.model_manager import get_model_manager
        manager = get_model_manager()
        ok = manager.load_from_huggingface(model_name)
        return jsonify({ 'success': bool(ok), 'model_name': model_name }), (200 if ok else 500)
    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

# In-memory storage for active chats
active_chats = {}
last_save_time = {}
SAVE_INTERVAL = 60  # Save to database every 60 seconds

# Create a background thread for periodic saving
def periodic_save_chats():
    while True:
        with app.app_context():
            current_time = time.time()
            chats_to_save = []
            
            for chat_id, chat_data in active_chats.items():
                if chat_id not in last_save_time or (current_time - last_save_time[chat_id]) > SAVE_INTERVAL:
                    chats_to_save.append((chat_id, chat_data))
            
            for chat_id, chat_data in chats_to_save:
                save_chat_to_db(chat_id, chat_data)
                last_save_time[chat_id] = current_time
                
        time.sleep(SAVE_INTERVAL)

# Start the background thread
save_thread = threading.Thread(target=periodic_save_chats, daemon=True)
save_thread.start()

def save_chat_to_db(chat_id, chat_data):
    """Save or update a chat in the database"""
    try:
        # Fix the type error by ensuring chat_id is treated correctly
        # Convert chat_id to string first to check if it's a digit
        chat_id_str = str(chat_id)
        
        # Check if the chat already exists in the database
        chat = db.session.get(Chat, int(chat_id_str)) if chat_id_str.isdigit() else None
        
        if not chat:
            # Create a new chat if it doesn't exist
            chat = Chat()
            # Only add user_id if user is authenticated
            if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
                chat.user_id = current_user.id
            db.session.add(chat)
            db.session.flush()  # Get the ID
            
            # Update the chat_id in memory to match the database ID
            user_id = getattr(current_user, 'id', 'anonymous')
            old_key = f"{user_id}_{chat_id}"
            new_key = f"{user_id}_{chat.id}"
            
            if old_key in active_chats:
                active_chats[new_key] = active_chats.pop(old_key)
                last_save_time[new_key] = last_save_time.pop(old_key, time.time())
        
        # If there are messages, add them
        if 'messages' in chat_data:
            # Get existing message IDs
            existing_msg_ids = [msg.id for msg in chat.messages]
            
            for msg_data in chat_data['messages']:
                # Skip if this message is already in the database
                if 'id' in msg_data and msg_data['id'] in existing_msg_ids:
                    continue
                    
                msg = Message(
                    chat_id=chat.id,
                    role=msg_data['role'],
                    content=msg_data['content']
                )
                db.session.add(msg)
        
        # Generate a title if none exists
        if not chat.title and len(chat_data.get('messages', [])) > 0:
            first_msg = next((m for m in chat_data.get('messages', []) if m['role'] == 'user'), None)
            if first_msg:
                # Use the first 50 characters of the first user message as title
                chat.title = first_msg['content'][:50] + ("..." if len(first_msg['content']) > 50 else "")
        
        db.session.commit()
        return chat.id
    except Exception as e:
        db.session.rollback()
        print(f"Error saving chat to database: {e}")
        return None

def get_or_create_active_chat(chat_id):
    import uuid
    
    # If no chat_id provided, try to get the user's most recent active chat
    if not chat_id:
        if hasattr(current_user, 'id') and current_user.id:
            # Get the user's most recent chat from database
            latest_chat = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).first()
            if latest_chat:
                chat_id = str(latest_chat.id)
                print(f"[chat] Using existing chat_id {chat_id} from user's latest chat")
            else:
                # Create new chat if user has no previous chats
                chat_id = str(uuid.uuid4())
                print(f"[chat] Generated new chat_id: {chat_id} for new user")
        else:
            chat_id = str(uuid.uuid4())
            print(f"[chat] Generated new chat_id: {chat_id} for anonymous user")
    
    # Convert to string to ensure consistency
    chat_id = str(chat_id)
    
    # Create user-specific chat key to prevent cross-user contamination
    user_id = getattr(current_user, 'id', 'anonymous')
    user_chat_key = f"{user_id}_{chat_id}"
    
    if user_chat_key not in active_chats:
        chat = None
        # Try to load from database if it's a numeric ID
        if chat_id.isdigit():
            try:
                chat = db.session.get(Chat, int(chat_id))
            except Exception:
                pass
        
        if chat and chat.user_id == getattr(current_user, 'id', None):
            # Load full conversation history from database
            messages = []
            for msg in sorted(chat.messages, key=lambda x: x.created_at):
                messages.append(msg.to_dict())
            
            active_chats[user_chat_key] = {
                'messages': messages,
                'db_chat_id': chat.id,  # Track the database ID
                'user_id': user_id  # Track the user ID
            }
            print(f"[chat] Loaded chat_id {chat_id} from DB with {len(messages)} messages for user {user_id}.")
        else:
            active_chats[user_chat_key] = {
                'messages': [],
                'user_id': user_id
            }
            print(f"[chat] Initialized new chat_id {chat_id} in memory for user {user_id}.")
    else:
        print(f"[chat] Using existing chat_id {chat_id} from memory with {len(active_chats[user_chat_key]['messages'])} messages for user {user_id}.")
    
    return chat_id, active_chats[user_chat_key]

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    message = request.json.get('message', '')
    chat_id, chat_data = get_or_create_active_chat(request.json.get('chat_id', None))

    # Add user message
    chat_data['messages'].append({
        'role': 'user',
        'content': message
    })
    # Save after user message
    save_chat_to_db(chat_id, chat_data)
    
    # Check if OpenAI client is available
    if client is None:
        return jsonify({
            'success': False,
            'error': 'OpenAI API key not configured. Please contact the administrator.'
        }), 500
    
    try:
        # Use OpenAI directly for now (bypass custom model issues)
        print(f"Using OpenAI for chat response")
        
        # Prepare chat history for OpenAI - include ALL previous messages for full context
        chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT_MEDIA_BIZ}
        ]
        
        # Add conversation context summary if this is a longer conversation
        if len(chat_data['messages']) > 10:
            # Get recent context (last 8 messages) + summary of earlier context
            recent_messages = chat_data['messages'][-8:]
            earlier_messages = chat_data['messages'][:-8]
            
            # Create a summary of earlier conversation
            if earlier_messages:
                earlier_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in earlier_messages[-10:]])
                summary_prompt = f"Summarize this earlier conversation context in 2-3 sentences, focusing on business details, challenges, and solutions discussed:\n{earlier_context}"
                
                try:
                    summary_response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Summarize conversation context concisely, focusing on business details."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    context_summary = summary_response.choices[0].message.content
                    chat_history.append({"role": "system", "content": f"Previous conversation context: {context_summary}"})
                except Exception as summary_error:
                    print(f"Summary generation failed: {summary_error}")
                    pass  # If summary fails, continue without it
            
            # Add recent messages
            for msg in recent_messages:
                chat_history.append({"role": msg['role'], "content": msg['content']})
        else:
            # For shorter conversations, include all messages
            for msg in chat_data['messages']:
                chat_history.append({"role": msg['role'], "content": msg['content']})
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=chat_history,
                temperature=0.7,  # Slightly more focused responses
                max_tokens=300    # Enforce shorter responses
            )
            ai_reply = response.choices[0].message.content
            
            # Add AI reply to chat
            chat_data['messages'].append({
                'role': 'assistant',
                'content': ai_reply
            })
            # Save after AI message
            save_chat_to_db(chat_id, chat_data)
            return jsonify({
                'success': True,
                'reply': ai_reply,
                'chat_id': chat_id,
                'model_source': 'openai_fallback'
            })
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return jsonify({
                'success': False,
                'error': f'OpenAI API error: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Chat processing error: {e}")
        return jsonify({
            'success': False,
            'error': f'Chat processing error: {str(e)}'
        }), 500

@app.route('/chats')
@login_required
def get_chats():
    """Render the chat history page"""
    return render_template('chats.html')

@app.route('/api/user_chats')
@login_required
def api_user_chats():
    """API endpoint to get user's chat history"""
    # Get chats from database
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
    
    # Convert to JSON
    chats_json = []
    for chat in chats:
        messages = [
            {
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at.isoformat()
            } for msg in chat.messages
        ]
        
        chats_json.append({
            'id': chat.id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat(),
            'fact_sheet': chat.fact_sheet,
            'strategies': chat.strategies,
            'messages': messages
        })
    
    return jsonify(chats_json)

@app.route('/chat/<int:chat_id>', methods=['GET'])
@login_required
def get_chat(chat_id):
    """Get a specific chat"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    
    # Convert to JSON
    messages = [
        {
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'created_at': msg.created_at.isoformat()
        } for msg in chat.messages
    ]
    
    chat_json = {
        'id': chat.id,
        'title': chat.title,
        'created_at': chat.created_at.isoformat(),
        'updated_at': chat.updated_at.isoformat(),
        'messages': messages
    }
    
    return jsonify(chat_json)

@app.route('/chat/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    """Delete a specific chat"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    
    try:
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

def process_with_ai(message, chat_history=None):
    """Process user message with OpenAI and return response"""
    try:
        # Build the messages array for context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MEDIA_BIZ}
        ]
        
        # Add chat history for context if available
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add the current user message
        messages.append({"role": "user", "content": message})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing with AI: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def get_current_user_id():
    """Safely get current user ID with logging"""
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        logger.info(f"Authenticated user: {current_user.id} ({current_user.username})")
        return current_user.id
    logger.info("No authenticated user")
    return None

def browse_website(url):
    """
    Browse a website and extract readable content
    Returns a dictionary with content and metadata
    """
    try:
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract main content
        # Try to find main content areas
        main_content = ""
        
        # Look for common content containers
        content_selectors = [
            'main', 'article', '.content', '.main-content', '.post-content',
            '.entry-content', '.article-content', '#content', '#main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text(separator='\n', strip=True)
                break
        
        # If no main content found, get body text
        if not main_content:
            main_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = main_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Only keep substantial lines
                cleaned_lines.append(line)
        
        cleaned_content = '\n\n'.join(cleaned_lines)
        
        # Limit content length to avoid token limits
        if len(cleaned_content) > 8000:
            cleaned_content = cleaned_content[:8000] + "... [Content truncated]"
        
        return {
            'success': True,
            'url': url,
            'title': title_text,
            'content': cleaned_content,
            'status_code': response.status_code,
            'content_length': len(cleaned_content)
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Failed to fetch website: {str(e)}",
            'url': url
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error processing website: {str(e)}",
            'url': url
        }

@app.route('/synthesize')
def synthesize_org_info():
    """Synthesize information about the organization from available data"""
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    print(f"Synthesize called with refresh={refresh} for user={current_user.username if hasattr(current_user, 'username') else 'anonymous'}")
    
    # Get current user id safely
    user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
    
    # Generic response for non-authenticated users
    if not user_id:
        return jsonify({
            'success': True,
            'org_info': {
                "Organization_Overview": "Please log in to view your organization",
                "Key_Projects": ["Login required"],
                "Team_Members": ["Login required"]
            }
        })
    
    try:
        # Always analyze chats when refresh is requested
        if refresh:
            print(f"⭐ Forced refresh requested - analyzing chats for {current_user.username}")
            
            # Get user's chats with explicit filtering
            chats = Chat.query.filter_by(user_id=user_id).order_by(Chat.updated_at.desc()).limit(10).all()
            print(f"Found {len(chats)} chats for user {current_user.username}")
            
            # Extract messages
            messages = []
            for chat in chats:
                chat_messages = Message.query.filter_by(chat_id=chat.id).all()
                messages.extend([msg.content for msg in chat_messages])
            
            print(f"Extracted {len(messages)} messages for user {current_user.username}")
            
            # Prepare default info
            username = current_user.username
            default_info = {
                "Organization_Overview": f"{username}'s Organization",
                "Key_Projects": ["No projects yet"],
                "Team_Members": [f"{username}"]
            }
            
            # If we have messages, analyze them
            if messages:
                content = "\n".join(messages)
                
                # Updated regex patterns to be more precise
                org_patterns = [
                    # Pattern for "company/organization name/called/is: NAME"
                    r"(?:company|organization|organisation|business|firm|agency)\s+(?:name|called|is|:)\s+([A-Za-z0-9][A-Za-z0-9\s&'-]+)",
                    # Pattern for "I work at NAME"
                    r"(?:I work|I'm working|I am working|employed|work)\s+(?:at|for|with)\s+([A-Za-z0-9][A-Za-z0-9\s&'-]+)",
                    # Pattern for "NAME is my company"
                    r"([A-Za-z0-9][A-Za-z0-9\s&'-]+)\s+(?:is my|is our|is the)\s+(?:company|organization|organisation|business|employer)"
                ]
                
                # Try to directly extract company name
                org_name = None
                for pattern in org_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            # Clean up the matched name
                            potential_name = match.strip()
                            
                            # More aggressive cleanup to remove prefix words like "called"
                            prefixes_to_remove = ["called", "named", "is", "the"]
                            for prefix in prefixes_to_remove:
                                if potential_name.lower().startswith(prefix + " "):
                                    potential_name = potential_name[len(prefix)+1:].strip()
                            
                            # Remove common noise words at the end
                            noise_words = ['that', 'which', 'and', 'is', 'a', 'an', 'the', 'called', 'named']
                            for word in noise_words:
                                if potential_name.lower().endswith(f" {word}"):
                                    potential_name = potential_name[:-len(word)-1].strip()
                            
                            # Also remove trailing punctuation
                            potential_name = re.sub(r'[.,;:!?]+$', '', potential_name).strip()
                            
                            # For names like "called TOTAL MEDIA", extract just "TOTAL MEDIA"
                            if "called " in potential_name.lower():
                                potential_name = potential_name.lower().split("called ")[1].strip().upper()
                            
                            if len(potential_name) > 3:  # Avoid short meaningless matches
                                org_name = potential_name
                                print(f"🔍 Direct regex match found organization: '{org_name}'")
                                break
                    
                    if org_name:
                        break
                
                # Also try to find project names
                project_patterns = [
                    r"(?:project|initiative|campaign) (?:called|named|titled) ([A-Za-z0-9\s&'-]+?)(?:\.|\band\b|\bthat\b|\bwhich\b|\,|\;|$)",
                    r"working on ([A-Za-z0-9\s&'-]+?) (?:project|initiative|campaign)"
                ]
                
                projects = []
                for pattern in project_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        project = match.strip()
                        if len(project) > 3 and project not in projects:
                            projects.append(project)
                
                if org_name:
                    # If we found a direct mention, use it
                    default_info["Organization_Overview"] = org_name
                
                if projects:
                    default_info["Key_Projects"] = projects[:5]  # Limit to 5 projects
                
                # Trim content if too long
                if len(content) > 8000:
                    content = content[:8000] + "..."
                
                print(f"Sending {len(content)} characters to OpenAI")
                
                try:
                    # Call OpenAI with a very explicit prompt
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": """You are an expert at extracting precise organization names. Given a conversation, your ONLY task is to extract the exact organization name mentioned. ONLY return the organization name without any prefixes like "called" or "named". Do not include any descriptions or additional text.

Return a JSON object with the following structure:
{
  "Organization_Overview": "EXACT ORGANIZATION NAME",
  "Key_Projects": ["Project 1", "Project 2"],
  "Team_Members": ["Person 1", "Person 2"]
}"""},
                            {"role": "user", "content": f"Find the exact organization name in this text. If someone says 'I work at Company X' or 'My company is called Company X', just return 'Company X'. DO NOT include words like 'called', 'named', or 'that': {content}"}
                        ],
                        temperature=0,
                        max_tokens=1000
                    )
                    
                    org_info_text = response.choices[0].message.content
                    print(f"AI response received: {org_info_text[:100]}...")
                    
                    try:
                        # Try to parse as JSON
                        org_data = json.loads(org_info_text)
                        print("Successfully parsed JSON response")
                    except json.JSONDecodeError:
                        print("JSON parse error, looking for code block")
                        # Look for JSON in code blocks
                        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                        json_match = re.search(json_pattern, org_info_text)
                        
                        if json_match:
                            try:
                                org_data = json.loads(json_match.group(1))
                                print("Successfully parsed JSON from code block")
                            except json.JSONDecodeError:
                                print("JSON parsing failed, using defaults")
                                org_data = default_info
                        else:
                            print("No JSON found, using regex-extracted data")
                            org_data = default_info
                    
                    # If the AI couldn't find an organization name but we found one with regex, use that
                    if org_name:
                        if (not org_data.get("Organization_Overview") or 
                            "unknown" in org_data.get("Organization_Overview", "").lower() or
                            len(org_data.get("Organization_Overview", "")) < 3):
                            print(f"Using regex-found org name: {org_name}")
                            org_data["Organization_Overview"] = org_name
                        else:
                            # Extra cleanup for the AI-provided org name
                            ai_org_name = org_data["Organization_Overview"]
                            
                            # Handle "called XXX" explicitly
                            if "called " in ai_org_name.lower():
                                ai_org_name = ai_org_name.lower().split("called ")[1].strip()
                                org_data["Organization_Overview"] = ai_org_name.upper() if ai_org_name.isupper() else ai_org_name
                                print(f"Cleaned up AI org name to: {org_data['Organization_Overview']}")
                    
                    # Save to database
                    org_info = OrganizationInfo.query.filter_by(user_id=user_id).first()
                    if not org_info:
                        org_info = OrganizationInfo(user_id=user_id)
                        db.session.add(org_info)
                    
                    org_info.org_info = json.dumps(org_data)
                    org_info.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
                    
                    print(f"✅ Saved new organization info: {org_data}")
                    
                    return jsonify(org_data)
                    
                except Exception as ai_error:
                    print(f"AI processing error: {str(ai_error)}")
                    return jsonify(default_info)
            else:
                # No messages, use defaults
                print(f"No messages for user {username}, using defaults")
                return jsonify(default_info)
        
        # Not a refresh, so return existing data if available
        org_info = OrganizationInfo.query.filter_by(user_id=user_id).first()
        if org_info and org_info.org_info:
            try:
                org_data = json.loads(org_info.org_info)
                print(f"Returning cached org info: {org_data}")
                return jsonify({
                    'success': True,
                    'org_info': org_data,
                    'source': 'cached'
                })
            except json.JSONDecodeError:
                print("Error parsing cached JSON, forcing refresh")
                # Recursive call with refresh=True
                return synthesize_org_info() 
        
        # No valid existing data, run a fresh analysis
        print(f"No valid existing data for user {current_user.username}, running fresh analysis")
        
        # Set refresh parameter in the request
        request.args = dict(request.args)
        request.args['refresh'] = 'true'
        
        # Call again with refresh=True
        return synthesize_org_info()
            
    except Exception as e:
        print(f"❌ Error in synthesize_org_info: {str(e)}")
        username = current_user.username if hasattr(current_user, 'username') else "Unknown"
        return jsonify({
            'success': True,
            'org_info': {
                "Organization_Overview": f"{username}'s Organization",
                "Key_Projects": ["Error occurred", "Please try again"],
                "Team_Members": [username]
            },
            'source': 'error_fallback'
        })

@app.route('/lessons')
@login_required
def get_lessons():
    try:
        # Get user's lesson progress
        user_lessons = UserLesson.query.filter_by(user_id=current_user.id).all()
        completed_lessons = {ul.lesson_id for ul in user_lessons if ul.completed}
        
        # Get current lesson or next available
        current_lesson = Lesson.query.filter(
            ~Lesson.id.in_(completed_lessons)
        ).order_by(Lesson.order).first()
        
        if not current_lesson:
            # Generate new lesson using OpenAI
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an AI workflow expert creating a lesson plan.
                    Create a lesson about implementing AI in workflows. Include:
                    1. A clear title
                    2. The main lesson content with practical examples
                    3. An exercise for practice
                    4. Key takeaways
                    Format in markdown."""},
                    {"role": "user", "content": "Generate a new lesson about AI workflows"}
                ]
            )
            
            lesson_content = response.choices[0].message.content
            
            # Create new lesson
            new_lesson = Lesson(
                title=f"Lesson {Lesson.query.count() + 1}",
                content=lesson_content,
                order=Lesson.query.count() + 1
            )
            db.session.add(new_lesson)
            db.session.commit()
            
            current_lesson = new_lesson
        
        return jsonify({
            "success": True,
            "lesson": {
                "id": current_lesson.id,
                "title": current_lesson.title,
                "content": current_lesson.content,
                "completed": current_lesson.id in completed_lessons
            }
        })
        
    except Exception as e:
        print(f"Error in get_lessons: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/lessons/complete/<int:lesson_id>', methods=['POST'])
@login_required
def complete_lesson(lesson_id):
    try:
        user_lesson = UserLesson.query.filter_by(
            user_id=current_user.id,
            lesson_id=lesson_id
        ).first()
        
        if not user_lesson:
            user_lesson = UserLesson(
                user_id=current_user.id,
                lesson_id=lesson_id
            )
            db.session.add(user_lesson)
        
        user_lesson.completed = True
        user_lesson.last_accessed = datetime.utcnow()
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/lessons-page')
@login_required
def lessons_page():
    return render_template('lessons.html')

@app.route('/lessons/create', methods=['POST'])
@login_required
def create_new_lesson():
    try:
        # Generate new lesson using OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an AI workflow expert creating a lesson plan.
                Create a lesson about implementing AI in workflows. Include:
                1. A clear title
                2. The main lesson content with practical examples
                3. An exercise for practice
                4. Key takeaways
                Format in markdown."""},
                {"role": "user", "content": "Generate a new lesson about AI workflows"}
            ]
        )
        
        lesson_content = response.choices[0].message.content
        
        # Create new lesson
        new_lesson = Lesson(
            title=f"Lesson {Lesson.query.count() + 1}",
            content=lesson_content,
            order=Lesson.query.count() + 1
        )
        db.session.add(new_lesson)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "lesson": {
                "id": new_lesson.id,
                "title": new_lesson.title,
                "content": new_lesson.content
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/map')
@login_required
def map():
    # Render the map template directly
    return render_template('map.html')

@app.route('/show-map')
@login_required
def show_map():
    return render_template('map.html')

@app.route('/api/user-locations')
@login_required
def get_user_locations():
    users = User.query.all()
    return jsonify({
        'users': [{
            'username': user.username,
            'latitude': user.latitude,
            'longitude': user.longitude,
            'location_name': user.location_name
        } for user in users if user.latitude and user.longitude]
    })

@app.route('/update-location', methods=['POST'])
@login_required
def update_location():
    data = request.json
    current_user.latitude = data.get('latitude')
    current_user.longitude = data.get('longitude')
    current_user.location_name = data.get('location_name')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/add-fact', methods=['POST'])
@login_required
def add_fact():
    try:
        data = request.json
        fact_content = data.get('fact', '')
        
        # Store the fact
        new_fact = OrganizationFact(
            user_id=current_user.id,
            fact=fact_content
        )
        db.session.add(new_fact)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Fact added successfully"
        })
        
    except Exception as e:
        print(f"Error adding fact: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze-chat', methods=['POST'])
@login_required
def analyze_chat():
    try:
        data = request.json
        message_content = data.get('message', '')
        
        # Get analysis from GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a media analysis expert. 
                Analyze the given content and provide insights about:
                1. Key themes and topics
                2. Potential implications
                3. Recommendations
                Format your response in clear sections."""},
                {"role": "user", "content": message_content}
            ]
        )
        
        analysis = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/translate', methods=['POST'])
@login_required
def translate_text():
    try:
        data = request.json
        text = data.get('text', '')
        target_language = data.get('target_language', '')
        
        # Get translation from GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Only respond with the translation, no additional text."},
                {"role": "user", "content": text}
            ]
        )
        
        translated_text = response.choices[0].message.content
        
        # Store translation
        translation = Translation(
            user_id=current_user.id,
            original_text=text,
            translated_text=translated_text,
            source_language='auto',
            target_language=target_language
        )
        db.session.add(translation)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "translation": translated_text,
            "translation_id": translation.id
        })
        
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/rate-translation', methods=['POST'])
@login_required
def rate_translation():
    try:
        data = request.json
        translation_id = data.get('translation_id')
        rating = data.get('rating')
        
        translation = Translation.query.get(translation_id)
        if translation and translation.user_id == current_user.id:
            translation.rating = rating
            db.session.commit()
            
            return jsonify({
                "success": True,
                "message": "Rating saved successfully"
            })
        
        return jsonify({
            "success": False,
            "error": "Translation not found"
        }), 404
        
    except Exception as e:
        print(f"Error rating translation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/translate')
@login_required
def translate_page():
    return render_template('translate.html')

@app.route('/submit-correction', methods=['POST'])
@login_required
def submit_correction():
    try:
        data = request.json
        translation_id = data.get('translation_id')
        corrected_text = data.get('corrected_text')
        
        # Get original translation
        translation = Translation.query.get(translation_id)
        if translation and translation.user_id == current_user.id:
            # Store the correction
            feedback = TranslationFeedback(
                translation_id=translation_id,
                user_id=current_user.id,
                corrected_text=corrected_text,
                source_language=translation.source_language,
                target_language=translation.target_language
            )
            db.session.add(feedback)
            db.session.commit()
            
            return jsonify({
                "success": True,
                "message": "Correction saved successfully"
            })
        
        return jsonify({
            "success": False,
            "error": "Translation not found"
        }), 404
        
    except Exception as e:
        print(f"Error submitting correction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/add-location', methods=['POST'])
@login_required
def add_location():
    try:
        data = request.json
        new_location = Location(
            user_id=current_user.id,
            name=data['name'],
            description=data.get('description', '')
        )
        db.session.add(new_location)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Location added successfully"
        })
        
    except Exception as e:
        print(f"Error adding location: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get-locations')
@login_required
def get_locations():
    try:
        locations = Location.query.filter_by(user_id=current_user.id).all()
        return jsonify({
            "success": True,
            "locations": [{
                "name": loc.name,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "description": loc.description
            } for loc in locations]
        })
        
    except Exception as e:
        print(f"Error getting locations: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/recommended-ai-tools')
@login_required
def recommended_ai_tools():
    return render_template('recommended_ai_tools.html')

@app.route('/generate-insights')
@login_required
def generate_insights():
    return render_template('generate_insights.html')

@app.route('/your-info')
def your_info():
    return render_template('your_info.html')

# Will be moved earlier in file

# Admin routes
@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard showing system overview"""
    user_count = User.query.count()
    analysis_count = MediaAnalysis.query.count()
    chat_count = Chat.query.count()
    lesson_count = Lesson.query.count()
    feedback_count = Feedback.query.count()
    message_count = Message.query.count()
    
    # Count admin users
    admin_count = 0
    for user in User.query.all():
        if hasattr(user, 'is_admin') and user.is_admin:
            admin_count += 1
    
    # Count strategies
    try:
        from backend.strategies_crawler import StrategyEntry
        strategy_count = StrategyEntry.query.count()
    except:
        strategy_count = 0
    
    # Get Flask version
    import flask
    flask_version = flask.__version__
    
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    
    # Plan/report counts
    try:
        plan_count = ImplementationPlan.query.count()
        report_count = DailyReport.query.count()
        cheatsheet_count = CheatSheet.query.count()
    except Exception:
        plan_count = report_count = cheatsheet_count = 0

    return render_template(
        'admin/dashboard.html', 
        user_count=user_count,
        analysis_count=analysis_count,
        chat_count=chat_count,
        lesson_count=lesson_count,
        feedback_count=feedback_count,
        message_count=message_count,
        strategy_count=strategy_count,
        recent_users=recent_users,
        admin_count=admin_count,
        flask_version=flask_version,
        plan_count=plan_count,
        report_count=report_count,
        cheatsheet_count=cheatsheet_count
    )

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin page to view all users"""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def admin_user_detail(user_id):
    """Admin page to view details of a specific user"""
    user = User.query.get_or_404(user_id)
    
    # Get user's media analyses
    analyses = MediaAnalysis.query.filter_by(user_id=user_id).order_by(MediaAnalysis.created_at.desc()).all()
    
    # Get user's chats
    chats = Chat.query.filter_by(user_id=user_id).order_by(Chat.created_at.desc()).all()
    
    # Get user's lesson progress
    lesson_progress = UserLesson.query.filter_by(user_id=user_id).all()
    
    # Get user's translations
    translations = Translation.query.filter_by(user_id=user_id).order_by(Translation.created_at.desc()).all()
    
    return render_template(
        'admin/user_detail.html',
        user=user,
        analyses=analyses,
        chats=chats,
        lesson_progress=lesson_progress,
        translations=translations
    )

@app.route('/admin/chats')
@login_required
@admin_required
def admin_chats():
    """Admin page to view all chats"""
    chats = Chat.query.order_by(Chat.created_at.desc()).all()
    return render_template('admin/chats.html', chats=chats)

@app.route('/admin/chat/<int:chat_id>')
@login_required
@admin_required
def admin_chat_detail(chat_id):
    """Admin page to view details of a specific chat"""
    chat = Chat.query.get_or_404(chat_id)
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    return render_template('admin/chat_detail.html', chat=chat, messages=messages)

@app.route('/admin/analyses')
@login_required
@admin_required
def admin_analyses():
    """Admin page to view all media analyses"""
    analyses = MediaAnalysis.query.order_by(MediaAnalysis.created_at.desc()).all()
    return render_template('admin/analyses.html', analyses=analyses)

@app.route('/admin/lessons')
@login_required
@admin_required
def admin_lessons():
    """Admin page to view all lessons"""
    lessons = Lesson.query.order_by(Lesson.order).all()
    return render_template('admin/lessons.html', lessons=lessons)

@app.route('/admin/create_admin', methods=['GET', 'POST'])
@login_required
@admin_required
def create_admin():
    """Admin page to create a new admin user"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('create_admin'))
        
        # Create new admin user
        new_admin = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=True
        )
        
        db.session.add(new_admin)
        db.session.commit()
        
        flash(f'Admin user {username} created successfully', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/create_admin.html')

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent removing admin status from yourself
    if user.id == current_user.id:
        flash('You cannot remove your own admin status', 'danger')
        return redirect(url_for('admin_users'))
    
    # Toggle admin status
    user.is_admin = not user.is_admin
    db.session.commit()
    
    status = 'granted' if user.is_admin else 'removed'
    flash(f'Admin status {status} for {user.username}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/feedback')
@login_required
@admin_required
def admin_feedback():
    """Admin page to view all user feedback"""
    feedback_list = Feedback.query.order_by(Feedback.created_at.desc()).all()
    return render_template('admin/feedback.html', feedback_list=feedback_list)

@app.route('/admin/feedback/<int:feedback_id>')
@login_required
@admin_required
def admin_feedback_detail(feedback_id):
    """Admin page to view details of specific feedback"""
    feedback = Feedback.query.get_or_404(feedback_id)
    return render_template('admin/feedback_detail.html', feedback=feedback)

@app.route('/admin/feedback/<int:feedback_id>/update', methods=['POST'])
@login_required
@admin_required
def update_feedback_status(feedback_id):
    """Update feedback status and admin notes"""
    feedback = Feedback.query.get_or_404(feedback_id)
    
    feedback.status = request.form.get('status', feedback.status)
    feedback.admin_notes = request.form.get('admin_notes', feedback.admin_notes)
    
    db.session.commit()
    flash('Feedback updated successfully', 'success')
    return redirect(url_for('admin_feedback_detail', feedback_id=feedback_id))

@app.route('/admin/training')
@login_required
@admin_required
def admin_training():
    """Admin page for AI model training management"""
    return render_template('admin/training.html')

# ===== Implementation Plans =====
@app.route('/admin/plans')
@login_required
@admin_required
def admin_plans():
    plans = ImplementationPlan.query.order_by(ImplementationPlan.created_at.desc()).all()
    return render_template('admin/plans.html', plans=plans)

@app.route('/admin/plans/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_plans_create():
    if request.method == 'POST':
        title = request.form.get('title') or 'Implementation Plan'
        summary = request.form.get('summary')
        plan = ImplementationPlan(user_id=current_user.id, title=title, summary=summary)
        db.session.add(plan)
        db.session.commit()
        flash('Plan created', 'success')
        return redirect(url_for('admin_plans'))
    return render_template('admin/plan_form.html', plan=None)

@app.route('/admin/plans/<int:plan_id>')
@login_required
@admin_required
def admin_plans_detail(plan_id):
    plan = ImplementationPlan.query.get_or_404(plan_id)
    return render_template('admin/plan_detail.html', plan=plan)

@app.route('/admin/plans/<int:plan_id>/generate', methods=['POST'])
@login_required
@admin_required
def admin_plans_generate(plan_id):
    plan = ImplementationPlan.query.get_or_404(plan_id)
    try:
        if not client:
            return jsonify({'success': False, 'error': 'OpenAI client not available'}), 500
        prompt = f"""
        You are Highlander, an AI strategy mentor for newsrooms. Create a concise, actionable implementation plan.
        Title: {plan.title}
        Context: {plan.summary or 'Mentoring newsroom to implement AI ethically to support strategic goals.'}
        Include sections: Objectives, Workstreams, Milestones, Tasks (with owners and due dates TBD), Risks, Metrics.
        Output in GitHub-flavored markdown with clear headings and bullet points.
        """
        resp = client.chat.completions.create(model='gpt-4o-mini', messages=[
            {"role": "system", "content": "You specialize in practical implementation planning for media organizations."},
            {"role": "user", "content": prompt}
        ])
        content = resp.choices[0].message.content
        plan.tasks = content
        db.session.commit()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== Daily Reports =====
@app.route('/admin/reports')
@login_required
@admin_required
def admin_reports():
    reports = DailyReport.query.order_by(DailyReport.date.desc()).limit(50).all()
    plans = ImplementationPlan.query.order_by(ImplementationPlan.created_at.desc()).all()
    return render_template('admin/reports.html', reports=reports, plans=plans)

@app.route('/admin/reports/create', methods=['POST'])
@login_required
@admin_required
def admin_reports_create():
    plan_id = request.form.get('plan_id')
    content = request.form.get('content')
    report = DailyReport(user_id=current_user.id, plan_id=plan_id, content=content or '')
    db.session.add(report)
    db.session.commit()
    flash('Report created', 'success')
    return redirect(url_for('admin_reports'))

@app.route('/admin/reports/generate', methods=['POST'])
@login_required
@admin_required
def admin_reports_generate():
    try:
        plan_id = request.json.get('plan_id') if request.is_json else request.form.get('plan_id')
        plan = ImplementationPlan.query.get(plan_id) if plan_id else None

        # Pull high-severity recent threats from DataSafe to include in the report
        high_threats = []
        try:
            from backend.datasafe_integration import DataSafeProcessor
            dsp = DataSafeProcessor()
            high_threats = dsp.get_high_severity_threats(hours=24)[:5]
        except Exception:
            high_threats = []

        # Gather recent strategy/news context
        recent_news = News.query.order_by(News.created_at.desc()).limit(5).all()
        news_bullets = "\n".join([f"- {n.title} ({n.source_name or 'source'})" for n in recent_news])
        threat_bullets = "\n".join([f"- [{t.get('severity')}] {t.get('title')} — risk {t.get('risk', '')}" for t in high_threats])

        if not client:
            return jsonify({'success': False, 'error': 'OpenAI client not available'}), 500

        prompt = f"""
        Create a daily implementation report for a newsroom AI program.
        If a plan is provided, align to it. Include sections: Progress, Blockers, Next Steps, Metrics to Track, Risks/Security, News/Signals.
        Plan: {plan.title if plan else 'No specific plan'} — {plan.summary if plan else ''}
        Recent News:\n{news_bullets or '- None'}
        High Severity Threats in last 24h:\n{threat_bullets or '- None'}
        Keep it concise, actionable, and ready to share with PMs.
        """
        resp = client.chat.completions.create(model='gpt-4o-mini', messages=[
            {"role": "system", "content": "You write crisp, actionable status reports for AI implementations in media."},
            {"role": "user", "content": prompt}
        ])
        content = resp.choices[0].message.content

        report = DailyReport(user_id=current_user.id, plan_id=plan.id if plan else None, content=content)
        db.session.add(report)
        db.session.commit()
        return jsonify({'success': True, 'report_id': report.id, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== Cheat Sheets =====
@app.route('/admin/cheatsheets')
@login_required
@admin_required
def admin_cheatsheets():
    items = CheatSheet.query.order_by(CheatSheet.created_at.desc()).all()
    return render_template('admin/cheatsheets.html', items=items)

@app.route('/admin/cheatsheets/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_cheatsheets_create():
    if request.method == 'POST':
        title = request.form.get('title') or 'Cheat Sheet'
        category = request.form.get('category')
        content = request.form.get('content')
        item = CheatSheet(user_id=current_user.id, title=title, category=category, content=content or '')
        db.session.add(item)
        db.session.commit()
        flash('Cheat sheet created', 'success')
        return redirect(url_for('admin_cheatsheets'))
    return render_template('admin/cheatsheet_form.html')

@app.route('/admin/cheatsheets/generate', methods=['POST'])
@login_required
@admin_required
def admin_cheatsheets_generate():
    try:
        topic = request.json.get('topic') if request.is_json else request.form.get('topic')
        if not topic:
            return jsonify({'success': False, 'error': 'Missing topic'}), 400
        if not client:
            return jsonify({'success': False, 'error': 'OpenAI client not available'}), 500
        prompt = f"Create a concise implementation cheat sheet for newsroom AI on: {topic}. Use headings, bullets, and include do/don'ts and quick steps."
        resp = client.chat.completions.create(model='gpt-4o-mini', messages=[
            {"role": "system", "content": "You produce practical, one-page cheat sheets for AI implementation in newsrooms."},
            {"role": "user", "content": prompt}
        ])
        content = resp.choices[0].message.content
        item = CheatSheet(user_id=current_user.id, title=topic, content=content)
        db.session.add(item)
        db.session.commit()
        return jsonify({'success': True, 'id': item.id, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/datasafe-hf')
@login_required
@admin_required
def admin_datasafe_hf():
    """Admin DataSafe Hugging Face integration dashboard."""
    return render_template('admin/datasafe_hf.html')

@app.route('/admin/training/collect-data', methods=['POST'])
@login_required
@admin_required
def collect_training_data():
    """Collect data for training"""
    try:
        # Use the real data collection function
        return real_collect_training_data()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/start-training', methods=['POST'])
@login_required
@admin_required
def start_training():
    """Start model training"""
    try:
        from training.model_trainer import HighlanderModelTrainer
        from training.training_history import get_training_history
        
        # Check if there's enough new data to warrant training
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(basedir, 'training', 'training_data')
        
        history = get_training_history()
        retrain_analysis = history.should_retrain(data_dir, min_new_data_threshold=5)
        
        if not retrain_analysis['should_retrain']:
            return jsonify({
                'success': False,
                'error': f'Not enough new data to warrant training. {retrain_analysis["reason"]}'
            }), 400
        
        # Start training in background
        import threading
        
        def train_model():
            import os
            basedir = os.path.abspath(os.path.dirname(__file__))
            config_path = os.path.join(basedir, 'training', 'training_config.yaml')
            data_dir = os.path.join(basedir, 'training', 'training_data')
            output_dir = os.path.join(basedir, 'training', 'models')
            
            trainer = HighlanderModelTrainer(
                config_path=config_path,
                data_dir=data_dir,
                output_dir=output_dir
            )
            model_path = trainer.train_model()
            
            # Record the training session
            training_data_path = os.path.join(data_dir, 'processed', 'training_dataset.json')
            training_stats = {
                'total_examples': retrain_analysis['new_data']['total_tokens'],
                'new_conversations': retrain_analysis['new_data']['conversations'],
                'new_pdfs': retrain_analysis['new_data']['pdfs'],
                'new_research': retrain_analysis['new_data']['research_papers'],
                'new_feedback': retrain_analysis['new_data']['feedback_entries']
            }
            
            history.record_training_session(
                training_data_path=training_data_path,
                model_path=model_path,
                training_stats=training_stats
            )
            
            print(f"Training completed: {model_path}")
        
        training_thread = threading.Thread(target=train_model)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Model training started in background. {retrain_analysis["reason"]}',
            'retrain_analysis': retrain_analysis
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/status')
@login_required
@admin_required
def training_status():
    """Get training status and model information"""
    try:
        from training.model_manager import get_model_manager
        import os
        import json
        
        manager = get_model_manager()
        model_info = manager.get_model_info()
        performance_metrics = manager.get_performance_metrics()
        
        # Check for training completion by reading metadata
        training_completed = False
        training_metadata = None
        try:
            basedir = os.path.abspath(os.path.dirname(__file__))
            metadata_file = os.path.join(basedir, 'training', 'models', 'training_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    training_metadata = json.load(f)
                    training_completed = True
        except Exception as e:
            print(f"Error reading training metadata: {e}")
        
        # Get real training data statistics
        try:
            dataset_stats_file = os.path.join(basedir, 'training', 'training_data', 'processed', 'dataset_stats.json')
            if os.path.exists(dataset_stats_file):
                with open(dataset_stats_file, 'r') as f:
                    real_stats = json.load(f)
                    model_info['real_data_stats'] = real_stats
                    model_info['total_examples'] = real_stats.get('total_examples', 0)
                    model_info['total_tokens'] = real_stats.get('total_tokens', 0)
            else:
                model_info['real_data_stats'] = None
                model_info['total_examples'] = 0
                model_info['total_tokens'] = 0
        except Exception as e:
            print(f"Error reading real stats: {e}")
            model_info['real_data_stats'] = None
            model_info['total_examples'] = 0
            model_info['total_tokens'] = 0
        
        # Update model_info to reflect actual training status
        if training_completed and training_metadata:
            model_info['training_completed'] = True
            model_info['training_date'] = training_metadata.get('training_completed_at')
            model_info['training_examples'] = training_metadata.get('training_examples')
            model_info['training_loss'] = training_metadata.get('performance', {}).get('training_loss')
            model_info['validation_accuracy'] = training_metadata.get('performance', {}).get('validation_accuracy')
            model_info['training_steps'] = training_metadata.get('performance', {}).get('training_steps')
            model_info['data_stats'] = training_metadata.get('data_stats')
        else:
            model_info['training_completed'] = False
        
        # Check for recent training errors
        training_errors = []
        try:
            # Look for any recent error logs or failed training attempts
            log_file = os.path.join(basedir, 'training', 'training_errors.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_errors = f.readlines()[-10:]  # Last 10 lines
                    training_errors = [line.strip() for line in recent_errors if line.strip()]
        except:
            pass
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'training_errors': training_errors,
            'training_metadata': training_metadata
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/deploy-model', methods=['POST'])
@login_required
@admin_required
def deploy_model():
    """Deploy latest trained model"""
    try:
        from training.model_manager import get_model_manager
        import os
        import shutil
        
        # Get the latest trained model
        basedir = os.path.abspath(os.path.dirname(__file__))
        models_dir = os.path.join(basedir, 'training', 'models')
        deployment_dir = os.path.join(models_dir, 'deployment')
        
        # Find the latest model checkpoint
        model_checkpoints = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                model_checkpoints.append(item_path)
        
        if not model_checkpoints:
            return jsonify({
                'success': False,
                'error': 'No trained models found. Please complete training first.'
            }), 404
        
        # Get the latest checkpoint (highest number)
        latest_checkpoint = max(model_checkpoints, key=lambda x: int(x.split('-')[-1]))
        
        # Create deployment directory
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Copy model files to deployment
        if os.path.exists(os.path.join(deployment_dir, 'model')):
            shutil.rmtree(os.path.join(deployment_dir, 'model'))
        
        shutil.copytree(latest_checkpoint, os.path.join(deployment_dir, 'model'))
        
        # Create deployment info
        deployment_info = {
            'model_path': os.path.join(deployment_dir, 'model'),
            'deployed_at': datetime.now().isoformat(),
            'checkpoint_source': latest_checkpoint
        }
        
        with open(os.path.join(deployment_dir, 'deployment_info.json'), 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Update the model manager
        manager = get_model_manager()
        success = manager.update_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model deployed successfully',
                'model_path': deployment_info['model_path'],
                'deployed_at': deployment_info['deployed_at']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model deployed but failed to load'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/model-status')
@login_required
@admin_required
def model_status():
    """Get current model status and usage statistics"""
    try:
        from training.model_manager import get_model_manager
        from training.training_history import get_training_history
        
        manager = get_model_manager()
        model_info = manager.get_model_info()
        performance_metrics = manager.get_performance_metrics()
        
        # Get training history
        history = get_training_history()
        training_summary = history.get_training_summary()
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'training_summary': training_summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/history')
@login_required
@admin_required
def training_history():
    """Get detailed training history"""
    try:
        from training.training_history import get_training_history
        import os
        
        history = get_training_history()
        basedir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(basedir, 'training', 'training_data')
        
        # Get current retrain analysis
        retrain_analysis = history.should_retrain(data_dir, min_new_data_threshold=5)
        
        return jsonify({
            'success': True,
            'training_summary': history.get_training_summary(),
            'retrain_analysis': retrain_analysis,
            'full_history': history.history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/preview-data')
@login_required
@admin_required
def preview_training_data():
    """Preview the collected training data"""
    try:
        import json
        import os
        
        # Try to find the training data file
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        training_data_path = os.path.join(basedir, 'training', 'training_data', 'processed', 'training_dataset.json')
        
        if not os.path.exists(training_data_path):
            return jsonify({
                'success': False,
                'message': 'No training data found. Please run data collection first.'
            }), 404
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Return a preview (first 10 examples) to avoid overwhelming the browser
        preview_data = {
            'total_examples': len(data),
            'preview_examples': data[:10] if len(data) > 10 else data,
            'full_data_available': len(data) > 10
        }
        
        return jsonify({
            'success': True,
            'data': preview_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading training data: {str(e)}'
        }), 500

@app.route('/admin/training/download-data')
@login_required
@admin_required
def download_training_data():
    """Download the complete training data as JSON file"""
    try:
        import os
        from flask import send_file
        
        training_data_path = os.path.join(basedir, 'training', 'training_data', 'processed', 'training_dataset.json')
        
        if not os.path.exists(training_data_path):
            return jsonify({
                'success': False,
                'message': 'No training data found. Please run data collection first.'
            }), 404
        
        return send_file(
            training_data_path,
            as_attachment=True,
            download_name='training_data.json',
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error downloading training data: {str(e)}'
        }), 500

@app.route('/content-calendar')
@login_required
def content_calendar():
    """Content Calendar page for AI Utility"""
    # Set the platform in session
    session['platform'] = 'ai_utility'
    return render_template('content_calendar.html', hide_right_sidebar=True)

@app.cli.command("reset-db")
def reset_db():
    """Reset the database tables."""
    db_path = os.path.join(basedir, "instance", "media_analysis.db")
    
    # Create a backup of the old database if it exists
    if os.path.exists(db_path):
        backup_path = db_path + ".backup"
        try:
            import shutil
            shutil.copy2(db_path, backup_path)
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {str(e)}")
        
        # Try to remove the corrupted file
        try:
            os.remove(db_path)
            print(f"Removed existing database file {db_path}")
        except Exception as e:
            print(f"Could not remove existing database: {str(e)}")
            # If we can't remove it, try to create a new database path
            db_path = os.path.join(basedir, "instance", "media_analysis_new.db")
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
            print(f"Using new database path: {db_path}")
    
    # Ensure the instance directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create all tables in the new database
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully.")
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")

# Create database tables
with app.app_context():
    # First, create all tables that are defined in models
    db.create_all()
    
    # Check if User model has all required columns
    from sqlalchemy import inspect, text
    inspector = inspect(db.engine)
    
    # Use 'users' table name instead of 'user' to match the model definition
    existing_columns = [col['name'] for col in inspector.get_columns('users')]
    
    # Define all expected columns based on your User model
    expected_columns = {
        'is_admin': 'BOOLEAN DEFAULT 0',
        'last_login': 'DATETIME',
        'latitude': 'FLOAT',
        'longitude': 'FLOAT',
        'location_name': 'VARCHAR(200)'
    }
    
    # Add any missing columns
    with db.engine.connect() as conn:
        for column_name, column_type in expected_columns.items():
            if column_name not in existing_columns:
                print(f"Adding {column_name} column to User model")
                # Use text() for raw SQL execution
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"))
                conn.commit()
                print(f"{column_name} column added successfully")
    
    # Check if any admin user exists
    admin_exists = False
    try:
        admin_user = User.query.filter_by(is_admin=True).first()
        if admin_user:
            admin_exists = True
            print(f"Admin user exists: {admin_user.username}")
    except Exception as e:
        print(f"Error checking for admin users: {str(e)}")
    
    # Create default admin if none exists
    if not admin_exists:
        print("Creating default admin user")
        try:
            admin_user = User(
                username="admin",
                email="admin@example.com",
                password_hash=generate_password_hash("admin123"),
                is_admin=True  # Set is_admin directly in constructor
            )
            
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin user created with username 'admin' and password 'admin123'")
        except Exception as e:
            db.session.rollback()
            print(f"Error creating admin user: {str(e)}")

@app.route('/justice-ai')
def justice_ai():
    return render_template('justice_ai.html')

@app.route('/language-ai')
def language_ai():
    return render_template('language_ai.html')

@app.route('/training-lab')
def training_lab():
    return render_template('training_lab.html')

@app.route('/crimecast')
def crimecast():
    return render_template('crimecast.html')

# Root route - redirect based on user role
@app.route('/')
def root():
    # For now, always show the login page for debugging
    return render_template('auth_landing.html')

# User Dashboard - Simple chat interface for regular users
@app.route('/user-dashboard')
@login_required
def user_dashboard():
    """Simple dashboard for regular users - just the chat interface"""
    if hasattr(current_user, 'is_admin') and current_user.is_admin:
        # Admins get redirected to the full landing page
        return redirect(url_for('landing_page1'))
    
    # Regular users get the simple chat interface
    return render_template('user_dashboard.html')

@app.route('/my-chats')
@login_required
def my_chats():
    """Simple chat history for regular users"""
    if hasattr(current_user, 'is_admin') and current_user.is_admin:
        # Admins use the full chat management interface
        return redirect(url_for('get_chats'))
    
    # Get user's chats
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).all()
    return render_template('user_chats.html', chats=chats)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password.', 'danger')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=request.form.get('remember'))
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            
            # Redirect based on user role
            if hasattr(user, 'is_admin') and user.is_admin:
                return redirect(url_for('admin_dashboard'))  # Admin dashboard for admins
            else:
                return redirect(url_for('user_dashboard'))  # Simple chat for regular users
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            flash('Username or email already exists.', 'danger')
            return render_template('register.html')
        
        try:
            # Create new user
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                created_at=datetime.now(timezone.utc)
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('root'))
            
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'danger')
            print(f"Registration error: {str(e)}")
    
    return render_template('register.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('root'))

# Landing pages
@app.route('/landing-page-1')
def landing_page1():
    # Redirect regular users to their simplified dashboard
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if not (hasattr(current_user, 'is_admin') and current_user.is_admin):
            return redirect(url_for('user_dashboard'))
    
    # Only admins and unauthenticated users see the full landing page
    return render_template('landing_page1.html')

@app.route('/landing-page-2')  
def landing_page2():
    return render_template('landing_page2.html')

@app.route('/datasafe-home')
def datasafe_home():
    # Redirect regular users to their simplified dashboard
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if not (hasattr(current_user, 'is_admin') and current_user.is_admin):
            return redirect(url_for('user_dashboard'))
    return render_template('datasafe_home.html')

@app.route('/ai-utility')
def ai_utility():
    # Redirect regular users to their simplified dashboard
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if not (hasattr(current_user, 'is_admin') and current_user.is_admin):
            return redirect(url_for('user_dashboard'))
    return render_template('ai_utility.html')

# Platform selector route
@app.route('/platform/<platform>')
def select_platform(platform):
    """Route for different platform pages"""
    # Redirect regular users to their simplified dashboard
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if not (hasattr(current_user, 'is_admin') and current_user.is_admin):
            return redirect(url_for('user_dashboard'))
    
    platform_templates = {
        'datasafe': 'datasafe_home.html',
        'language': 'language_ai.html', 
        'contentflow': 'content_flow.html',
        'justice': 'justice_ai.html',
        'guardpass': 'guardpass.html',
        'crimecast': 'crimecast.html',
        'training': 'training_lab.html',
        'store': 'ai_store.html'
    }
    
    template = platform_templates.get(platform)
    if template:
        return render_template(template, active_section=platform)
    else:
        # Fallback to a generic platform page or 404
        return render_template('platform_not_found.html', platform=platform), 404

@app.route('/ai-store')
def ai_store():
    """AI Store page"""
    return render_template('ai_store.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # In a real application, you would process the form data here
        # For example, save to database or send email to admin
        name = request.form.get('name')
        email = request.form.get('email')
        feedback_type = request.form.get('feedbackType')
        subject = request.form.get('subject')
        message = request.form.get('message')
        followup = 'followup' in request.form
        
        # Process the feedback (e.g., save to database, send email)
        # ...
        
        # For AJAX requests, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True})
        
        # For regular form submissions, redirect with a flash message
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))
        
    # For GET requests, just render the template
    return render_template('feedback.html')

@app.route('/extract_facts', methods=['POST'])
@login_required
def extract_facts():
    chat_id = request.json.get('chat_id')
    
    # Get all chats for this user to build complete conversation history
    all_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.asc()).all()
    
    print(f"[EXTRACT_FACTS] Found {len(all_chats)} chats for user {current_user.id}")
    
    if not all_chats:
        return jsonify({'success': False, 'error': 'No chats found for this user'}), 404
    
    # Build conversation history from all chats
    all_messages = []
    for chat in all_chats:
        # Load messages for this chat
        chat_with_messages = Chat.query.options(joinedload(Chat.messages)).filter_by(id=chat.id).first()
        if chat_with_messages and chat_with_messages.messages:
            all_messages.extend(chat_with_messages.messages)
            print(f"[EXTRACT_FACTS] Chat {chat.id}: {len(chat_with_messages.messages)} messages")
    
    print(f"[EXTRACT_FACTS] Total messages: {len(all_messages)}")
    
    if not all_messages:
        return jsonify({'success': False, 'error': 'No messages found in any chats'}), 400
    
    # If we have too many messages, create a summary of older conversations and keep recent ones detailed
    if len(all_messages) > 100:
        # Keep the most recent 30 messages detailed
        recent_messages = all_messages[-30:]
        older_messages = all_messages[:-30]
        
        # Create a summary of older conversations
        older_text = '\n'.join([f"{m.role}: {m.content}" for m in older_messages])
        
        # First, create a summary of older conversations
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst. Create a concise summary of the key business information discussed."},
                    {"role": "user", "content": f"Summarize the key business information from this conversation history:\n\n{older_text}"}
                ],
                max_tokens=1000
            )
            older_summary = summary_response.choices[0].message.content
        except Exception as e:
            # If summarization fails, just use the last 50 messages
            recent_messages = all_messages[-50:]
            older_summary = ""
        
        # Combine summary with recent detailed messages
        recent_text = '\n'.join([f"{m.role}: {m.content}" for m in recent_messages])
        chat_text = f"Previous conversations summary:\n{older_summary}\n\nRecent detailed conversation:\n{recent_text}"
    else:
        # If we have fewer messages, use them all
        chat_text = '\n'.join([f"{m.role}: {m.content}" for m in all_messages])
    prompt = (
        "Extract the most important facts about this company from the following conversation history. "
        "Focus on business name, mission, goals, challenges, products/services, audience, and any other relevant details. "
        "Return the facts as a clear, structured fact sheet.\n\n" + chat_text
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert business analyst."}, {"role": "user", "content": prompt}]
        )
        fact_sheet = response.choices[0].message.content
        
        # Store the fact sheet in the most recent chat
        latest_chat = all_chats[-1]
        latest_chat.fact_sheet = fact_sheet
        db.session.commit()
        
        return jsonify({'success': True, 'fact_sheet': fact_sheet})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/develop_strategies', methods=['POST'])
@login_required
def develop_strategies():
    # Get all chats for this user to build complete conversation history
    all_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.asc()).all()
    
    if not all_chats:
        return jsonify({'success': False, 'error': 'No chats found for this user'}), 404
    
    # Check if we have a fact sheet from the most recent chat
    latest_chat = all_chats[-1]
    if not latest_chat.fact_sheet:
        return jsonify({'success': False, 'error': 'Please extract company information first before developing strategies'}), 400
    
    # Build conversation history for context
    all_messages = []
    for chat in all_chats:
        # Load messages for this chat
        chat_with_messages = Chat.query.options(joinedload(Chat.messages)).filter_by(id=chat.id).first()
        if chat_with_messages and chat_with_messages.messages:
            all_messages.extend(chat_with_messages.messages)
    
    # If we have too many messages, create a summary of older conversations and keep recent ones detailed
    if len(all_messages) > 80:
        # Keep the most recent 20 messages detailed
        recent_messages = all_messages[-20:]
        older_messages = all_messages[:-20]
        
        # Create a summary of older conversations
        older_text = '\n'.join([f"{m.role}: {m.content}" for m in older_messages])
        
        # First, create a summary of older conversations
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert business strategist. Create a concise summary of the key business context discussed."},
                    {"role": "user", "content": f"Summarize the key business context from this conversation history:\n\n{older_text}"}
                ],
                max_tokens=800
            )
            older_summary = summary_response.choices[0].message.content
        except Exception as e:
            # If summarization fails, just use the last 30 messages
            recent_messages = all_messages[-30:]
            older_summary = ""
        
        # Combine summary with recent detailed messages
        recent_text = '\n'.join([f"{m.role}: {m.content}" for m in recent_messages])
        conversation_context = f"Previous conversations summary:\n{older_summary}\n\nRecent detailed conversation:\n{recent_text}"
    else:
        # If we have fewer messages, use them all
        conversation_context = '\n'.join([f"{m.role}: {m.content}" for m in all_messages])
    
    prompt = (
        "Given the following company fact sheet and recent conversation context, develop a set of actionable strategies "
        "to help the business grow, improve, or solve its challenges. Be specific and practical.\n\n"
        f"Fact Sheet:\n{latest_chat.fact_sheet}\n\n"
        f"Recent Conversation Context:\n{conversation_context}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert business strategist."}, {"role": "user", "content": prompt}]
        )
        strategies = response.choices[0].message.content
        latest_chat.strategies = strategies
        db.session.commit()
        return jsonify({'success': True, 'strategies': strategies})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_company_info', methods=['POST'])
@login_required
def save_company_info():
    """Save company information to database"""
    try:
        data = request.json
        content = data.get('content')
        
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'}), 400
        
        # Get the user's latest chat to save the company info
        latest_chat = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).first()
        
        if latest_chat:
            latest_chat.fact_sheet = content
            db.session.commit()
            return jsonify({'success': True, 'message': 'Company information saved to database'})
        else:
            return jsonify({'success': False, 'error': 'No chat found to save to'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_strategies', methods=['POST'])
@login_required
def save_strategies():
    """Save AI strategies to database"""
    try:
        data = request.json
        content = data.get('content')
        
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'}), 400
        
        # Get the user's latest chat to save the strategies
        latest_chat = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).first()
        
        if latest_chat:
            latest_chat.strategies = content
            db.session.commit()
            return jsonify({'success': True, 'message': 'AI strategies saved to database'})
        else:
            return jsonify({'success': False, 'error': 'No chat found to save to'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit-feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Handle user feedback submission"""
    try:
        data = request.json
        
        # Validate required fields - handle both 'type' and 'feedbackType' for compatibility
        feedback_type = data.get('type') or data.get('feedbackType')
        subject = data.get('subject')
        message = data.get('message')
        
        if not feedback_type:
            return jsonify({'success': False, 'error': 'Missing required field: type/feedbackType'}), 400
        if not subject:
            return jsonify({'success': False, 'error': 'Missing required field: subject'}), 400
        if not message:
            return jsonify({'success': False, 'error': 'Missing required field: message'}), 400
        
        # Create new feedback record
        new_feedback = Feedback(
            user_id=current_user.id,
            username=current_user.username,
            feedback_type=feedback_type,
            subject=subject,
            message=message,
            allow_followup=data.get('followup', False)
        )
        
        db.session.add(new_feedback)
        db.session.commit()
        
        # Log the feedback for immediate visibility
        print(f"📢 NEW FEEDBACK from {current_user.username}:")
        print(f"   Type: {feedback_type}")
        print(f"   Subject: {subject}")
        print(f"   Message: {message}")
        print(f"   Follow-up OK: {data.get('followup', False)}")
        print(f"   Timestamp: {new_feedback.created_at}")
        print("-" * 50)
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'feedback_id': new_feedback.id
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to submit feedback: {str(e)}'}), 500

@app.route('/browse_website', methods=['POST'])
@login_required
def api_browse_website():
    """API endpoint to browse a website"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Browse the website
        result = browse_website(url)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/training/upload-pdf', methods=['POST'])
@login_required
@admin_required
def upload_training_pdf():
    """Upload PDF files for training data"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'File must be a PDF'}), 400
        
        # Create training data directory if it doesn't exist
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        pdf_dir = os.path.join(basedir, 'training', 'training_data', 'pdfs')
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Save the PDF file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(pdf_dir, filename)
        file.save(filepath)
        
        # Extract text from PDF
        try:
            import PyPDF2
            text_content = ""
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            # Save extracted text
            text_filename = filename.replace('.pdf', '.txt')
            text_filepath = os.path.join(pdf_dir, text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as text_file:
                text_file.write(text_content)
            
            return jsonify({
                'success': True,
                'message': f'PDF uploaded and processed successfully: {filename}',
                'filename': filename,
                'text_filename': text_filename,
                'pages': len(pdf_reader.pages),
                'text_length': len(text_content)
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to extract text from PDF: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/training/upload-website', methods=['POST'])
@login_required
@admin_required
def upload_training_website():
    """Add website URLs for training data"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        description = data.get('description', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Browse the website to get content
        result = browse_website(url)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch website: {result["error"]}'
            }), 400
        
        # Create training data directory if it doesn't exist
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        website_dir = os.path.join(basedir, 'training', 'training_data', 'websites')
        os.makedirs(website_dir, exist_ok=True)
        
        # Save website content
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{url.replace('://', '_').replace('/', '_').replace('.', '_')}.json"
        filepath = os.path.join(website_dir, filename)
        
        website_data = {
            'url': url,
            'title': result['title'],
            'content': result['content'],
            'description': description,
            'uploaded_at': datetime.now().isoformat(),
            'uploaded_by': current_user.username
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(website_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': f'Website content saved successfully: {result["title"]}',
            'filename': filename,
            'title': result['title'],
            'content_length': len(result['content'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/training/list-uploads', methods=['GET'])
@login_required
@admin_required
def list_training_uploads():
    """List all uploaded training data"""
    try:
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        training_dir = os.path.join(basedir, 'training', 'training_data')
        
        uploads = {
            'pdfs': [],
            'websites': []
        }
        
        # List PDFs
        pdf_dir = os.path.join(training_dir, 'pdfs')
        if os.path.exists(pdf_dir):
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(pdf_dir, filename)
                    stat = os.stat(filepath)
                    uploads['pdfs'].append({
                        'filename': filename,
                        'size': stat.st_size,
                        'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # List websites
        website_dir = os.path.join(training_dir, 'websites')
        if os.path.exists(website_dir):
            for filename in os.listdir(website_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(website_dir, filename)
                    stat = os.stat(filepath)
                    uploads['websites'].append({
                        'filename': filename,
                        'size': stat.st_size,
                        'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({
            'success': True,
            'uploads': uploads
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/training/connect-notion', methods=['POST'])
@login_required
@admin_required
def connect_notion():
    """Connect to Notion and import pages for training"""
    try:
        data = request.json
        notion_token = data.get('notion_token', '').strip()
        database_id = data.get('database_id', '').strip()
        
        if not notion_token or not database_id:
            return jsonify({'success': False, 'error': 'Notion token and database ID are required'}), 400
        
        # Test Notion connection
        try:
            from notion_client import Client
            notion = Client(auth=notion_token)
            
            # Test connection by querying the database
            response = notion.databases.query(database_id=database_id)
            
            if not response.get('results'):
                return jsonify({'success': False, 'error': 'No pages found in the specified database'}), 400
            
            # Import pages for training
            imported_pages = []
            for page in response['results']:
                try:
                    # Get page content
                    page_id = page['id']
                    page_content = notion.pages.retrieve(page_id=page_id)
                    
                    # Get page blocks (content)
                    blocks = notion.blocks.children.list(block_id=page_id)
                    
                    # Extract text content
                    text_content = ""
                    if page.get('properties', {}).get('title', {}).get('title'):
                        text_content += page['properties']['title']['title'][0]['plain_text'] + "\n\n"
                    
                    for block in blocks['results']:
                        if block['type'] == 'paragraph' and block['paragraph']['rich_text']:
                            for text in block['paragraph']['rich_text']:
                                text_content += text['plain_text'] + "\n"
                        elif block['type'] == 'heading_1' and block['heading_1']['rich_text']:
                            for text in block['heading_1']['rich_text']:
                                text_content += "# " + text['plain_text'] + "\n"
                        elif block['type'] == 'heading_2' and block['heading_2']['rich_text']:
                            for text in block['heading_2']['rich_text']:
                                text_content += "## " + text['plain_text'] + "\n"
                        elif block['type'] == 'bulleted_list_item' and block['bulleted_list_item']['rich_text']:
                            for text in block['bulleted_list_item']['rich_text']:
                                text_content += "• " + text['plain_text'] + "\n"
                        elif block['type'] == 'numbered_list_item' and block['numbered_list_item']['rich_text']:
                            for text in block['numbered_list_item']['rich_text']:
                                text_content += "1. " + text['plain_text'] + "\n"
                    
                    # Save to training data
                    import os
                    basedir = os.path.abspath(os.path.dirname(__file__))
                    notion_dir = os.path.join(basedir, 'training', 'training_data', 'notion')
                    os.makedirs(notion_dir, exist_ok=True)
                    
                    # Create filename from page title
                    page_title = page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'untitled')
                    safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_title[:50]}.json"
                    filepath = os.path.join(notion_dir, filename)
                    
                    notion_data = {
                        'page_id': page_id,
                        'title': page_title,
                        'content': text_content,
                        'url': page.get('url', ''),
                        'imported_at': datetime.now().isoformat(),
                        'imported_by': current_user.username,
                        'notion_database_id': database_id
                    }
                    
                    import json
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(notion_data, f, indent=2, ensure_ascii=False)
                    
                    imported_pages.append({
                        'title': page_title,
                        'filename': filename,
                        'content_length': len(text_content)
                    })
                    
                except Exception as e:
                    print(f"Error importing page {page.get('id', 'unknown')}: {str(e)}")
                    continue
            
            return jsonify({
                'success': True,
                'message': f'Successfully imported {len(imported_pages)} pages from Notion',
                'imported_pages': imported_pages
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to connect to Notion: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/training/real-collect-data', methods=['POST'])
@login_required
@admin_required
def real_collect_training_data():
    """Collect real training data from all sources"""
    try:
        import os
        import json
        from datetime import datetime
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        training_dir = os.path.join(basedir, 'training', 'training_data')
        
        # Ensure training directories exist
        for subdir in ['conversations', 'pdfs', 'research', 'feedback', 'notion', 'websites', 'processed']:
            os.makedirs(os.path.join(training_dir, subdir), exist_ok=True)
        
        stats = {
            'conversations': 0,
            'pdfs': 0,
            'research': 0,
            'feedback': 0,
            'notion_pages': 0,
            'websites': 0,
            'total_examples': 0
        }
        
        # 1. Collect real conversations from database
        try:
            try:
                from .models import Chat, Message, User
            except ImportError:
                from models import Chat, Message, User
            conversations = Chat.query.all()
            stats['conversations'] = len(conversations)

            # Save conversations to training data
            conversations_file = os.path.join(training_dir, 'conversations', 'all_conversations.json')
            conversations_data = []

            for chat in conversations:
                messages = (
                    Message.query
                    .filter_by(chat_id=chat.id)
                    .order_by(Message.created_at)
                    .all()
                )
                if messages:
                    conversation = {
                        'chat_id': chat.id,
                        'user_id': chat.user_id,
                        'title': chat.title,
                        'created_at': chat.created_at.isoformat() if chat.created_at else None,
                        'messages': [{'role': msg.role, 'content': msg.content} for msg in messages],
                        'fact_sheet': chat.fact_sheet,
                        'strategies': chat.strategies
                    }
                    conversations_data.append(conversation)

            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error collecting conversations: {e}")
        
        # 2. Count existing PDFs
        pdf_dir = os.path.join(training_dir, 'pdfs')
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            stats['pdfs'] = len(pdf_files)
        
        # 3. Count existing research papers
        research_dir = os.path.join(training_dir, 'research')
        if os.path.exists(research_dir):
            research_files = [f for f in os.listdir(research_dir) if f.endswith('.json')]
            stats['research'] = len(research_files)
        
        # 4. Count feedback entries
        feedback_dir = os.path.join(training_dir, 'feedback')
        if os.path.exists(feedback_dir):
            feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
            stats['feedback'] = len(feedback_files)
        
        # 5. Count Notion pages
        notion_dir = os.path.join(training_dir, 'notion')
        if os.path.exists(notion_dir):
            notion_files = [f for f in os.listdir(notion_dir) if f.endswith('.json')]
            stats['notion_pages'] = len(notion_files)
        
        # 6. Count websites
        website_dir = os.path.join(training_dir, 'websites')
        if os.path.exists(website_dir):
            website_files = [f for f in os.listdir(website_dir) if f.endswith('.json')]
            stats['websites'] = len(website_files)
        
        # Calculate total examples
        stats['total_examples'] = (
            stats['conversations'] + 
            stats['pdfs'] + 
            stats['research'] + 
            stats['feedback'] + 
            stats['notion_pages'] + 
            stats['websites']
        )
        
        # Update dataset stats
        dataset_stats = {
            'total_examples': stats['total_examples'],
            'total_tokens': stats['total_examples'] * 100,  # Rough estimate
            'sources': {
                'user_conversations': stats['conversations'],
                'pdf_documents': stats['pdfs'],
                'research_papers': stats['research'],
                'feedback_entries': stats['feedback'],
                'notion_pages': stats['notion_pages'],
                'websites': stats['websites']
            },
            'collected_at': datetime.now().isoformat(),
            'collected_by': current_user.username
        }
        
        stats_file = os.path.join(training_dir, 'processed', 'dataset_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': f'Real data collection completed. Found {stats["total_examples"]} total examples.',
            'stats': stats,
            'dataset_stats': dataset_stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for load balancers and monitoring"""
    try:
        # Basic health checks
        from training.model_manager import get_model_manager
        
        manager = get_model_manager()
        model_info = manager.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_info.get('model_loaded', False),
            'version': '1.0.0'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/admin/notion')
@login_required
@admin_required
def admin_notion():
    """Admin page for Notion integration management"""
    notion_integration = NotionIntegration.query.first()
    return render_template('admin/notion.html', notion_integration=notion_integration)

@app.route('/admin/notion/configure', methods=['POST'])
@login_required
@admin_required
def configure_notion():
    """Configure Notion integration"""
    try:
        notion_token = request.form.get('notion_token')
        workspace_id = request.form.get('workspace_id')
        database_id = request.form.get('database_id')
        
        if not notion_token or not workspace_id:
            flash('Notion token and workspace ID are required', 'error')
            return redirect(url_for('admin_notion'))
        
        # Check if integration already exists
        notion_integration = NotionIntegration.query.first()
        if notion_integration:
            notion_integration.notion_token = notion_token
            notion_integration.workspace_id = workspace_id
            notion_integration.database_id = database_id
            notion_integration.updated_at = datetime.utcnow()
        else:
            notion_integration = NotionIntegration(
                notion_token=notion_token,
                workspace_id=workspace_id,
                database_id=database_id
            )
            db.session.add(notion_integration)
        
        db.session.commit()
        flash('Notion integration configured successfully!', 'success')
        
    except Exception as e:
        flash(f'Error configuring Notion: {str(e)}', 'error')
    
    return redirect(url_for('admin_notion'))

@app.route('/admin/notion/test')
@login_required
@admin_required
def test_notion_connection():
    """Test Notion API connection"""
    try:
        notion_integration = NotionIntegration.query.first()
        if not notion_integration:
            return jsonify({'success': False, 'error': 'No Notion integration configured'})
        
        # Test the connection by making a simple API call
        headers = {
            'Authorization': f'Bearer {notion_integration.notion_token}',
            'Notion-Version': '2022-06-28',
            'Content-Type': 'application/json'
        }
        
        # Test by getting user info
        response = requests.get('https://api.notion.com/v1/users/me', headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            return jsonify({
                'success': True, 
                'message': f'Connected successfully! User: {user_data.get("name", "Unknown")}',
                'user_data': user_data
            })
        else:
            return jsonify({
                'success': False, 
                'error': f'API Error: {response.status_code} - {response.text}'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/notion/sync-chat/<int:chat_id>')
@login_required
@admin_required
def sync_chat_to_notion(chat_id):
    """Sync a specific chat to Notion"""
    try:
        # Get the chat
        chat = Chat.query.get_or_404(chat_id)
        notion_integration = NotionIntegration.query.first()
        
        if not notion_integration:
            return jsonify({'success': False, 'error': 'No Notion integration configured'})
        
        # Initialize Notion client
        notion = NotionClient(auth=notion_integration.notion_token)
        
        # Create a new page in Notion
        page_data = {
            "parent": {"type": "page_id", "page_id": notion_integration.workspace_id},
            "properties": {
                "title": {
                    "title": [
                        {
                            "text": {
                                "content": f"DataSafe Chat #{chat.id} - {chat.created_at.strftime('%Y-%m-%d')}"
                            }
                        }
                    ]
                },
                "Type": {
                    "select": {
                        "name": "AI Consultation"
                    }
                },
                "Date": {
                    "date": {
                        "start": chat.created_at.isoformat()
                    }
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": "Conversation Summary"
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"Chat ID: {chat.id}\nUser: {chat.user.username}\nDate: {chat.created_at.strftime('%Y-%m-%d %H:%M')}\nMessages: {len(chat.messages)}"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        # Add conversation content
        for message in chat.messages:
            role = "User" if message.role == "user" else "AI Assistant"
            page_data["children"].append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": role
                            }
                        }
                    ]
                }
            })
            
            page_data["children"].append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": message.content[:2000]  # Limit content length
                            }
                        }
                    ]
                }
            })
        
        # Create the page in Notion
        response = notion.pages.create(**page_data)
        
        return jsonify({
            'success': True,
            'message': f'Chat synced to Notion successfully! Page ID: {response["id"]}',
            'notion_page_id': response["id"]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/company-info')
@login_required
def company_info():
    """Display company information page"""
    # Get the most recent chat for this user
    latest_chat = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).first()
    
    if latest_chat and latest_chat.fact_sheet:
        company_data = latest_chat.fact_sheet
    else:
        company_data = "No company information available yet. Start a conversation with Highlander to extract company details."
    
    return render_template('company_info.html', company_data=company_data)

@app.route('/ai-strategies')
@login_required
def ai_strategies():
    """Display AI strategies page"""
    # Get the most recent chat for this user
    latest_chat = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).first()
    
    if latest_chat and latest_chat.strategies:
        strategies_data = latest_chat.strategies
    else:
        strategies_data = "No AI strategies available yet. Start a conversation with Highlander to develop AI strategies for your business."
    
    return render_template('ai_strategies.html', strategies_data=strategies_data)

@app.route('/find-news', methods=['POST'])
@login_required
def find_news():
    """Find relevant news based on user's chat history"""
    try:
        data = request.json or {}
        force_refresh = data.get('force_refresh', False)
        
        # Check if user already has recent news (within last 2 hours, reduced from 24)
        from datetime import timedelta
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)
        existing_news = News.query.filter_by(user_id=current_user.id).filter(News.created_at >= two_hours_ago).all()
        
        # Only use cache if not forcing refresh and we have recent news
        if existing_news and not force_refresh:
            # Return existing news from database
            news_data = [article.to_dict() for article in existing_news[:3]]
            return jsonify({'success': True, 'news': news_data, 'cached': True})
        
        # If forcing refresh, clear old news first
        if force_refresh:
            print(f"🔄 Force refresh requested - clearing old news for user {current_user.id}")
            # Delete old news articles for this user
            News.query.filter_by(user_id=current_user.id).delete()
            db.session.commit()
            print(f"🗑️ Cleared old news articles")
        
        # Get all chats for this user to build complete conversation history
        all_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
        
        if not all_chats:
            return jsonify({'success': False, 'error': 'No conversation history found. Please chat with Highlander first.'})
        
        # Build conversation history
        all_messages = []
        for chat in all_chats:
            for message in chat.messages:
                all_messages.append(f"{message.role}: {message.content}")
        
        conversation_text = "\n".join(all_messages[-50:])  # Last 50 messages for context
        
        # Use OpenAI to analyze the conversation and extract relevant topics
        if not client:
            return jsonify({'success': False, 'error': 'OpenAI client not available'})
        
        # Create a prompt to extract relevant topics from the conversation
        analysis_prompt = f"""
        Based on the following conversation between a user and an AI assistant about their media business, 
        extract 3-5 key topics, industries, or themes that would be relevant for finding news articles.
        
        Focus on:
        - Industry sectors mentioned (media, journalism, technology, etc.)
        - Specific companies or organizations discussed
        - Geographic regions or markets mentioned
        - Technology trends or challenges discussed
        - Business challenges or opportunities mentioned
        
        Conversation:
        {conversation_text}
        
        Return only a JSON array of relevant search terms, like:
        ["media industry", "journalism technology", "AI in newsrooms", "digital transformation"]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        # Parse the response to get search terms
        try:
            search_terms = json.loads(response.choices[0].message.content)
        except:
            # Fallback search terms if parsing fails
            search_terms = ["media industry", "journalism", "AI technology"]
        
        # Use NewsAPI to find relevant articles
        news_api_key = "a5e5898731c74bfe97bae546ef04dea6"  # Your NewsAPI key
        
        print(f"🔍 Fetching fresh news with search terms: {search_terms}")
        
        # Fetch real news using NewsAPI
        articles = []
        for term in search_terms[:3]:  # Use top 3 search terms
            try:
                print(f"📡 Fetching news for term: '{term}'")
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': term,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 5,
                    'apiKey': news_api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                print(f"📊 NewsAPI response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('articles'):
                        articles.extend(data['articles'][:2])  # Get top 2 articles per term
                        print(f"✅ Found {len(data['articles'][:2])} articles for '{term}'")
                    else:
                        print(f"⚠️ No articles found for term '{term}'")
                else:
                    print(f"❌ NewsAPI error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Error fetching news for term '{term}': {e}")
                continue
        
        # Remove duplicates and limit to top 3
        unique_articles = []
        seen_urls = set()
        for article in articles:
            if article.get('url') and article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
                if len(unique_articles) >= 3:
                    break
        
        # If we don't have enough articles, add some fallback ones
        while len(unique_articles) < 3:
            unique_articles.append({
                "title": "Media Industry Insights: Latest Trends and Developments",
                "description": "Stay updated with the latest developments in the media industry, from technological innovations to business strategies.",
                "url": "https://example.com/media-insights",
                "publishedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": {"name": "Media Weekly"}
            })
        
        # Save articles to database
        search_terms_str = json.dumps(search_terms)
        for article in unique_articles[:3]:
            try:
                published_at = None
                if article.get('publishedAt'):
                    try:
                        published_at = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    except:
                        published_at = datetime.utcnow()
                
                news_article = News(
                    user_id=current_user.id,
                    title=article.get('title', ''),
                    description=article.get('description', ''),
                    url=article.get('url', ''),
                    source_name=article.get('source', {}).get('name', ''),
                    published_at=published_at,
                    search_terms=search_terms_str
                )
                db.session.add(news_article)
            except Exception as e:
                print(f"Error saving news article: {e}")
                continue
        
        db.session.commit()
        
        # Return the saved articles
        news_data = []
        for article in unique_articles[:3]:
            news_data.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source_name': article.get('source', {}).get('name', ''),
                'published_at': article.get('publishedAt', ''),
                'id': len(news_data) + 1  # Temporary ID for frontend
            })
        print(f"🎯 Returning {len(news_data)} fresh news articles")
        return jsonify({'success': True, 'news': news_data, 'cached': False})
        
    except Exception as e:
        print(f"Error in find_news: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/today-news')
@login_required
def today_news():
    """Display today's relevant news page"""
    # Get user's recent news (within last 24 hours)
    from datetime import timedelta
    yesterday = datetime.utcnow() - timedelta(hours=24)
    existing_news = News.query.filter_by(user_id=current_user.id).filter(News.created_at >= yesterday).order_by(News.created_at.desc()).limit(3).all()
    
    return render_template('today_news.html', existing_news=existing_news)

@app.route('/save-strategy', methods=['POST'])
@login_required
def save_strategy():
    """Save a strategy to the database"""
    try:
        data = request.get_json()
        title = data.get('title', 'AI Strategy')
        content = data.get('content', '')
        category = data.get('category', 'general')
        priority = data.get('priority', 'medium')
        notes = data.get('notes', '')
        
        if not content.strip():
            return jsonify({'success': False, 'error': 'Strategy content cannot be empty'})
        
        # Create new saved strategy
        strategy = SavedStrategy(
            user_id=current_user.id,
            title=title,
            content=content,
            category=category,
            priority=priority,
            notes=notes
        )
        
        db.session.add(strategy)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Strategy saved successfully!',
            'strategy_id': strategy.id
        })
        
    except Exception as e:
        print(f"Error saving strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate-new-strategy', methods=['POST'])
@login_required
def generate_new_strategy():
    """Generate a new AI strategy based on user's chat history"""
    try:
        # Get all chats for this user to build complete conversation history
        all_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
        
        if not all_chats:
            return jsonify({'success': False, 'error': 'No conversation history found. Please chat with Highlander first.'})
        
        # Build conversation history
        all_messages = []
        for chat in all_chats:
            for message in chat.messages:
                all_messages.append(f"{message.role}: {message.content}")
        
        conversation_text = "\n".join(all_messages[-50:])  # Last 50 messages for context
        
        # Use OpenAI to generate new AI strategies
        if not client:
            return jsonify({'success': False, 'error': 'OpenAI client not available'})
        
        # Create a prompt to generate new AI strategies
        strategy_prompt = f"""
        Based on the following conversation between a user and an AI assistant about their media business, 
        generate a comprehensive, actionable AI strategy that would help their business grow and improve.
        
        The strategy should be:
        - Specific and actionable
        - Relevant to their business context
        - Focused on practical AI implementation
        - Include clear steps and recommendations
        - Address their specific challenges and goals
        
        Conversation context:
        {conversation_text}
        
        Generate a detailed AI strategy with:
        1. Strategy overview and objectives
        2. Specific AI tools and technologies to implement
        3. Step-by-step implementation plan
        4. Expected outcomes and benefits
        5. Potential challenges and mitigation strategies
        
        Make it comprehensive and immediately actionable.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": strategy_prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        
        new_strategy = response.choices[0].message.content
        
        return jsonify({
            'success': True, 
            'strategy': new_strategy,
            'message': 'New AI strategy generated successfully!'
        })
        
    except Exception as e:
        print(f"Error generating new strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get-saved-strategies')
@login_required
def get_saved_strategies():
    """Get all saved strategies for the current user"""
    try:
        strategies = SavedStrategy.query.filter_by(user_id=current_user.id).order_by(SavedStrategy.created_at.desc()).all()
        return jsonify({
            'success': True,
            'strategies': [strategy.to_dict() for strategy in strategies]
        })
    except Exception as e:
        print(f"Error getting saved strategies: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save-news', methods=['POST'])
@login_required
def save_news():
    """Save a news article to the database"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        description = data.get('description', '')
        url = data.get('url', '')
        source_name = data.get('source_name', '')
        published_at = data.get('published_at', '')
        notes = data.get('notes', '')
        
        if not title or not url:
            return jsonify({'success': False, 'error': 'Title and URL are required'})
        
        # Parse published_at if provided
        parsed_date = None
        if published_at:
            try:
                parsed_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            except:
                parsed_date = datetime.utcnow()
        
        # Check if news article already exists for this user
        existing_news = SavedNews.query.filter_by(
            user_id=current_user.id, 
            url=url
        ).first()
        
        if existing_news:
            return jsonify({'success': False, 'error': 'This article is already saved'})
        
        # Create new saved news
        news_article = SavedNews(
            user_id=current_user.id,
            title=title,
            description=description,
            url=url,
            source_name=source_name,
            published_at=parsed_date,
            notes=notes
        )
        
        db.session.add(news_article)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': 'News article saved successfully!',
            'news_id': news_article.id
        })
        
    except Exception as e:
        print(f"Error saving news: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get-saved-news')
@login_required
def get_saved_news():
    """Get all saved news articles for the current user"""
    try:
        news_articles = SavedNews.query.filter_by(user_id=current_user.id).order_by(SavedNews.created_at.desc()).all()
        return jsonify({
            'success': True,
            'news': [article.to_dict() for article in news_articles]
        })
    except Exception as e:
        print(f"Error getting saved news: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/crawl', methods=['POST'])
@login_required
def crawl_strategies():
    """Trigger strategy crawling and generation"""
    try:
        crawler = StrategiesCrawler()
        strategies = crawler.run_full_crawl()
        
        # Save to database
        crawler.save_strategies_to_database(strategies, db.session)
        
        return jsonify({
            'success': True,
            'message': f'Successfully crawled and generated {len(strategies)} strategies',
            'strategies_count': len(strategies)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategies', methods=['GET'])
@login_required
def get_strategies():
    """Get all strategies with optional filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        category = request.args.get('category', '')
        source = request.args.get('source', '')
        
        query = StrategyEntry.query.filter_by(is_active=True)
        
        if category:
            query = query.filter(StrategyEntry.category == category)
        if source:
            query = query.filter(StrategyEntry.source == source)
        
        strategies = query.order_by(StrategyEntry.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        strategies_data = []
        for strategy in strategies.items:
            strategies_data.append({
                'id': strategy.id,
                'title': strategy.title,
                'description': strategy.description,
                'category': strategy.category,
                'source': strategy.source,
                'url': strategy.url,
                'use_cases': json.loads(strategy.use_cases) if strategy.use_cases else [],
                'code_examples': json.loads(strategy.code_examples) if strategy.code_examples else [],
                'implementation_steps': json.loads(strategy.implementation_steps) if strategy.implementation_steps else [],
                'ai_insights': strategy.ai_insights,
                'created_at': strategy.created_at.isoformat() if strategy.created_at else None
            })
        
        return jsonify({
            'success': True,
            'strategies': strategies_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': strategies.total,
                'pages': strategies.pages,
                'has_next': strategies.has_next,
                'has_prev': strategies.has_prev
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategies/<int:strategy_id>', methods=['GET'])
@login_required
def get_strategy(strategy_id):
    """Get a specific strategy by ID"""
    try:
        strategy = StrategyEntry.query.get_or_404(strategy_id)
        
        return jsonify({
            'success': True,
            'strategy': {
                'id': strategy.id,
                'title': strategy.title,
                'description': strategy.description,
                'category': strategy.category,
                'source': strategy.source,
                'url': strategy.url,
                'use_cases': json.loads(strategy.use_cases) if strategy.use_cases else [],
                'code_examples': json.loads(strategy.code_examples) if strategy.code_examples else [],
                'implementation_steps': json.loads(strategy.implementation_steps) if strategy.implementation_steps else [],
                'ai_insights': strategy.ai_insights,
                'created_at': strategy.created_at.isoformat() if strategy.created_at else None
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategies/categories', methods=['GET'])
@login_required
def get_strategy_categories():
    """Get all available strategy categories"""
    try:
        categories = db.session.query(StrategyEntry.category).distinct().all()
        return jsonify({
            'success': True,
            'categories': [cat[0] for cat in categories if cat[0]]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategies/sources', methods=['GET'])
@login_required
def get_strategy_sources():
    """Get all available strategy sources"""
    try:
        sources = db.session.query(StrategyEntry.source).distinct().all()
        return jsonify({
            'success': True,
            'sources': [src[0] for src in sources if src[0]]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategies/generate', methods=['POST'])
@login_required
def generate_vision_strategies():
    """Generate new strategies based on vision and expertise"""
    try:
        crawler = StrategiesCrawler()
        strategies = crawler.generate_vision_aligned_strategies()
        
        # Save to database
        crawler.save_strategies_to_database(strategies, db.session)
        
        return jsonify({
            'success': True,
            'message': f'Successfully generated {len(strategies)} vision-aligned strategies',
            'strategies_count': len(strategies)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/strategies-dashboard')
@login_required
def strategies_dashboard():
    """Strategies dashboard page"""
    return render_template('strategies_dashboard.html')

@app.route('/admin/strategies')
@login_required
def admin_strategies():
    """Admin strategies management page"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('user_dashboard'))
    return render_template('admin/strategies.html')

# ===== CRAWLING ROUTES =====

@app.route('/admin/crawling')
@login_required
@admin_required
def admin_crawling():
    """Admin crawling management page"""
    from .models import CrawlSource, CrawledContent, CrawlJob
    from crawler_service import CrawlManager
    
    # Get crawling statistics
    crawl_manager = CrawlManager(db, openai_api_key)
    stats = crawl_manager.get_crawl_stats()
    
    # Get recent sources and jobs
    sources = CrawlSource.query.order_by(CrawlSource.created_at.desc()).limit(10).all()
    recent_jobs = CrawlJob.query.order_by(CrawlJob.created_at.desc()).limit(10).all()
    
    return render_template('admin/crawling.html', 
                         stats=stats, 
                         sources=sources, 
                         recent_jobs=recent_jobs)

@app.route('/admin/crawling/sources')
@login_required
@admin_required
def admin_crawl_sources():
    """Manage crawl sources"""
    from .models import CrawlSource
    
    sources = CrawlSource.query.order_by(CrawlSource.created_at.desc()).all()
    return render_template('admin/crawl_sources.html', sources=sources)

@app.route('/admin/crawling/sources/add', methods=['POST'])
@login_required
@admin_required
def add_crawl_source():
    """Add a new crawl source"""
    from .models import CrawlSource
    
    try:
        data = request.get_json()
        
        source = CrawlSource(
            name=data['name'],
            url=data['url'],
            source_type=data['source_type'],
            description=data.get('description', ''),
            crawl_frequency=data.get('crawl_frequency', 'daily'),
            config=data.get('config', '{}')
        )
        
        db.session.add(source)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Source "{source.name}" added successfully',
            'source_id': source.id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/crawling/sources/<int:source_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_crawl_source(source_id):
    """Toggle crawl source active status"""
    from .models import CrawlSource
    
    try:
        source = CrawlSource.query.get_or_404(source_id)
        source.is_active = not source.is_active
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Source "{source.name}" {"activated" if source.is_active else "deactivated"}',
            'is_active': source.is_active
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/crawling/sources/<int:source_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_crawl_source(source_id):
    """Delete a crawl source"""
    from .models import CrawlSource
    
    try:
        source = CrawlSource.query.get_or_404(source_id)
        source_name = source.name
        db.session.delete(source)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Source "{source_name}" deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/crawling/jobs/start', methods=['POST'])
@login_required
@admin_required
def start_crawl_job():
    """Start a new crawl job"""
    from models import CrawlSource
    from crawler_service import CrawlManager
    
    try:
        data = request.get_json()
        source_id = data.get('source_id')
        
        if source_id:
            # Start job for specific source
            source = CrawlSource.query.get_or_404(source_id)
            crawl_manager = CrawlManager(db, openai_api_key)
            job_id = crawl_manager.create_crawl_job(source_id)
            
            # Run the job in a background thread
            def run_job():
                crawl_manager.run_crawl_job(job_id)
            
            thread = threading.Thread(target=run_job)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Crawl job started for "{source.name}"',
                'job_id': job_id
            })
        else:
            # Start jobs for all active sources
            active_sources = CrawlSource.query.filter_by(is_active=True).all()
            crawl_manager = CrawlManager(db, openai_api_key)
            job_ids = []
            
            for source in active_sources:
                job_id = crawl_manager.create_crawl_job(source.id)
                job_ids.append(job_id)
                
                # Run each job in a background thread
                def run_job(job_id=job_id):
                    crawl_manager.run_crawl_job(job_id)
                
                thread = threading.Thread(target=run_job)
                thread.daemon = True
                thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Started {len(job_ids)} crawl jobs',
                'job_ids': job_ids
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/crawling/jobs')
@login_required
@admin_required
def admin_crawl_jobs():
    """View crawl jobs"""
    from .models import CrawlJob
    
    jobs = CrawlJob.query.order_by(CrawlJob.created_at.desc()).all()
    return render_template('admin/crawl_jobs.html', jobs=jobs)

@app.route('/admin/crawling/content')
@login_required
@admin_required
def admin_crawled_content():
    """View crawled content"""
    from .models import CrawledContent
    
    # Get filter parameters
    content_type = request.args.get('type')
    source_id = request.args.get('source_id')
    processed = request.args.get('processed')
    
    query = CrawledContent.query
    
    if content_type:
        query = query.filter_by(content_type=content_type)
    if source_id:
        query = query.filter_by(source_id=source_id)
    if processed:
        query = query.filter_by(is_processed=processed == 'true')
    
    content = query.order_by(CrawledContent.created_at.desc()).limit(100).all()
    
    # Get sources for filter dropdown
    from .models import CrawlSource
    sources = CrawlSource.query.all()
    
    return render_template('admin/crawled_content.html', 
                         content=content, 
                         sources=sources,
                         filters={'type': content_type, 'source_id': source_id, 'processed': processed})

@app.route('/admin/crawling/content/<int:content_id>')
@login_required
@admin_required
def view_crawled_content(content_id):
    """View specific crawled content"""
    from .models import CrawledContent
    
    content = CrawledContent.query.get_or_404(content_id)
    return render_template('admin/crawled_content_detail.html', content=content)

@app.route('/admin/crawling/content/<int:content_id>/process', methods=['POST'])
@login_required
@admin_required
def process_crawled_content(content_id):
    """Process crawled content with AI"""
    from .models import CrawledContent
    from crawler_service import ContentCrawler
    
    try:
        content = CrawledContent.query.get_or_404(content_id)
        
        # Re-analyze content with AI
        crawler = ContentCrawler(openai_api_key)
        analysis = crawler.analyze_content(content.content)
        
        # Update content with new analysis
        content.content_type = analysis.get('content_type', 'article')
        content.tags = json.dumps(analysis.get('tags', []))
        content.summary = analysis.get('summary', '')
        content.relevance_score = analysis.get('relevance_score', 0.0)
        content.is_processed = True
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Content processed successfully',
            'analysis': analysis
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/crawling/stats')
@login_required
@admin_required
def get_crawl_stats():
    """Get crawling statistics"""
    from crawler_service import CrawlManager
    
    try:
        crawl_manager = CrawlManager(db, openai_api_key)
        stats = crawl_manager.get_crawl_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    sys.path.append('/path/to/your/directory')
    # Handle subdirectory deployment
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)
    app.run(host='0.0.0.0', port=8000, debug=True) 
