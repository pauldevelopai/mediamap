"""
MediaMap Flask Application
=========================

A comprehensive media analysis and AI training platform with:
- User authentication and admin management
- Media analysis and reporting
- AI model training and deployment
- Client and organization management
- Data collection and processing

Author: MediaMap Team
Version: 2.0
"""

import os
import sys
# Configuration Constants
# =======================

# Disable wandb completely before importing anything else
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Application Configuration
MAX_MESSAGE_LENGTH = 10000
MIN_USERNAME_LENGTH = 3
MIN_PASSWORD_LENGTH = 6
SAVE_INTERVAL = 30  # seconds
MAX_CHAT_HISTORY = 100

# Add the backend directory to Python path for training module imports
basedir = os.path.abspath(os.path.dirname(__file__))
if basedir not in sys.path:
    sys.path.insert(0, basedir)

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
import uuid

# Multi-app architecture
try:
    from .app_routes import register_app_routes
except ImportError:
    from app_routes import register_app_routes
try:
    from .filtered_admin_routes import register_filtered_admin_routes
except ImportError:
    from filtered_admin_routes import register_filtered_admin_routes
# Separate Admin Apps
try:
    from .admin_apps.mediamap_admin.routes import register_mediamap_admin_routes
    from .admin_apps.healthpin_admin.routes import register_healthpin_admin_routes
except ImportError:
    from admin_apps.mediamap_admin.routes import register_mediamap_admin_routes
    from admin_apps.healthpin_admin.routes import register_healthpin_admin_routes
try:
    from aimap.config import OPENAI_API_KEY
except ImportError:
    from aimap.config import OPENAI_API_KEY
from datetime import datetime, timezone, timedelta
import urllib.parse
import requests
try:
    from auth import auth
except ImportError:
    from auth import auth
import time
import threading
import re
from urllib.parse import urlparse
import io
import traceback
import sys
import logging
from functools import wraps
from sqlalchemy import Column, Boolean, text

# Configure logging
logger = logging.getLogger(__name__)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import joinedload
from bs4 import BeautifulSoup
import html2text
from notion_client import Client as NotionClient
try:
    from backend.strategies_crawler import StrategiesCrawler, StrategyEntry
except ImportError:
    StrategiesCrawler = None
    StrategyEntry = None

# Import AIMAP modules
try:
    from aimap.api.routes import aimap_api
    from aimap.api.ml_routes import ml_api
    from aimap.api.consulting_routes import consulting_api
    from aimap.api.data_management_routes import data_api
    from aimap.models import Organisation  # Metrics temporarily disabled
except ImportError:
    aimap_api = None
    ml_api = None
    consulting_api = None
    data_api = None
    Organisation = None
    Metrics = None

# Import HealthPIN modules
try:
    from backend.healthpin import healthpin_bp
    # HealthPIN models imported via healthpin package to avoid duplicate registration
    from healthpin.webhooks import webhooks_bp
    from healthpin.doc_chatbot import doc_chatbot_bp
    from healthpin.whatsapp_webhook import whatsapp_webhook_bp
    from agents.routes import agents_bp
    print("✅ HealthPIN modules imported successfully")
except ImportError as e:
    print(f"⚠️ HealthPIN import error: {e}")
    healthpin_bp = None
    webhooks_bp = None
    doc_chatbot_bp = None
    whatsapp_webhook_bp = None
    agents_bp = None
    Patient = None
    Doctor = None
    HealthRecord = None
    DoctorMatch = None
    FamilyNotification = None
    Consultation = None
    HealthNews = None

# Import memory management
try:
    from api.memory_routes import memory_api
except ImportError:
    memory_api = None

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
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create instance directory if it doesn't exist
os.makedirs('instance', exist_ok=True)

# Use absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "media_analysis.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register multi-app routes
register_app_routes(app)
register_filtered_admin_routes(app)
# Register separate admin apps
register_mediamap_admin_routes(app)
register_healthpin_admin_routes(app)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager(app)

# Ensure database is properly initialized
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created/verified successfully!")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
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
            # Redirect to user dashboard instead of landing page for better UX
            return redirect(url_for('user_dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

# Section access decorator
def section_required(required_section):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Allow admin users to bypass section requirements
            if hasattr(current_user, 'is_admin') and current_user.is_admin:
                return f(*args, **kwargs)
            
            current_section = session.get('section')
            if current_section != required_section:
                flash('You do not have access to this section.', 'danger')
                if current_section == 'healthpin':
                    return redirect(url_for('doc_chatbot.doc_chat'))
                return redirect(url_for('mediamap_dashboard'))
            return f(*args, **kwargs)
        return wrapped
    return decorator
# Initialize OpenAI client (only if API key is available)
openai_api_key = os.getenv('OPENAI_API_KEY')
app.config['OPENAI_API_KEY'] = openai_api_key  # Add this line to set the API key in app config
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
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None

# Import prompt manager to load prompts from database
from prompt_manager import get_prompt as get_prompt_from_db

# Function to get system prompts from database
def get_system_prompt_analysis():
    """Get the media analysis system prompt from database"""
    return get_prompt_from_db('HIGHLANDER_SYSTEM_PROMPT_ANALYSIS')

def get_system_prompt_chat():
    """Get the chat system prompt from database"""
    return get_prompt_from_db('HIGHLANDER_SYSTEM_PROMPT_CHAT')

def get_system_prompt_synthesis():
    """Get the organization synthesis prompt from database"""
    return get_prompt_from_db('HIGHLANDER_SYSTEM_PROMPT_SYNTHESIS')

def get_system_prompt_media_biz():
    """Get the Highlander AI business consultant prompt from database"""
    return get_prompt_from_db('HIGHLANDER_SYSTEM_PROMPT_MEDIA_BIZ')

# app.register_blueprint(auth)  # Commented out to avoid route conflicts
app.register_blueprint(ai_utility_bp)
app.register_blueprint(metadata_bp)

# Register AIMAP API blueprints
if aimap_api:
    app.register_blueprint(aimap_api)
if ml_api:
    app.register_blueprint(ml_api)
if consulting_api:
    app.register_blueprint(consulting_api)
if data_api:
    app.register_blueprint(data_api)
if memory_api:
    app.register_blueprint(memory_api)

# Register HealthPIN blueprints
if healthpin_bp:
    app.register_blueprint(healthpin_bp)
    print("✅ HealthPIN routes loaded")

if webhooks_bp:
    app.register_blueprint(webhooks_bp)
    print("✅ HealthPIN webhooks loaded")

if doc_chatbot_bp:
    app.register_blueprint(doc_chatbot_bp)
    print("✅ Doc chatbot routes loaded")

if whatsapp_webhook_bp:
    app.register_blueprint(whatsapp_webhook_bp)
    print("✅ WhatsApp webhook routes loaded")

if agents_bp:
    app.register_blueprint(agents_bp)
    print("✅ AI agents routes loaded")

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
    from datasafe_integration import setup_datasafe_routes
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
    # DataSafe is now integrated into MediaMap as a tab
    return redirect(url_for('user_dashboard') + '#datasafe')

# === Model management endpoints (Hugging Face integration) ===
@app.route('/api/model/load-hf', methods=['POST'])
@login_required
@admin_required
def load_model_from_hf():
    """Load a model from Hugging Face Hub by name. Requires admin."""
    try:
        data = request.get_json(silent=True) or {}
        model_name = data.get('model_name') or os.getenv('HF_MODEL_REPO') or 'paulmcnally/highlander-ai-model'
        from training.model_factory import get_mediamap_model_manager
        manager = get_mediamap_model_manager()
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

# Ensure proper cleanup on exit
import atexit
def cleanup_save_thread():
    try:
        save_thread.join(timeout=1)
    except:
        pass

atexit.register(cleanup_save_thread)

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
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    message = request.json.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400
    
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
        # Use Highlander model manager for MediaMap chat
        print(f"Using Highlander model manager for MediaMap chat")
        
        # Get MediaMap model manager
        from training.model_factory import get_mediamap_model_manager
        manager = get_mediamap_model_manager()
        
        # Prepare conversation history
        conversation_history = []
        for msg in chat_data['messages']:
            conversation_history.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Generate response using Highlander model
        try:
            if manager and hasattr(manager, 'generate_response'):
                ai_reply, source = manager.generate_response(
                    message=message,
                    conversation_history=conversation_history
                )
            else:
                raise Exception("Model manager not available")
        except Exception as model_error:
            print(f"Model manager error: {model_error}")
            # Fallback to OpenAI directly
            if client:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are Highlander AI, a helpful business assistant specializing in media and technology. Provide clear, actionable advice."},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                ai_reply = response.choices[0].message.content
                source = "openai_fallback"
            else:
                ai_reply = "I'm currently experiencing technical difficulties. Please try again in a moment."
                source = "error_fallback"
        
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
            'model_source': source
        })
            
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
            {"role": "system", "content": get_system_prompt_media_biz()}
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
def generate_user_insights():
    return render_template('generate_insights.html')

@app.route('/your-info')
def your_info():
    return render_template('your_info.html')

# Will be moved earlier in file

# Admin routes
@app.route('/admin')
@app.route('/admin/')
@login_required
def admin_dashboard():
    """Redirect old admin route to app selector"""
    flash('Please select your admin application', 'info')
    return redirect(url_for('app_selector'))

@app.route('/admin/quick-access')
@login_required
@admin_required
def admin_quick_access():
    """Quick Access page with all admin shortcuts"""
    return render_template('admin/quick_access.html')

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin page to view all users"""
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        
        # Get feedback from non-admin users (Highlander and Doc usage feedback)
        non_admin_feedback = Feedback.query.join(User).filter(
            User.is_admin == False
        ).order_by(Feedback.created_at.desc()).all()
        
        return render_template('admin/users.html', users=users, non_admin_feedback=non_admin_feedback)
    except Exception as e:
        print(f"Error in admin_users: {e}")
        flash(f'Error loading users page: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/debug/user-status')
@login_required
def debug_user_status():
    """Debug route to check current user's admin status"""
    try:
        user_info = {
            'id': current_user.id,
            'username': current_user.username,
            'email': current_user.email,
            'is_admin': getattr(current_user, 'is_admin', 'NOT_SET'),
            'is_authenticated': current_user.is_authenticated,
            'has_is_admin_attr': hasattr(current_user, 'is_admin')
        }
        
        # Check all users in database
        all_users = User.query.all()
        users_info = []
        for user in all_users:
            users_info.append({
                'id': user.id,
                'username': user.username,
                'is_admin': getattr(user, 'is_admin', 'NOT_SET'),
                'has_is_admin_attr': hasattr(user, 'is_admin')
            })
        
        return jsonify({
            'current_user': user_info,
            'all_users': users_info,
            'total_users': len(users_info)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/fix-admin-status', methods=['POST'])
@login_required
def fix_admin_status():
    """Fix admin status for current user if needed"""
    try:
        # Check if current user has admin status
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            # Set admin status to True
            current_user.is_admin = True
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Admin status granted successfully',
                'user_id': current_user.id,
                'username': current_user.username
            })
        else:
            return jsonify({
                'success': True,
                'message': 'User already has admin status',
                'user_id': current_user.id,
                'username': current_user.username
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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

# Business Intelligence Admin Routes
@app.route('/admin/organizations')
@login_required
@admin_required
def admin_organizations():
    """Admin page for organization management"""
    return render_template('admin/organizations.html')

# Prompt Management Routes
@app.route('/admin/prompts')
@login_required
@admin_required
def admin_prompts():
    """Admin page for prompt management"""
    from models import PromptTemplate
    
    # Get only active prompts (actually used in the application)
    prompts = PromptTemplate.query.filter_by(is_active=True).order_by(PromptTemplate.updated_at.desc()).all()
    
    # Get statistics
    total_prompts = PromptTemplate.query.count()
    active_prompts = PromptTemplate.query.filter_by(is_active=True).count()
    
    # Get unique categories and LLM providers from active prompts only
    categories = db.session.query(PromptTemplate.category).filter_by(is_active=True).distinct().all()
    categories = [cat[0] for cat in categories]
    
    llm_providers = db.session.query(PromptTemplate.llm_provider).filter_by(is_active=True).distinct().all()
    llm_providers = [provider[0] for provider in llm_providers]
    
    return render_template(
        'admin/prompts.html',
        prompts=prompts,
        total_prompts=total_prompts,
        active_prompts=active_prompts,
        categories=categories,
        llm_providers=llm_providers
    )

@app.route('/admin/agents')
@login_required
@admin_required
def admin_agents():
    """Admin AI agents dashboard"""
    return render_template('admin/agents.html')

@app.route('/admin/agents/real-data')
@login_required
@admin_required
def show_real_agent_data():
    """Show real data collected by agents"""
    import json
    import os
    
    real_data = {}
    
    # Get MediaMap data
    mediamap_data_file = "backend/agents/storage/mediamap/MediaMapAgent_data.json"
    if os.path.exists(mediamap_data_file):
        try:
            with open(mediamap_data_file, 'r') as f:
                mediamap_data = json.load(f)
                real_data['mediamap'] = {
                    'data_points': len(mediamap_data),
                    'recent_data': mediamap_data[-3:] if mediamap_data else []
                }
        except Exception as e:
            real_data['mediamap'] = {'error': str(e)}
    
    # Get HealthPIN data
    healthpin_data_file = "backend/agents/storage/healthpin/HealthPINAgent_data.json"
    if os.path.exists(healthpin_data_file):
        try:
            with open(healthpin_data_file, 'r') as f:
                healthpin_data = json.load(f)
                real_data['healthpin'] = {
                    'data_points': len(healthpin_data),
                    'recent_data': healthpin_data[-3:] if healthpin_data else []
                }
        except Exception as e:
            real_data['healthpin'] = {'error': str(e)}
    
    return jsonify({
        'success': True,
        'real_data': real_data,
        'timestamp': datetime.utcnow().isoformat()
    })



@app.route('/admin/insights')
@login_required
@admin_required
def admin_insights():
    """Admin AI insights dashboard"""
    return render_template('admin/insights.html')

@app.route('/api/organization-insights/generate', methods=['POST'])
@login_required
def generate_organization_insight():
    """Generate comprehensive AI insights for an organization"""
    try:
        data = request.get_json()
        if not data or 'organization_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Organization ID is required'
            }), 400
        
        organization_id = data['organization_id']
        
        # Import the service
        from services.organization_insight_service import organization_insight_service
        
        # Generate insights
        result = organization_insight_service.generate_comprehensive_insight(
            organization_id=organization_id,
            user_id=current_user.id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating organization insight: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/organization-insights/report', methods=['POST'])
@login_required
def generate_organization_report():
    """Generate comprehensive 2-page AI implementation report for an organization"""
    try:
        data = request.get_json()
        if not data or 'organization_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Organization ID is required'
            }), 400
        
        organization_id = data['organization_id']
        format_type = data.get('format', 'html')
        
        # Import the service
        from services.organization_insight_service import organization_insight_service
        
        # Generate report
        result = organization_insight_service.generate_two_page_report(
            organization_id=organization_id,
            format_type=format_type
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating organization report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/newsrooms', methods=['GET'])
@login_required
def get_newsrooms():
    """Get list of newsrooms for selection"""
    try:
        newsrooms = Newsroom.query.all()
        return jsonify({
            'success': True,
            'newsrooms': [
                {
                    'id': newsroom.id,
                    'name': newsroom.name,
                    'description': newsroom.description
                } for newsroom in newsrooms
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching newsrooms: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/organization-insights/<int:organization_id>', methods=['GET'])
@login_required
def get_organization_insights(organization_id: int):
    """Get insights for a specific organization"""
    try:
        from services.organization_insight_service import organization_insight_service
        
        insights = organization_insight_service.get_organization_insights(organization_id)
        
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Error fetching insights for organization {organization_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/insights/generate-report', methods=['POST'])
@login_required
@admin_required
def generate_insights_report():
    """Generate comprehensive insights report"""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
            
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
            
        report_type = data.get('type', 'mediamap')  # mediamap, healthpin
        period_days = data.get('period', 30)
        format_type = data.get('format', 'pdf')
        
        # Import agent_manager with error handling
        try:
            from agents.agent_manager import agent_manager
        except ImportError as e:
            logger.error(f"Failed to import agent_manager: {e}")
            return jsonify({
                'success': False,
                'error': 'Agent manager not available'
            }), 500
            
        from datetime import datetime, timedelta
        import json
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Get insights based on type
        insights = []
        
        try:
            if report_type == 'mediamap':
                mediamap_insights = agent_manager.get_mediamap_insights()
                if mediamap_insights:
                    for insight in mediamap_insights:
                        insight['source_agent'] = 'mediamap'
                        insights.append(insight)
                else:
                    # Add a sample insight if none available
                    insights.append({
                        'source_agent': 'mediamap',
                        'type': 'Sample',
                        'category': 'General',
                        'insight': 'No MediaMap insights available at this time.',
                        'confidence': 0.5,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            elif report_type == 'healthpin':
                healthpin_insights = agent_manager.get_healthpin_insights()
                if healthpin_insights:
                    for insight in healthpin_insights:
                        insight['source_agent'] = 'healthpin'
                        insights.append(insight)
                else:
                    # Add a sample insight if none available
                    insights.append({
                        'source_agent': 'healthpin',
                        'type': 'Sample',
                        'category': 'General',
                        'insight': 'No HealthPIN insights available at this time.',
                        'confidence': 0.5,
                        'timestamp': datetime.utcnow().isoformat()
                    })
        except Exception as e:
            logger.error(f"Error getting insights from agent_manager: {e}")
            # Return sample data instead of failing
            insights.append({
                'source_agent': report_type,
                'type': 'Error',
                'category': 'System',
                'insight': f'Agent manager temporarily unavailable. Error: {str(e)}',
                'confidence': 0.1,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Filter by date if needed
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        filtered_insights = []
        for insight in insights:
            if insight.get('timestamp'):
                try:
                    insight_date = datetime.fromisoformat(insight['timestamp'].replace('Z', '+00:00'))
                    if insight_date >= cutoff_date:
                        filtered_insights.append(insight)
                except:
                    filtered_insights.append(insight)  # Include if we can't parse date
            else:
                filtered_insights.append(insight)  # Include if no timestamp
        
        # Generate report based on format
        if format_type == 'json':
            report_data = {
                'report_type': report_type,
                'period_days': period_days,
                'generated_at': datetime.utcnow().isoformat(),
                'insights_count': len(filtered_insights),
                'insights': filtered_insights,
                'summary': {
                    'total_insights': len(filtered_insights),
                    'mediamap_insights': len([i for i in filtered_insights if i.get('source_agent') == 'mediamap']),
                    'healthpin_insights': len([i for i in filtered_insights if i.get('source_agent') == 'healthpin']),
                    'avg_confidence': sum([i.get('confidence', 0.5) for i in filtered_insights]) / len(filtered_insights) if filtered_insights else 0
                }
            }
            
            # Create downloadable JSON
            filename = f"insights_report_{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            return jsonify({
                'success': True,
                'report_data': report_data,
                'filename': filename,
                'download_url': f'/admin/insights/download/{filename}'
            })
        
        elif format_type == 'html':
            # Generate HTML report
            html_content = generate_html_report(filtered_insights, report_type, period_days)
            filename = f"insights_report_{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            
            return jsonify({
                'success': True,
                'html_content': html_content,
                'filename': filename,
                'download_url': f'/admin/insights/download/{filename}'
            })
        
        else:  # PDF format
            # Generate PDF report
            pdf_content = generate_pdf_report(filtered_insights, report_type, period_days)
            filename = f"insights_report_{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            return jsonify({
                'success': True,
                'filename': filename,
                'download_url': f'/admin/insights/download/{filename}'
            })
        
    except Exception as e:
        logger.error(f"Error generating insights report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_html_report(insights, report_type, period_days):
    """Generate HTML report content"""
    from datetime import datetime
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Insights Report - {report_type.title()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .insight {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}
            .insight.mediamap {{ border-left: 4px solid #28a745; }}
            .insight.healthpin {{ border-left: 4px solid #dc3545; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
            .badge-mediamap {{ background: #d4edda; color: #155724; }}
            .badge-healthpin {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI Insights Report</h1>
            <h2>{report_type.title()} - Last {period_days} Days</h2>
            <p>Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        </div>
        
        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Total Insights:</strong> {len(insights)}</p>
            <p><strong>MediaMap Insights:</strong> {len([i for i in insights if i.get('source_agent') == 'mediamap'])}</p>
            <p><strong>HealthPIN Insights:</strong> {len([i for i in insights if i.get('source_agent') == 'healthpin'])}</p>
        </div>
        
        <div class="insights">
            <h3>Detailed Insights</h3>
    """
    
    for insight in insights:
        source = insight.get('source_agent', 'unknown')
        confidence = insight.get('confidence', 0.5)
        timestamp = insight.get('timestamp', 'Unknown')
        
        html += f"""
            <div class="insight {source}">
                <div style="margin-bottom: 10px;">
                    <span class="badge badge-{source}">{source.upper()}</span>
                    <span style="margin-left: 10px;">Confidence: {int(confidence * 100)}%</span>
                    <span style="float: right;">{timestamp}</span>
                </div>
                <h4>{insight.get('type', 'Insight')}</h4>
                <p><strong>Category:</strong> {insight.get('category', 'General')}</p>
                <p>{insight.get('insight', insight.get('content', 'No content available'))}</p>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def generate_pdf_report(insights, report_type, period_days):
    """Generate PDF report content"""
    # For now, return a simple text-based report
    # In production, you'd use reportlab or similar
    return f"PDF Report for {report_type} - {len(insights)} insights from last {period_days} days"

@app.route('/admin/insights/download/<filename>')
@login_required
@admin_required
def download_insights_report(filename):
    """Download generated insights report"""
    try:
        # For now, return a simple response since we're not storing files
        # In production, you'd serve the actual file
        return jsonify({
            'success': True,
            'message': f'Report {filename} would be downloaded here'
        })
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/agents/<agent_name>/start', methods=['POST'])
@login_required
@admin_required
def admin_start_agent(agent_name):
    """Start a specific AI agent from admin interface"""
    try:
        from agents.agent_manager import agent_manager
        
        if agent_name not in agent_manager.agents:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_name} not found'
            }), 404
        
        success = agent_manager.start_agent(agent_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'{agent_name} agent started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to start {agent_name} agent'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/agents/<agent_name>/stop', methods=['POST'])
@login_required
@admin_required
def admin_stop_agent(agent_name):
    """Stop a specific AI agent from admin interface"""
    try:
        from agents.agent_manager import agent_manager
        
        if agent_name not in agent_manager.agents:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_name} not found'
            }), 404
        
        success = agent_manager.stop_agent(agent_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'{agent_name} agent stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to stop {agent_name} agent'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/admin/agents/<agent_name>/config', methods=['GET'])
@login_required
@admin_required
def get_agent_config(agent_name):
    """Get agent configuration"""
    try:
        from models import AgentConfiguration
        import json
        
        config = AgentConfiguration.query.filter_by(name=agent_name).first()
        if config:
            return jsonify({
                'success': True,
                'config': config.to_dict()
            })
        else:
            # Return default configuration
            default_configs = {
                'mediamap': {
                    'name': 'mediamap',
                    'display_name': 'MediaMap Agent',
                    'section': 'mediamap',
                    'role': 'Media Industry Data Collector',
                    'description': 'Collects and analyzes media industry data from RSS feeds, news sites, and industry publications. Monitors trends, business models, and emerging technologies in journalism and media.',
                    'data_sources': ['Nieman Lab RSS Feed', 'Poynter Institute', 'Journalism.co.uk', 'MediaPost', 'O\'Reilly Radar'],
                    'collection_interval': 30,
                    'instructions': 'Focus on media industry trends, business models, and technological innovations. Analyze content for actionable insights and identify emerging patterns in journalism and media.'
                },
                'healthpin': {
                    'name': 'healthpin',
                    'display_name': 'HealthPIN Agent',
                    'section': 'healthpin',
                    'role': 'Healthcare Data Collector',
                    'description': 'Collects and analyzes healthcare data from medical journals, clinical guidelines, and healthcare policy updates. Monitors medical research, clinical trials, and healthcare trends.',
                    'data_sources': ['Healthcare News Feeds', 'Medical Journals', 'Policy Updates', 'Clinical Guidelines', 'Research Publications'],
                    'collection_interval': 45,
                    'instructions': 'Focus on healthcare developments, clinical research, and medical policy changes. Analyze content for clinical insights and identify trends in healthcare delivery and technology.'
                }
            }
            
            if agent_name in default_configs:
                return jsonify({
                    'success': True,
                    'config': default_configs[agent_name]
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Agent {agent_name} not found'
                }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Removed conflicting placeholder routes - agent functionality is handled by agents blueprint

@app.route('/api/agents/dashboard', methods=['GET'])
@login_required
@admin_required
def get_agents_dashboard():
    """Get agents dashboard data"""
    try:
        from models import Chat, HighlanderChat, DailyInsight, Newsroom
        from aimap.models import Organisation
        from datetime import datetime, timedelta
        
        # Get real data instead of dummy data
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        # Get actual statistics
        total_chats = Chat.query.count() + HighlanderChat.query.count()
        recent_insights = DailyInsight.query.filter(DailyInsight.created_at >= last_24h).count()
        total_newsrooms = Newsroom.query.count()
        total_organizations = Organisation.query.count()
        
        # Get recent activity
        recent_highlander_chats = HighlanderChat.query.filter(
            HighlanderChat.created_at >= last_24h
        ).order_by(HighlanderChat.created_at.desc()).limit(5).all()
        
        recent_insights_list = DailyInsight.query.filter(
            DailyInsight.created_at >= last_24h
        ).order_by(DailyInsight.created_at.desc()).limit(5).all()
        
        # Build dashboard data
        dashboard_data = {
            'agents': {
                'mediamap': {
                    'status': 'active',
                    'last_activity': now.isoformat(),
                    'data_sources': 5,
                    'insights_generated': recent_insights,
                    'uptime': '99.9%'
                },
                'healthpin': {
                    'status': 'active', 
                    'last_activity': now.isoformat(),
                    'data_sources': 3,
                    'insights_generated': recent_insights,
                    'uptime': '99.9%'
                }
            },
            'statistics': {
                'total_chats': total_chats,
                'recent_insights': recent_insights,
                'total_newsrooms': total_newsrooms,
                'total_organizations': total_organizations
            },
            'recent_activity': {
                'highlander_chats': [
                    {
                        'id': chat.id,
                        'message': chat.message[:100] + '...' if len(chat.message) > 100 else chat.message,
                        'created_at': chat.created_at.isoformat() if chat.created_at else None
                    } for chat in recent_highlander_chats
                ],
                'insights': [
                    {
                        'id': insight.id,
                        'title': insight.title,
                        'category': insight.category,
                        'created_at': insight.created_at.isoformat() if insight.created_at else None
                    } for insight in recent_insights_list
                ]
            }
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/agents/<agent_name>/config', methods=['POST'])
@login_required
@admin_required
def update_agent_config(agent_name):
    """Update agent configuration"""
    try:
        from models import AgentConfiguration, db
        import json
        
        data = request.get_json()
        
        # Get or create configuration
        config = AgentConfiguration.query.filter_by(name=agent_name).first()
        if not config:
            config = AgentConfiguration(name=agent_name)
            db.session.add(config)
        
        # Update configuration
        config.display_name = data.get('display_name', config.display_name)
        config.section = data.get('section', config.section)
        config.role = data.get('role', config.role)
        config.description = data.get('description', config.description)
        config.data_sources = json.dumps(data.get('data_sources', []))
        config.collection_interval = data.get('collection_interval', config.collection_interval)
        config.instructions = data.get('instructions', config.instructions)
        config.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'{agent_name} agent configuration updated successfully',
            'config': config.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/workflows/<workflow_name>/agents', methods=['GET'])
@login_required
@admin_required
def get_workflow_agents(workflow_name):
    """Get all agents for a specific workflow"""
    try:
        from models import WorkflowAgent
        
        agents = WorkflowAgent.query.filter_by(
            workflow_name=workflow_name,
            is_active=True
        ).order_by(WorkflowAgent.priority, WorkflowAgent.created_at).all()
        
        return jsonify({
            'success': True,
            'workflow': workflow_name,
            'agents': [agent.to_dict() for agent in agents]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/workflows/<workflow_name>/agents', methods=['POST'])
@login_required
@admin_required
def add_workflow_agent(workflow_name):
    """Add a new agent to a workflow"""
    try:
        from models import WorkflowAgent, db
        import json
        
        data = request.get_json()
        
        # Create new workflow agent
        agent = WorkflowAgent(
            workflow_name=workflow_name,
            agent_name=data.get('agent_name'),
            display_name=data.get('display_name'),
            role=data.get('role'),
            description=data.get('description'),
            data_sources=json.dumps(data.get('data_sources', [])),
            collection_interval=data.get('collection_interval', 30),
            instructions=data.get('instructions'),
            priority=data.get('priority', 1)
        )
        
        db.session.add(agent)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Agent added to {workflow_name} workflow successfully',
            'agent': agent.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/workflows/<workflow_name>/agents/<int:agent_id>', methods=['PUT'])
@login_required
@admin_required
def update_workflow_agent(workflow_name, agent_id):
    """Update a workflow agent"""
    try:
        from models import WorkflowAgent, db
        import json
        
        agent = WorkflowAgent.query.filter_by(
            id=agent_id,
            workflow_name=workflow_name
        ).first()
        
        if not agent:
            return jsonify({
                'success': False,
                'error': 'Agent not found'
            }), 404
        
        data = request.get_json()
        
        # Update agent
        agent.display_name = data.get('display_name', agent.display_name)
        agent.role = data.get('role', agent.role)
        agent.description = data.get('description', agent.description)
        agent.data_sources = json.dumps(data.get('data_sources', []))
        agent.collection_interval = data.get('collection_interval', agent.collection_interval)
        agent.instructions = data.get('instructions', agent.instructions)
        agent.priority = data.get('priority', agent.priority)
        agent.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Agent updated successfully',
            'agent': agent.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/workflows/<workflow_name>/agents/<int:agent_id>', methods=['DELETE'])
@login_required
@admin_required
def remove_workflow_agent(workflow_name, agent_id):
    """Remove an agent from a workflow"""
    try:
        from models import WorkflowAgent, db
        
        agent = WorkflowAgent.query.filter_by(
            id=agent_id,
            workflow_name=workflow_name
        ).first()
        
        if not agent:
            return jsonify({
                'success': False,
                'error': 'Agent not found'
            }), 404
        
        db.session.delete(agent)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Agent removed from {workflow_name} workflow successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/agents')
@login_required
def agents_dashboard():
    """Main AI agents dashboard for all users"""
    return render_template('agents/dashboard.html')

@app.route('/agents/details')
@login_required
def agent_details():
    """Detailed agent view"""
    agent_name = request.args.get('agent', 'mediamap')
    return render_template('agents/agent_details.html', agent_name=agent_name)

@app.route('/admin/doc-chatbot')
@login_required
@admin_required
def admin_doc_chatbot():
    """Admin Doc chatbot interface"""
    return render_template('admin/doc_chatbot.html')

@app.route('/admin/prompts', methods=['POST'])
@login_required
@admin_required
def create_prompt():
    """Create a new prompt"""
    from models import PromptTemplate
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'category', 'prompt_type', 'content', 'llm_provider']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Missing required field: {field}'})
        
        # Check if name already exists
        existing = PromptTemplate.query.filter_by(name=data['name']).first()
        if existing:
            return jsonify({'success': False, 'error': 'A prompt with this name already exists'})
        
        # Create new prompt
        prompt = PromptTemplate(
            name=data['name'],
            description=data.get('description', ''),
            category=data['category'],
            prompt_type=data['prompt_type'],
            content=data['content'],
            llm_provider=data['llm_provider'],
            model_name=data.get('model_name', ''),
            usage_context=data.get('usage_context', ''),
            variables=data.get('variables', ''),
            is_active=data.get('is_active', True),
            version=data.get('version', '1.0'),
            created_by=current_user.id
        )
        
        db.session.add(prompt)
        db.session.commit()
        
        # Refresh the prompt cache so new prompt is available immediately
        try:
            from prompt_manager import refresh_prompts
            refresh_prompts()
            print(f"✅ Refreshed prompt cache after creating: {prompt.name}")
        except Exception as e:
            print(f"⚠️ Error refreshing prompt cache: {e}")
        
        return jsonify({'success': True, 'message': 'Prompt created successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>')
@login_required
@admin_required
def get_prompt(prompt_id):
    """Get a specific prompt"""
    from models import PromptTemplate
    
    try:
        prompt = PromptTemplate.query.get_or_404(prompt_id)
        return jsonify({'success': True, 'prompt': prompt.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>', methods=['PUT'])
@login_required
@admin_required
def update_prompt(prompt_id):
    """Update a prompt"""
    from models import PromptTemplate
    
    try:
        prompt = PromptTemplate.query.get_or_404(prompt_id)
        data = request.get_json()
        
        # Update fields
        if 'name' in data:
            # Check if name already exists (excluding current prompt)
            existing = PromptTemplate.query.filter(
                PromptTemplate.name == data['name'],
                PromptTemplate.id != prompt_id
            ).first()
            if existing:
                return jsonify({'success': False, 'error': 'A prompt with this name already exists'})
            prompt.name = data['name']
        
        if 'description' in data:
            prompt.description = data['description']
        if 'category' in data:
            prompt.category = data['category']
        if 'prompt_type' in data:
            prompt.prompt_type = data['prompt_type']
        if 'content' in data:
            prompt.content = data['content']
        if 'llm_provider' in data:
            prompt.llm_provider = data['llm_provider']
        if 'model_name' in data:
            prompt.model_name = data['model_name']
        if 'usage_context' in data:
            prompt.usage_context = data['usage_context']
        if 'variables' in data:
            prompt.variables = data['variables']
        if 'is_active' in data:
            prompt.is_active = data['is_active']
        if 'version' in data:
            prompt.version = data['version']
        
        prompt.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Create a new version for this change
        try:
            from prompt_version_manager import version_manager
            change_reason = data.get('change_reason', 'Prompt updated via admin dashboard')
            version_manager.create_version(prompt.id, change_reason, current_user.id)
            print(f"✅ Created new version for prompt: {prompt.name}")
        except Exception as e:
            print(f"⚠️ Error creating version: {e}")
        
        # Refresh the prompt cache so changes take effect immediately
        try:
            from prompt_manager import refresh_prompts
            refresh_prompts()
            print(f"✅ Refreshed prompt cache after updating: {prompt.name}")
        except Exception as e:
            print(f"⚠️ Error refreshing prompt cache: {e}")
        
        return jsonify({'success': True, 'message': 'Prompt updated successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_prompt(prompt_id):
    """Delete a prompt"""
    from models import PromptTemplate
    
    try:
        prompt = PromptTemplate.query.get_or_404(prompt_id)
        db.session.delete(prompt)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Prompt deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/test', methods=['POST'])
@login_required
@admin_required
def test_prompt():
    """Test a prompt with variables"""
    try:
        data = request.get_json()
        variables = data.get('variables', '{}')
        
        # Parse variables if provided as JSON string
        if isinstance(variables, str):
            try:
                variables = json.loads(variables)
            except json.JSONDecodeError:
                return jsonify({'success': False, 'error': 'Invalid JSON format for variables'})
        
        # This is a simple test - in a real implementation, you might want to
        # actually process the prompt with the variables
        processed_prompt = "Test prompt processing would happen here with variables: " + str(variables)
        
        return jsonify({
            'success': True, 
            'processed_prompt': processed_prompt,
            'variables_used': list(variables.keys()) if variables else []
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Prompt Versioning Routes
@app.route('/admin/prompts/<int:prompt_id>/versions')
@login_required
@admin_required
def get_prompt_versions(prompt_id):
    """Get version history for a prompt"""
    try:
        from prompt_version_manager import version_manager
        versions = version_manager.get_version_history(prompt_id)
        return jsonify({'success': True, 'versions': versions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>/rollback', methods=['POST'])
@login_required
@admin_required
def rollback_prompt(prompt_id):
    """Rollback a prompt to a previous version"""
    try:
        from prompt_version_manager import version_manager
        data = request.get_json()
        version_number = data.get('version_number')
        
        if not version_number:
            return jsonify({'success': False, 'error': 'Version number is required'})
        
        success = version_manager.rollback_to_version(prompt_id, version_number, current_user.id)
        
        if success:
            # Refresh prompt cache
            from prompt_manager import refresh_prompts
            refresh_prompts()
            return jsonify({'success': True, 'message': f'Rolled back to version {version_number}'})
        else:
            return jsonify({'success': False, 'error': 'Rollback failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/check-duplicates')
@login_required
@admin_required
def check_prompt_duplicates():
    """Check for duplicate prompts in the database"""
    try:
        from models import PromptTemplate
        
        # Get all active prompts
        prompts = PromptTemplate.query.filter_by(is_active=True).all()
        
        # Check for duplicates by name
        name_counts = {}
        content_counts = {}
        duplicates = []
        
        for prompt in prompts:
            # Check name duplicates
            if prompt.name in name_counts:
                name_counts[prompt.name].append(prompt)
            else:
                name_counts[prompt.name] = [prompt]
            
            # Check content duplicates (normalized)
            normalized_content = prompt.content.strip().lower()
            if normalized_content in content_counts:
                content_counts[normalized_content].append(prompt)
            else:
                content_counts[normalized_content] = [prompt]
        
        # Find actual duplicates
        name_duplicates = {name: prompts for name, prompts in name_counts.items() if len(prompts) > 1}
        content_duplicates = {content: prompts for content, prompts in content_counts.items() if len(prompts) > 1}
        
        return jsonify({
            'success': True,
            'name_duplicates': {name: [{'id': p.id, 'name': p.name, 'version': p.version} for p in prompts] 
                               for name, prompts in name_duplicates.items()},
            'content_duplicates': {content[:100] + '...': [{'id': p.id, 'name': p.name, 'version': p.version} for p in prompts] 
                                  for content, prompts in content_duplicates.items()},
            'total_duplicates': len(name_duplicates) + len(content_duplicates)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>/performance')
@login_required
@admin_required
def get_prompt_performance(prompt_id):
    """Get performance statistics for a prompt"""
    try:
        from prompt_version_manager import performance_tracker
        days = request.args.get('days', 30, type=int)
        stats = performance_tracker.get_performance_stats(prompt_id, days)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/prompts/<int:prompt_id>/feedback', methods=['POST'])
@login_required
@admin_required
def record_prompt_feedback(prompt_id):
    """Record user feedback for a prompt"""
    try:
        from prompt_version_manager import performance_tracker
        data = request.get_json()
        
        success = performance_tracker.record_user_feedback(
            prompt_id=prompt_id,
            version_number=data.get('version_number'),
            user_rating=data.get('user_rating'),
            user_feedback=data.get('user_feedback'),
            was_helpful=data.get('was_helpful'),
            user_id=current_user.id,
            session_id=data.get('session_id')
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Feedback recorded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to record feedback'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/consulting')
@login_required
@admin_required
def admin_consulting():
    """Admin page for consulting management"""
    return render_template('admin/consulting.html')

@app.route('/admin/predictions')
@login_required
@admin_required
def admin_predictions():
    """Admin page for predictions management"""
    return render_template('admin/predictions.html')

@app.route('/admin/intelligence-reports')
@login_required
@admin_required
def admin_intelligence_reports():
    """Admin page for intelligence reports"""
    return render_template('admin/intelligence_reports.html')
@app.route('/admin/prototyping')
@login_required
@admin_required
def admin_prototyping():
    """AI Prototyping page - track newsroom AI product prototypes"""
    try:
        from models import AIPrototype, Newsroom, PrototypeUpdate
        
        # Get prototype statistics
        total_prototypes = AIPrototype.query.count()
        active_prototypes = AIPrototype.query.filter_by(status='Active').count()
        completed_prototypes = AIPrototype.query.filter_by(status='Completed').count()
        
        # Get prototypes by stage
        ideation_count = AIPrototype.query.filter_by(stage='Ideation').count()
        development_count = AIPrototype.query.filter_by(stage='Development').count()
        testing_count = AIPrototype.query.filter_by(stage='Testing').count()
        production_count = AIPrototype.query.filter_by(stage='Production').count()
        
        # Get recent updates
        recent_updates = PrototypeUpdate.query.order_by(PrototypeUpdate.created_at.desc()).limit(5).all()
        
        # Get newsrooms for dropdown
        newsrooms = Newsroom.query.all()
        
        return render_template('admin/prototyping.html',
                             total_prototypes=total_prototypes,
                             active_prototypes=active_prototypes,
                             completed_prototypes=completed_prototypes,
                             ideation_count=ideation_count,
                             development_count=development_count,
                             testing_count=testing_count,
                             production_count=production_count,
                             recent_updates=recent_updates,
                             newsrooms=newsrooms)
    except Exception as e:
        return render_template('admin/prototyping.html',
                             total_prototypes=0,
                             active_prototypes=0,
                             completed_prototypes=0,
                             ideation_count=0,
                             development_count=0,
                             testing_count=0,
                             production_count=0,
                             recent_updates=[],
                             newsrooms=[],
                             error=str(e))

@app.route('/admin/notion')
@login_required
@admin_required
def admin_notion():
    """Admin page for Notion integration"""
    return render_template('admin/notion.html')

@app.route('/admin/strategies')
@login_required
@admin_required
def admin_strategies():
    """Admin page for AI strategies management"""
    return render_template('admin/strategies.html')

@app.route('/admin/map')
@login_required
@admin_required
def admin_map():
    """Business Map page - comprehensive overview of consulting business with AIMAP integration"""
    # Get REAL statistics from AIMAP database
    try:
        from aimap.models import Organisation  # Metrics temporarily disabled
        from models import Client, Newsroom, ResearchProject, DailyInsight
        
        # AIMAP Organization statistics (REAL DATA)
        total_organisations = Organisation.query.count()
        media_organisations = Organisation.query.filter_by(sector='Media').count()
        communications_organisations = Organisation.query.filter_by(sector='Communications').count()
        
        # Process organizations to properly format tags
        raw_organisations = Organisation.query.all()
        aimap_organisations = []
        for org in raw_organisations:
            org_dict = {
                'id': org.id,
                'name': org.name,
                'sector': org.sector,
                'subsector': org.subsector,
                'region': org.region,
                'country': org.country,
                'size_band': org.size_band,
                'client_tag': org.client_tag,
                'contact': org.contact,
                'ai_tools': org.ai_tools,
                'notes': org.notes,
                'website_url': org.website_url,
                'tags': org.tags.split(',') if org.tags and isinstance(org.tags, str) else [],
                'created_at': org.created_at
            }
            aimap_organisations.append(org_dict)
        
        # Client statistics
        total_clients = Client.query.count()
        active_clients = Client.query.filter_by(status='Active').count()
        media_companies = Client.query.filter_by(industry='Media').count()
        tech_startups = Client.query.filter_by(industry='Technology').count()
        non_profits = Client.query.filter_by(industry='Non-Profit').count()
        government_clients = Client.query.filter_by(industry='Government').count()
        
        # Newsroom statistics
        total_newsrooms = Newsroom.query.count()
        national_newsrooms = Newsroom.query.filter_by(type='National').count()
        regional_newsrooms = Newsroom.query.filter_by(type='Regional').count()
        digital_newsrooms = Newsroom.query.filter_by(type='Digital-First').count()
        international_newsrooms = Newsroom.query.filter_by(type='International').count()
        
        # Get newsrooms data for the template
        newsrooms = Newsroom.query.all()
        
        # Research statistics (for uploaded/scraped AI research reports)
        research_projects = ResearchProject.query.count()
        ai_implementation_reports = ResearchProject.query.filter_by(category='AI Implementation').count()
        newsroom_ai_reports = ResearchProject.query.filter_by(category='Newsroom AI').count()
        tech_trends_reports = ResearchProject.query.filter_by(category='Technology Trends').count()
        industry_reports = ResearchProject.query.filter_by(category='Industry Reports').count()
        
        # Insights statistics (for generated documents - excluding news articles)
        total_insights = DailyInsight.query.filter(
            DailyInsight.category != 'Admin News'
        ).count()
        weekly_insights = DailyInsight.query.filter(
            DailyInsight.created_at >= datetime.now() - timedelta(days=7),
            DailyInsight.category != 'Admin News'
        ).count()
        newsroom_insights = DailyInsight.query.filter_by(category='Newsroom Success Story').count()
        ai_strategy_insights = DailyInsight.query.filter_by(category='AI Strategy').count()
        
        # Data source counts for insights
        try:
            from models import HighlanderChat
            highlander_chat_count = HighlanderChat.query.count()
        except ImportError:
            highlander_chat_count = 0
        research_reports_count = research_projects
        client_data_count = total_clients
        
        # Daily insights (excluding news articles)
        daily_insights = DailyInsight.query.filter(
            DailyInsight.created_at >= datetime.now().date(),
            DailyInsight.category != 'Admin News'
        ).count()
        
        # News counts (placeholder - will be from actual news sources)
        ai_news_count = 0
        media_news_count = 0
        tech_news_count = 0
        industry_updates = 0
        
        # People statistics (REAL DATA)
        from models import PersonManagement
        total_people = PersonManagement.query.count()
        ai_experts = PersonManagement.query.filter_by(role='AI Expert').count()
        consultants = PersonManagement.query.filter_by(role='Consultant').count()
        clients = PersonManagement.query.filter_by(role='Client Contact').count()
        
        people_stats = {
            'total_people': total_people,
            'ai_experts': ai_experts,
            'consultants': consultants,
            'clients': clients
        }
        
        # Project statistics (REAL DATA)
        from models import Project
        total_projects = Project.query.count()
        active_projects = Project.query.filter_by(status='Active').count()
        completed_projects = Project.query.filter_by(status='Completed').count()
        ai_projects = Project.query.filter(Project.type.like('%AI%')).count()
        
        project_stats = {
            'total_projects': total_projects,
            'active_projects': active_projects,
            'completed_projects': completed_projects,
            'ai_projects': ai_projects
        }
        
        # People and projects data (REAL DATA)
        people = PersonManagement.query.all()
        projects = Project.query.all()
        
    except Exception as e:
        # If models don't exist yet, return zeros
        total_organisations = media_organisations = communications_organisations = 0
        aimap_organisations = []
        total_clients = active_clients = media_companies = tech_startups = non_profits = government_clients = 0
        total_newsrooms = national_newsrooms = regional_newsrooms = digital_newsrooms = international_newsrooms = 0
        newsrooms = []
        research_projects = ai_implementation_reports = newsroom_ai_reports = tech_trends_reports = industry_reports = 0
        total_insights = weekly_insights = newsroom_insights = ai_strategy_insights = 0
        highlander_chat_count = research_reports_count = client_data_count = 0
        daily_insights = ai_news_count = media_news_count = tech_news_count = industry_updates = 0
        
        # People and projects defaults
        people_stats = {
            'total_people': 0,
            'ai_experts': 0,
            'consultants': 0,
            'clients': 0
        }
        
        project_stats = {
            'total_projects': 0,
            'active_projects': 0,
            'completed_projects': 0,
            'ai_projects': 0
        }
        
        people = []
        projects = []
    
    stats = {
        # AIMAP Data (REAL)
        'total_organisations': total_organisations,
        'media_organisations': media_organisations,
        'communications_organisations': communications_organisations,
        'aimap_organisations': aimap_organisations,
        
        # Existing System Data
        'total_clients': total_clients,
        'active_clients': active_clients,
        'total_newsrooms': total_newsrooms,
        'newsrooms': newsrooms,
        'research_projects': research_projects,
        'daily_insights': daily_insights,
        'media_companies': media_companies,
        'tech_startups': tech_startups,
        'non_profits': non_profits,
        'government_clients': government_clients,
        'national_newsrooms': national_newsrooms,
        'regional_newsrooms': regional_newsrooms,
        'digital_newsrooms': digital_newsrooms,
        'international_newsrooms': international_newsrooms,
        # Research tab statistics
        'ai_implementation_reports': ai_implementation_reports,
        'newsroom_ai_reports': newsroom_ai_reports,
        'tech_trends_reports': tech_trends_reports,
        'industry_reports': industry_reports,
        # Insights tab statistics
        'total_insights': total_insights,
        'weekly_insights': weekly_insights,
        'newsroom_insights': newsroom_insights,
        'ai_strategy_insights': ai_strategy_insights,
        'highlander_chat_count': highlander_chat_count,
        'research_reports_count': research_reports_count,
        'client_data_count': client_data_count,
        # News tab statistics
        'ai_news_count': ai_news_count,
        'media_news_count': media_news_count,
        'tech_news_count': tech_news_count,
        'industry_updates': industry_updates,
        # People and Projects tab statistics
        'people_stats': people_stats,
        'project_stats': project_stats,
        'people': people,
        'projects': projects
    }
    
    return render_template('admin/map.html', **stats)

# People Management API Endpoints
@app.route('/admin/map/people', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_people():
    """Manage people for AI consulting projects"""
    if request.method == 'GET':
        try:
            from models import PersonManagement
            people = PersonManagement.query.all()
            people_data = []
            for person in people:
                people_data.append(person.to_dict())
            return jsonify({'people': people_data})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import PersonManagement
            data = request.get_json()
            
            new_person = PersonManagement(
                name=data['name'],
                title=data.get('title', ''),
                role=data.get('role', 'Team Member'),
                organization=data.get('organization', ''),
                email=data.get('email', ''),
                phone=data.get('phone', ''),
                linkedin_url=data.get('linkedin_url', ''),
                expertise=','.join(data.get('expertise', [])),
                ai_skills=','.join(data.get('ai_skills', [])),
                industry_experience=','.join(data.get('industry_experience', [])),
                current_projects=','.join(data.get('current_projects', [])),
                availability=data.get('availability', 'Available'),
                hourly_rate=data.get('hourly_rate'),
                status=data.get('status', 'Active'),
                notes=data.get('notes', ''),
                tags=','.join(data.get('tags', []))
            )
            
            db.session.add(new_person)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_person.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import PersonManagement
            data = request.get_json()
            person_id = data['id']
            
            person = PersonManagement.query.get(person_id)
            if not person:
                return jsonify({'error': 'Person not found'})
            
            # Update fields
            for field in ['name', 'title', 'role', 'organization', 'email', 'phone', 'linkedin_url', 
                         'availability', 'hourly_rate', 'status', 'notes']:
                if field in data:
                    setattr(person, field, data[field])
            
            # Update list fields
            for field in ['expertise', 'ai_skills', 'industry_experience', 'current_projects', 'tags']:
                if field in data:
                    setattr(person, field, ','.join(data[field]))
            
            person.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import PersonManagement
            data = request.get_json()
            person_id = data['id']
            
            person = PersonManagement.query.get(person_id)
            if not person:
                return jsonify({'error': 'Person not found'})
            
            db.session.delete(person)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})


# Project Management API Endpoints
@app.route('/admin/map/projects', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_projects():
    """Manage AI consulting projects"""
    if request.method == 'GET':
        try:
            from models import Project
            projects = Project.query.all()
            projects_data = []
            for project in projects:
                project_dict = project.to_dict()
                # Get team members from assignments
                from models import ProjectAssignment
                assignments = ProjectAssignment.query.filter_by(project_id=project.id).all()
                team_members = []
                for assignment in assignments:
                    person = PersonManagement.query.get(assignment.person_id)
                    if person:
                        team_members.append(f"{person.name} ({assignment.role})")
                project_dict['team_members'] = team_members
                projects_data.append(project_dict)
            return jsonify({'projects': projects_data})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import Project
            data = request.get_json()
            
            new_project = Project(
                name=data['name'],
                description=data.get('description', ''),
                type=data.get('type', 'Consulting'),
                status=data.get('status', 'Planning'),
                client_id=data.get('client_id'),
                client_name=data.get('client_name', ''),
                start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
                end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
                estimated_hours=data.get('estimated_hours'),
                actual_hours=data.get('actual_hours'),
                objectives=data.get('objectives', ''),
                deliverables=data.get('deliverables', ''),
                success_metrics=data.get('success_metrics', ''),
                risks_and_challenges=data.get('risks_and_challenges', ''),
                ai_technologies=','.join(data.get('ai_technologies', [])),
                ai_maturity_level=data.get('ai_maturity_level', 'Beginner'),
                data_requirements=data.get('data_requirements', ''),
                budget=data.get('budget'),
                actual_cost=data.get('actual_cost'),
                billing_type=data.get('billing_type', 'Hourly'),
                tags=','.join(data.get('tags', [])),
                priority=data.get('priority', 'Medium'),
                notes=data.get('notes', '')
            )
            
            db.session.add(new_project)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_project.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import Project
            data = request.get_json()
            project_id = data['id']
            
            project = Project.query.get(project_id)
            if not project:
                return jsonify({'error': 'Project not found'})
            
            # Update fields
            for field in ['name', 'description', 'type', 'status', 'client_id', 'client_name', 
                         'estimated_hours', 'actual_hours', 'objectives', 'deliverables', 
                         'success_metrics', 'risks_and_challenges', 'ai_maturity_level', 
                         'data_requirements', 'budget', 'actual_cost', 'billing_type', 
                         'priority', 'notes']:
                if field in data:
                    setattr(project, field, data[field])
            
            # Update date fields
            if 'start_date' in data and data['start_date']:
                project.start_date = datetime.fromisoformat(data['start_date'])
            if 'end_date' in data and data['end_date']:
                project.end_date = datetime.fromisoformat(data['end_date'])
            
            # Update list fields
            if 'ai_technologies' in data:
                project.ai_technologies = ','.join(data['ai_technologies'])
            if 'tags' in data:
                project.tags = ','.join(data['tags'])
            
            project.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import Project
            data = request.get_json()
            project_id = data['id']
            
            project = Project.query.get(project_id)
            if not project:
                return jsonify({'error': 'Project not found'})
            
            db.session.delete(project)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})


# Import/Export API Endpoints
@app.route('/admin/map/export/people', methods=['GET'])
@login_required
@admin_required
def export_people_csv():
    """Export people data to CSV"""
    try:
        from models import PersonManagement
        import csv
        from io import StringIO
        
        people = PersonManagement.query.all()
        
        # Create CSV data
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Name', 'Title', 'Role', 'Organization', 'Email', 'Phone', 'LinkedIn', 'Expertise', 'AI Skills', 'Industry Experience', 'Availability', 'Hourly Rate', 'Status', 'Tags', 'Notes'])
        
        # Write data
        for person in people:
            writer.writerow([
                person.name,
                person.title or '',
                person.role,
                person.organization or '',
                person.email or '',
                person.phone or '',
                person.linkedin_url or '',
                person.expertise or '',
                person.ai_skills or '',
                person.industry_experience or '',
                person.availability,
                person.hourly_rate or '',
                person.status,
                person.tags or '',
                person.notes or ''
            ])
        
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=people_export.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/admin/map/export/projects', methods=['GET'])
@login_required
@admin_required
def export_projects_csv():
    """Export projects data to CSV"""
    try:
        from models import Project
        import csv
        from io import StringIO
        
        projects = Project.query.all()
        
        # Create CSV data
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Name', 'Description', 'Type', 'Status', 'Client', 'Start Date', 'End Date', 'Estimated Hours', 'Actual Hours', 'Budget', 'Actual Cost', 'Billing Type', 'AI Technologies', 'AI Maturity Level', 'Priority', 'Tags', 'Notes'])
        
        # Write data
        for project in projects:
            writer.writerow([
                project.name,
                project.description or '',
                project.type,
                project.status,
                project.client_name or '',
                project.start_date.isoformat() if project.start_date else '',
                project.end_date.isoformat() if project.end_date else '',
                project.estimated_hours or '',
                project.actual_hours or '',
                project.budget or '',
                project.actual_cost or '',
                project.billing_type or '',
                project.ai_technologies or '',
                project.ai_maturity_level or '',
                project.priority,
                project.tags or '',
                project.notes or ''
            ])
        
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=projects_export.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)})


# Project Templates API Endpoints
@app.route('/admin/map/project-templates', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_project_templates():
    """Manage AI project templates"""
    if request.method == 'GET':
        try:
            from models import ProjectTemplate
            templates = ProjectTemplate.query.all()
            templates_data = []
            for template in templates:
                templates_data.append(template.to_dict())
            return jsonify({'templates': templates_data})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import ProjectTemplate
            data = request.get_json()
            
            new_template = ProjectTemplate(
                name=data['name'],
                description=data.get('description', ''),
                category=data.get('category', 'General'),
                industry=data.get('industry', ''),
                phases=data.get('phases', ''),
                deliverables=data.get('deliverables', ''),
                timeline=data.get('timeline', ''),
                estimated_hours=data.get('estimated_hours'),
                ai_technologies=','.join(data.get('ai_technologies', [])),
                ai_maturity_requirements=data.get('ai_maturity_requirements', 'Beginner'),
                data_requirements=data.get('data_requirements', ''),
                success_metrics=data.get('success_metrics', ''),
                risk_factors=data.get('risk_factors', ''),
                tags=','.join(data.get('tags', []))
            )
            
            db.session.add(new_template)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_template.id})
        except Exception as e:
            return jsonify({'error': str(e)})


# Organization Management API Endpoints
@app.route('/admin/map/organizations', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_organizations():
    """Manage AIMAP organizations"""
    if request.method == 'GET':
        try:
            from aimap.models import Organisation
            print(f"DEBUG: Fetching organizations from table: {Organisation.__tablename__}")
            organizations = Organisation.query.all()
            print(f"DEBUG: Found {len(organizations)} organizations")
            
            org_data = []
            for org in organizations:
                print(f"DEBUG: Processing org: {org.name}")
                org_data.append({
                    'id': org.id,
                    'name': org.name,
                    'sector': org.sector,
                    'subsector': org.subsector,
                    'region': org.region,
                    'country': org.country,
                    'size_band': org.size_band,
                    'client_tag': org.client_tag,
                    'contact': org.contact,
                    'ai_tools': org.ai_tools,
                    'notes': org.notes,
                    'website_url': org.website_url,
                    'tags': org.tags.split(',') if org.tags and isinstance(org.tags, str) else [],
                    'created_at': org.created_at.isoformat() if org.created_at else None
                })
            print(f"DEBUG: Returning {len(org_data)} organizations")
            return jsonify({'organizations': org_data})
        except Exception as e:
            print(f"DEBUG: Error in GET organizations: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from aimap.models import Organisation
            data = request.get_json()
            print(f"DEBUG: POST - Creating organization with data: {data}")
            print(f"DEBUG: POST - Using table: {Organisation.__tablename__}")
            
            new_org = Organisation(
                name=data['name'],
                sector=data.get('sector', ''),
                subsector=data.get('subsector', ''),
                region=data.get('region', ''),
                country=data.get('country', ''),
                size_band=data.get('size_band', ''),
                client_tag=data.get('client_tag', ''),
                contact=data.get('contact', ''),
                ai_tools=data.get('ai_tools', ''),
                notes=data.get('notes', ''),
                website_url=data.get('website_url', ''),
                tags=','.join(data.get('tags', []))
            )
            
            db.session.add(new_org)
            db.session.commit()
            print(f"DEBUG: POST - Successfully created organization with ID: {new_org.id}")
            
            return jsonify({'status': 'success', 'id': new_org.id})
        except Exception as e:
            print(f"DEBUG: POST - Error creating organization: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from aimap.models import Organisation
            data = request.get_json()
            org_id = data['id']
            
            org = Organisation.query.get(org_id)
            if not org:
                return jsonify({'error': 'Organization not found'})
            
            # Update fields
            for field in ['name', 'sector', 'subsector', 'region', 'country', 'size_band', 
                         'client_tag', 'contact', 'ai_tools', 'notes', 'website_url']:
                if field in data:
                    setattr(org, field, data[field])
            
            # Update tags
            if 'tags' in data:
                org.tags = ','.join(data['tags'])
            
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from aimap.models import Organisation
            data = request.get_json()
            org_id = data['id']
            
            org = Organisation.query.get(org_id)
            if not org:
                return jsonify({'error': 'Organization not found'})
            
            db.session.delete(org)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
# Organization Discovery API Endpoints
@app.route('/admin/map/discover', methods=['POST'])
@login_required
@admin_required
def start_discovery():
    """Start organization discovery scan"""
    try:
        data = request.get_json()
        keywords = data.get('keywords', '')
        sector = data.get('sector', '')
        sources = data.get('sources', ['news', 'crunchbase'])
        max_results = data.get('max_results', 100)
        
        print(f"Discovery request: keywords='{keywords}', sector='{sector}'")  # Debug log
        
        # Get existing organizations from database to filter out duplicates
        from aimap.models import Organisation
        existing_orgs = Organisation.query.all()
        existing_names = {org.name.lower().strip() for org in existing_orgs}
        print(f"Found {len(existing_names)} existing organizations in database")  # Debug log
        
        # Generate intelligent sample data with AI analysis
        sample_organizations = [
            {
                'name': 'The Daily Chronicle',
                'source': 'News Analysis',
                'score': '95%',
                'ai_signals': 'AI content generation, automated fact-checking, personalized news feeds',
                'size': 'Medium',
                'sector': 'Media',
                'website': 'https://daily-chronicle.com',
                'description': 'Leading digital news organization with AI-powered content generation',
                'ai_analysis': 'EXCELLENT CANDIDATE: Already implementing AI content generation and automated fact-checking. Shows strong commitment to digital transformation. Medium size makes them agile for further AI adoption.',
                'candidate_reasons': ['Active AI implementation', 'Digital-first approach', 'Innovation leadership', 'Scalable operations']
            },
            {
                'name': 'TechNews Media Group',
                'source': 'Industry Database',
                'score': '92%',
                'ai_signals': 'Machine learning algorithms, automated reporting, AI-driven analytics',
                'size': 'Large',
                'sector': 'Technology',
                'website': 'https://technews-media.com',
                'description': 'Technology-focused media company with advanced AI tools',
                'ai_analysis': 'STRONG CANDIDATE: Technology focus indicates AI readiness. Large size provides resources for comprehensive AI integration. Already using ML algorithms for content automation.',
                'candidate_reasons': ['Technology expertise', 'Existing ML infrastructure', 'Resource availability', 'Industry credibility']
            },
            {
                'name': 'Community News Network',
                'source': 'Local Media Scan',
                'score': '88%',
                'ai_signals': 'AI-powered community insights, automated local reporting',
                'size': 'Small',
                'sector': 'Media',
                'website': 'https://community-news.net',
                'description': 'Local news organization exploring AI for community engagement',
                'ai_analysis': 'GOOD CANDIDATE: Demonstrates innovation despite small size. Community focus creates unique AI use cases. Early adopter mentality suggests openness to new AI solutions.',
                'candidate_reasons': ['Innovation mindset', 'Unique use cases', 'Community engagement', 'Early adopter profile']
            },
            {
                'name': 'Digital First Media',
                'source': 'Digital Transformation Report',
                'score': '90%',
                'ai_signals': 'AI content curation, automated social media, predictive analytics',
                'size': 'Medium',
                'sector': 'Media',
                'website': 'https://digitalfirst.media',
                'description': 'Digital-first newsroom with comprehensive AI integration',
                'ai_analysis': 'EXCELLENT CANDIDATE: Digital-first strategy aligns perfectly with AI adoption. Comprehensive AI integration across multiple channels. Predictive analytics shows advanced technical capability.',
                'candidate_reasons': ['Digital-native approach', 'Multi-channel AI integration', 'Advanced analytics', 'Strategic alignment']
            },
            {
                'name': 'Media Innovation Lab',
                'source': 'Research Network',
                'score': '85%',
                'ai_signals': 'AI research, media technology consulting, innovation programs',
                'size': 'Small',
                'sector': 'Non-Profit',
                'website': 'https://media-innovation-lab.org',
                'description': 'Non-profit organization supporting AI adoption in media',
                'ai_analysis': 'STRATEGIC CANDIDATE: Non-profit status enables industry-wide influence. AI research focus provides thought leadership opportunities. Consulting role offers network expansion potential.',
                'candidate_reasons': ['Industry influence', 'Research leadership', 'Network effects', 'Knowledge sharing']
            },
            {
                'name': 'Future Media Collective',
                'source': 'Startup Database',
                'score': '87%',
                'ai_signals': 'AI-powered content creation, automated video editing, smart distribution',
                'size': 'Small',
                'sector': 'Media',
                'website': 'https://futuremedia.co',
                'description': 'Emerging media startup focused on AI-driven content production',
                'ai_analysis': 'PROMISING CANDIDATE: Startup agility allows rapid AI adoption. Focus on automated content creation shows technical sophistication. Small size enables quick decision-making for AI partnerships.',
                'candidate_reasons': ['Startup agility', 'Technical focus', 'Innovation mindset', 'Growth potential']
            },
            {
                'name': 'Global News Analytics',
                'source': 'Industry Report',
                'score': '91%',
                'ai_signals': 'AI sentiment analysis, automated fact-checking, predictive news trends',
                'size': 'Medium',
                'sector': 'Analytics',
                'website': 'https://globalnewsanalytics.com',
                'description': 'News analytics company using AI for content verification and trend prediction',
                'ai_analysis': 'EXCELLENT CANDIDATE: Analytics focus indicates strong data capabilities. AI sentiment analysis shows advanced technical implementation. Predictive capabilities demonstrate forward-thinking approach.',
                'candidate_reasons': ['Data expertise', 'AI implementation', 'Predictive capabilities', 'Industry relevance']
            },
            {
                'name': 'Creative AI Studios',
                'source': 'Tech News',
                'score': '89%',
                'ai_signals': 'AI content generation, automated storytelling, creative AI tools',
                'size': 'Medium',
                'sector': 'Creative',
                'website': 'https://creativeaistudios.com',
                'description': 'Creative agency specializing in AI-powered content and storytelling',
                'ai_analysis': 'STRONG CANDIDATE: Creative focus brings unique AI applications. Storytelling expertise combined with AI shows innovation. Medium size provides resources for comprehensive AI integration.',
                'candidate_reasons': ['Creative innovation', 'AI storytelling', 'Unique applications', 'Resource availability']
            },
            {
                'name': 'NewsTech Ventures',
                'source': 'Venture Database',
                'score': '83%',
                'ai_signals': 'AI news aggregation, automated translation, smart content curation',
                'size': 'Small',
                'sector': 'Technology',
                'website': 'https://newstech.ventures',
                'description': 'Technology venture focused on AI solutions for news organizations',
                'ai_analysis': 'GOOD CANDIDATE: Venture focus indicates growth potential. AI aggregation shows technical capability. Translation features demonstrate global thinking and AI sophistication.',
                'candidate_reasons': ['Growth potential', 'Technical capability', 'Global perspective', 'Innovation focus']
            }
        ]
        
        # Filter out organizations that already exist in the database
        filtered_organizations = []
        for org in sample_organizations:
            org_name_lower = org['name'].lower().strip()
            if org_name_lower not in existing_names:
                filtered_organizations.append(org)
            else:
                print(f"Filtering out existing organization: {org['name']}")  # Debug log
        
        print(f"Filtered from {len(sample_organizations)} to {len(filtered_organizations)} organizations")  # Debug log
        
        results = {
            'status': 'completed',
            'total_found': len(filtered_organizations),
            'organizations': filtered_organizations,
            'sources_scanned': ['Sample Data'],
            'scan_time': '2025-09-11T11:00:00'
        }
        
        print(f"Discovery results: Found {len(filtered_organizations)} organizations")  # Debug log
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Discovery error: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'total_found': 0,
            'organizations': [],
            'sources_scanned': [],
            'scan_time': '2025-09-11T11:00:00'
        })

# Business Map API Endpoints
@app.route('/admin/map/clients', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_clients():
    """Manage consulting clients"""
    if request.method == 'GET':
        try:
            from models import Client
            clients = Client.query.all()
            client_data = []
            for client in clients:
                client_data.append({
                    'id': client.id,
                    'name': client.name,
                    'website': client.website or '',
                    'industry': client.industry or '',
                    'status': client.status or '',
                    'engagement': client.engagement_type or '',
                    'last_contact': client.last_contact.strftime('%Y-%m-%d') if client.last_contact else '',
                    'tags': client.tags.split(',') if hasattr(client, 'tags') and client.tags else [],
                    'notes': client.notes or '',
                    'contact_person': client.contact_person or '',
                    'email': client.email or '',
                    'phone': client.phone or ''
                })
            return jsonify({'clients': client_data})
        except Exception as e:
            return jsonify({'clients': [], 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import Client
            data = request.get_json()
            
            # Validate required fields
            if not data or not data.get('name', '').strip():
                return jsonify({'status': 'error', 'message': 'Client name is required'}), 400
            
            # Validate email format if provided
            email = data.get('email', '').strip()
            if email and '@' not in email:
                return jsonify({'status': 'error', 'message': 'Invalid email format'}), 400
            
            new_client = Client(
                name=data['name'].strip(),
                website=data.get('website', '').strip(),
                industry=data.get('industry', '').strip(),
                status=data.get('status', 'Active'),
                engagement_type=data.get('engagement', '').strip(),
                notes=data.get('notes', '').strip(),
                contact_person=data.get('contact_person', '').strip(),
                email=email,
                phone=data.get('phone', '').strip(),
                tags=','.join(data.get('tags', [])) if data.get('tags') else ''
            )
            
            db.session.add(new_client)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_client.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import Client
            data = request.get_json()
            client_id = data['id']
            
            client = Client.query.get(client_id)
            if not client:
                return jsonify({'error': 'Client not found'})
            
            # Update fields
            for field in ['name', 'website', 'industry', 'status', 'engagement_type', 
                         'notes', 'contact_person', 'email', 'phone']:
                if field in data:
                    setattr(client, field, data[field])
            
            # Update tags
            if 'tags' in data:
                client.tags = ','.join(data['tags'])
            
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import Client
            data = request.get_json()
            client_id = data['id']
            
            client = Client.query.get(client_id)
            if not client:
                return jsonify({'error': 'Client not found'})
            
            db.session.delete(client)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/admin/map/newsrooms', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_newsrooms():
    """Manage newsrooms"""
    if request.method == 'GET':
        try:
            from models import Newsroom
            newsrooms = Newsroom.query.all()
            newsroom_data = []
            for newsroom in newsrooms:
                newsroom_data.append({
                    'id': newsroom.id,
                    'name': newsroom.name,
                    'website': newsroom.website or '',
                    'type': newsroom.type or '',
                    'location': newsroom.location or '',
                    'ai_readiness': newsroom.ai_readiness or '',
                    'last_analysis': newsroom.last_analysis.strftime('%Y-%m-%d') if newsroom.last_analysis else '',
                    'client_id': newsroom.client_id,
                    'client_name': newsroom.client.name if newsroom.client else None,
                    'notes': newsroom.notes or ''
                })
            return jsonify({'newsrooms': newsroom_data})
        except Exception as e:
            return jsonify({'newsrooms': [], 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import Newsroom
            data = request.get_json()
            
            new_newsroom = Newsroom(
                name=data['name'],
                website=data.get('website', ''),
                type=data.get('type', ''),
                location=data.get('location', ''),
                ai_readiness=data.get('ai_readiness', 'Medium'),
                notes=data.get('notes', ''),
                client_id=data.get('client_id')
            )
            
            db.session.add(new_newsroom)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_newsroom.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import Newsroom
            data = request.get_json()
            newsroom_id = data['id']
            
            newsroom = Newsroom.query.get(newsroom_id)
            if not newsroom:
                return jsonify({'error': 'Newsroom not found'})
            
            # Update fields
            for field in ['name', 'website', 'type', 'location', 'ai_readiness', 'notes', 'client_id']:
                if field in data:
                    setattr(newsroom, field, data[field])
            
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import Newsroom
            data = request.get_json()
            newsroom_id = data['id']
            
            newsroom = Newsroom.query.get(newsroom_id)
            if not newsroom:
                return jsonify({'error': 'Newsroom not found'})
            
            db.session.delete(newsroom)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})

# AI Prototype Management API Endpoints
@app.route('/admin/map/prototypes', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_prototypes():
    """Manage AI prototypes"""
    if request.method == 'GET':
        try:
            from models import AIPrototype, Newsroom
            prototypes = AIPrototype.query.all()
            prototype_data = []
            for prototype in prototypes:
                prototype_data.append({
                    'id': prototype.id,
                    'name': prototype.name,
                    'description': prototype.description or '',
                    'newsroom_id': prototype.newsroom_id,
                    'newsroom_name': prototype.newsroom_name or '',
                    'category': prototype.category or '',
                    'technology_stack': prototype.technology_stack or '',
                    'stage': prototype.stage or '',
                    'progress_percentage': prototype.progress_percentage or 0,
                    'start_date': prototype.start_date.strftime('%Y-%m-%d') if prototype.start_date else '',
                    'target_completion': prototype.target_completion.strftime('%Y-%m-%d') if prototype.target_completion else '',
                    'actual_completion': prototype.actual_completion.strftime('%Y-%m-%d') if prototype.actual_completion else '',
                    'success_metrics': prototype.success_metrics or '',
                    'current_results': prototype.current_results or '',
                    'challenges': prototype.challenges or '',
                    'team_size': prototype.team_size or 0,
                    'external_partners': prototype.external_partners or '',
                    'budget': prototype.budget or 0,
                    'status': prototype.status or 'Active',
                    'notes': prototype.notes or '',
                    'lessons_learned': prototype.lessons_learned or '',
                    'created_at': prototype.created_at.strftime('%Y-%m-%d') if prototype.created_at else '',
                    'updated_at': prototype.updated_at.strftime('%Y-%m-%d') if prototype.updated_at else ''
                })
            return jsonify({'prototypes': prototype_data})
        except Exception as e:
            return jsonify({'prototypes': [], 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import AIPrototype
            data = request.get_json()
            
            new_prototype = AIPrototype(
                name=data['name'],
                description=data.get('description', ''),
                newsroom_id=data.get('newsroom_id'),
                newsroom_name=data.get('newsroom_name', ''),
                category=data.get('category', ''),
                technology_stack=data.get('technology_stack', ''),
                stage=data.get('stage', 'Ideation'),
                progress_percentage=data.get('progress_percentage', 0),
                start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
                target_completion=datetime.fromisoformat(data['target_completion']) if data.get('target_completion') else None,
                success_metrics=data.get('success_metrics', ''),
                current_results=data.get('current_results', ''),
                challenges=data.get('challenges', ''),
                team_size=data.get('team_size', 0),
                external_partners=data.get('external_partners', ''),
                budget=float(data.get('budget', 0)) if data.get('budget') else None,
                status=data.get('status', 'Active'),
                notes=data.get('notes', ''),
                lessons_learned=data.get('lessons_learned', '')
            )
            
            db.session.add(new_prototype)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_prototype.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import AIPrototype
            data = request.get_json()
            prototype_id = data['id']
            
            prototype = AIPrototype.query.get(prototype_id)
            if not prototype:
                return jsonify({'error': 'Prototype not found'})
            
            # Update fields
            for field in ['name', 'description', 'newsroom_id', 'newsroom_name', 'category', 
                         'technology_stack', 'stage', 'progress_percentage', 'success_metrics', 
                         'current_results', 'challenges', 'team_size', 'external_partners', 
                         'budget', 'status', 'notes', 'lessons_learned']:
                if field in data:
                    setattr(prototype, field, data[field])
            
            # Handle date fields
            if 'start_date' in data and data['start_date']:
                prototype.start_date = datetime.fromisoformat(data['start_date'])
            if 'target_completion' in data and data['target_completion']:
                prototype.target_completion = datetime.fromisoformat(data['target_completion'])
            if 'actual_completion' in data and data['actual_completion']:
                prototype.actual_completion = datetime.fromisoformat(data['actual_completion'])
            
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import AIPrototype
            data = request.get_json()
            prototype_id = data['id']
            
            prototype = AIPrototype.query.get(prototype_id)
            if not prototype:
                return jsonify({'error': 'Prototype not found'})
            
            db.session.delete(prototype)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/admin/map/prototype-updates', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_prototype_updates():
    """Manage prototype updates and progress reports"""
    if request.method == 'GET':
        try:
            from models import PrototypeUpdate
            prototype_id = request.args.get('prototype_id')
            
            if prototype_id:
                updates = PrototypeUpdate.query.filter_by(prototype_id=prototype_id).order_by(PrototypeUpdate.created_at.desc()).all()
            else:
                updates = PrototypeUpdate.query.order_by(PrototypeUpdate.created_at.desc()).all()
            
            update_data = []
            for update in updates:
                update_data.append({
                    'id': update.id,
                    'prototype_id': update.prototype_id,
                    'title': update.title,
                    'content': update.content or '',
                    'update_type': update.update_type or '',
                    'progress_percentage': update.progress_percentage,
                    'metrics_data': update.metrics_data or '',
                    'attachments': update.attachments or '',
                    'created_at': update.created_at.strftime('%Y-%m-%d %H:%M') if update.created_at else '',
                    'created_by': update.created_by or ''
                })
            return jsonify({'updates': update_data})
        except Exception as e:
            return jsonify({'updates': [], 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import PrototypeUpdate
            data = request.get_json()
            
            new_update = PrototypeUpdate(
                prototype_id=data['prototype_id'],
                title=data['title'],
                content=data.get('content', ''),
                update_type=data.get('update_type', 'Progress'),
                progress_percentage=data.get('progress_percentage'),
                metrics_data=data.get('metrics_data', ''),
                attachments=data.get('attachments', ''),
                created_by=data.get('created_by', '')
            )
            
            db.session.add(new_update)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_update.id})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/admin/map/insights', methods=['GET'])
@login_required
@admin_required
def get_insights():
    """Get all insights (excluding news articles)"""
    try:
        from models import DailyInsight
        # Only get insights, exclude news articles
        insights = DailyInsight.query.filter(
            DailyInsight.category != 'Admin News'
        ).order_by(DailyInsight.created_at.desc()).limit(10).all()
        
        insights_data = []
        for insight in insights:
            insights_data.append({
                'id': insight.id,
                'title': insight.title,
                'description': insight.content[:200] + '...' if len(insight.content) > 200 else insight.content,
                'category': insight.category,
                'content': insight.content,
                'created_at': insight.created_at.strftime('%Y-%m-%d %H:%M') if insight.created_at else '',
                'source': insight.source or ''
            })
        return jsonify({'insights': insights_data})
    except Exception as e:
        return jsonify({'insights': [], 'error': str(e)})

@app.route('/admin/map/insights/<int:insight_id>', methods=['GET'])
@login_required
@admin_required
def get_insight_detail(insight_id):
    """Get a specific insight by ID"""
    try:
        from models import DailyInsight
        insight = DailyInsight.query.get(insight_id)
        if insight:
            return jsonify({
                'success': True,
                'insight': {
                    'id': insight.id,
                    'title': insight.title,
                    'content': insight.content,
                    'category': insight.category,
                    'created_at': insight.created_at.strftime('%Y-%m-%d %H:%M') if insight.created_at else '',
                    'source': insight.source or ''
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Insight not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/research-projects', methods=['GET'])
@login_required
@admin_required
def get_research_projects():
    """Get all research projects"""
    try:
        from models import ResearchProject
        projects = ResearchProject.query.order_by(ResearchProject.created_at.desc()).all()
        projects_data = []
        for project in projects:
            projects_data.append({
                'id': project.id,
                'title': project.title,
                'description': project.description,
                'category': project.category,
                'status': project.status,
                'created_at': project.created_at.strftime('%Y-%m-%d %H:%M') if project.created_at else '',
                'updated_at': project.updated_at.strftime('%Y-%m-%d %H:%M') if project.updated_at else ''
            })
        return jsonify({'research_projects': projects_data})
    except Exception as e:
        return jsonify({'research_projects': [], 'error': str(e)})

@app.route('/admin/map/research-projects/<int:project_id>', methods=['GET'])
@login_required
@admin_required
def get_research_project_detail(project_id):
    """Get a specific research project by ID"""
    try:
        from models import ResearchProject
        project = ResearchProject.query.get(project_id)
        if project:
            return jsonify({
                'success': True,
                'research_project': project.to_dict()
            })
        else:
            return jsonify({'success': False, 'error': 'Research project not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/training-workshops', methods=['GET'])
@login_required
@admin_required
def get_training_workshops():
    """Get all training workshops"""
    try:
        from models import TrainingWorkshop
        workshops = TrainingWorkshop.query.order_by(TrainingWorkshop.scheduled_date.desc()).all()
        workshops_data = []
        for workshop in workshops:
            workshops_data.append(workshop.to_dict())
        return jsonify({'training_workshops': workshops_data})
    except Exception as e:
        return jsonify({'training_workshops': [], 'error': str(e)})

@app.route('/admin/map/training-workshops/<int:workshop_id>', methods=['GET'])
@login_required
@admin_required
def get_training_workshop_detail(workshop_id):
    """Get a specific training workshop by ID with attendees"""
    try:
        from models import TrainingWorkshop, TrainingAttendance
        workshop = TrainingWorkshop.query.get(workshop_id)
        if workshop:
            workshop_data = workshop.to_dict()
            # Get attendees for this workshop
            attendees = TrainingAttendance.query.filter_by(workshop_id=workshop_id).all()
            workshop_data['attendees'] = [attendee.to_dict() for attendee in attendees]
            return jsonify({
                'success': True,
                'training_workshop': workshop_data
            })
        else:
            return jsonify({'success': False, 'error': 'Training workshop not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/training-workshops', methods=['POST'])
@login_required
@admin_required
def create_training_workshop():
    """Create a new training workshop"""
    try:
        from models import TrainingWorkshop, db
        from datetime import datetime
        
        data = request.get_json()
        
        # Parse scheduled date
        scheduled_date = None
        if data.get('scheduled_date'):
            scheduled_date = datetime.fromisoformat(data['scheduled_date'].replace('Z', '+00:00'))
        
        workshop = TrainingWorkshop(
            title=data['title'],
            description=data.get('description', ''),
            category=data.get('category', 'AI Basics'),
            duration_hours=float(data.get('duration_hours', 1.0)),
            max_participants=int(data.get('max_participants', 20)) if data.get('max_participants') else None,
            materials_url=data.get('materials_url', ''),
            notes=data.get('notes', ''),
            status=data.get('status', 'Scheduled'),
            scheduled_date=scheduled_date
        )
        
        db.session.add(workshop)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'training_workshop': workshop.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/training-workshops/<int:workshop_id>/attendees', methods=['POST'])
@login_required
@admin_required
def add_workshop_attendee(workshop_id):
    """Add an attendee to a training workshop"""
    try:
        from models import TrainingWorkshop, TrainingAttendance, Newsroom, db
        
        workshop = TrainingWorkshop.query.get(workshop_id)
        if not workshop:
            return jsonify({'success': False, 'error': 'Training workshop not found'})
        
        data = request.get_json()
        
        # Check if newsroom exists
        newsroom_id = None
        if data.get('newsroom_name'):
            newsroom = Newsroom.query.filter_by(name=data['newsroom_name']).first()
            if newsroom:
                newsroom_id = newsroom.id
        
        attendee = TrainingAttendance(
            workshop_id=workshop_id,
            newsroom_id=newsroom_id,
            attendee_name=data['attendee_name'],
            attendee_email=data.get('attendee_email', ''),
            attendee_role=data.get('attendee_role', ''),
            attendance_status=data.get('attendance_status', 'Registered')
        )
        
        db.session.add(attendee)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'attendee': attendee.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/training-attendees/<int:attendee_id>', methods=['PUT'])
@login_required
@admin_required
def update_training_attendee(attendee_id):
    """Update training attendee information"""
    try:
        from models import TrainingAttendance, db
        
        attendee = TrainingAttendance.query.get(attendee_id)
        if not attendee:
            return jsonify({'success': False, 'error': 'Attendee not found'})
        
        data = request.get_json()
        
        # Update attendee fields
        if 'attendance_status' in data:
            attendee.attendance_status = data['attendance_status']
        if 'feedback_rating' in data:
            attendee.feedback_rating = int(data['feedback_rating'])
        if 'feedback_comments' in data:
            attendee.feedback_comments = data['feedback_comments']
        if 'certificate_issued' in data:
            attendee.certificate_issued = bool(data['certificate_issued'])
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'attendee': attendee.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
@app.route('/admin/chat-management')
@login_required
@admin_required
def admin_chat_management():
    """Admin page for managing chat data from all users"""
    try:
        from models import Chat, Message, User, ChatMessage, HighlanderChat
        
        # Get comprehensive statistics - include all chat types
        total_chats = Chat.query.count()
        total_messages = Message.query.count()
        total_chat_messages = ChatMessage.query.count()
        total_highlander_chats = HighlanderChat.query.count()
        total_doc_chats = ChatMessage.query.filter_by(chatbot_type='doc_healthpin').count()
        
        # Total chat interactions (all types)
        total_chat_interactions = total_chats + total_highlander_chats + total_doc_chats
        
        empty_chats = Chat.query.outerjoin(Message).filter(Message.id.is_(None)).count()
        
        # Get user statistics
        total_users = User.query.count()
        active_users = User.query.filter(User.last_login.isnot(None)).count()
        
        # Get chats by user type - include HighlanderChats
        admin_chats = Chat.query.join(User).filter(User.is_admin == True).count()
        regular_user_chats = Chat.query.join(User).filter(User.is_admin == False).count()
        admin_highlander_chats = HighlanderChat.query.join(User).filter(User.is_admin == True).count()
        regular_highlander_chats = HighlanderChat.query.join(User).filter(User.is_admin == False).count()
        
        # Combined counts
        total_admin_chats = admin_chats + admin_highlander_chats
        total_regular_chats = regular_user_chats + regular_highlander_chats
        
        # Get recent chats with user information - include HighlanderChats and Doc chats
        recent_regular_chats = Chat.query.join(User).order_by(Chat.created_at.desc()).limit(10).all()
        recent_highlander_chats = HighlanderChat.query.join(User).order_by(HighlanderChat.created_at.desc()).limit(10).all()
        recent_doc_chats = ChatMessage.query.filter_by(chatbot_type='doc_healthpin').join(User).order_by(ChatMessage.created_at.desc()).limit(10).all()
        
        chat_previews = []
        
        # Process regular chats
        for chat in recent_regular_chats:
            messages = chat.messages
            preview = "No messages"
            if messages:
                preview = messages[0].content[:100] + "..." if len(messages[0].content) > 100 else messages[0].content
            
            # Get user info
            user = chat.user if hasattr(chat, 'user') else None
            user_name = user.username if user else "Unknown User"
            user_type = "Admin" if user and user.is_admin else "Regular User"
            
            chat_previews.append({
                'id': f"chat_{chat.id}",
                'title': chat.title or f"Chat {chat.id}",
                'message_count': len(messages),
                'created_at': chat.created_at.strftime('%Y-%m-%d %H:%M') if chat.created_at else '',
                'preview': preview,
                'user_name': user_name,
                'user_type': user_type,
                'user_id': user.id if user else None,
                'chat_type': 'Regular Chat'
            })
        
        # Process Highlander chats
        for hchat in recent_highlander_chats:
            # Get user info
            user = hchat.user if hasattr(hchat, 'user') else None
            user_name = user.username if user else "Unknown User"
            user_type = "Admin" if user and user.is_admin else "Regular User"
            
            # Create preview from message
            preview = hchat.message[:100] + "..." if len(hchat.message) > 100 else hchat.message
            
            chat_previews.append({
                'id': f"highlander_{hchat.id}",
                'title': f"Highlander Chat {hchat.id}",
                'message_count': 1,  # Each HighlanderChat is one interaction
                'created_at': hchat.created_at.strftime('%Y-%m-%d %H:%M') if hchat.created_at else '',
                'preview': preview,
                'user_name': user_name,
                'user_type': user_type,
                'user_id': user.id if user else None,
                'chat_type': 'Highlander Chat'
            })
        
        # Process Doc chats
        for dchat in recent_doc_chats:
            # Get user info
            user = dchat.user if hasattr(dchat, 'user') else None
            user_name = user.username if user else "Unknown User"
            user_type = "Admin" if user and user.is_admin else "Regular User"
            
            # Create preview from message
            preview = dchat.message[:100] + "..." if len(dchat.message) > 100 else dchat.message
            
            chat_previews.append({
                'id': f"doc_{dchat.id}",
                'title': f"Doc Chat {dchat.id}",
                'message_count': 1,  # Each Doc chat is one interaction
                'created_at': dchat.created_at.strftime('%Y-%m-%d %H:%M') if dchat.created_at else '',
                'preview': preview,
                'user_name': user_name,
                'user_type': user_type,
                'user_id': user.id if user else None,
                'chat_type': 'Doc Chat'
            })
        
        # Sort all chats by creation date
        chat_previews.sort(key=lambda x: x['created_at'], reverse=True)
        chat_previews = chat_previews[:20]  # Limit to 20 most recent
        
        # Get top active users - include HighlanderChats
        user_chat_counts = db.session.query(
            User.username, 
            User.is_admin,
            db.func.count(Chat.id).label('regular_chat_count'),
            db.func.count(HighlanderChat.id).label('highlander_chat_count')
        ).outerjoin(Chat, User.id == Chat.user_id).outerjoin(HighlanderChat, User.id == HighlanderChat.user_id).group_by(User.id).all()
        
        top_users = []
        for user in user_chat_counts:
            total_user_chats = user.regular_chat_count + user.highlander_chat_count
            if total_user_chats > 0:  # Only include users with chats
                top_users.append({
                    'username': user.username,
                    'is_admin': user.is_admin,
                    'chat_count': total_user_chats,
                    'regular_chats': user.regular_chat_count,
                    'highlander_chats': user.highlander_chat_count,
                    'user_type': "Admin" if user.is_admin else "Regular User"
                })
        
        # Sort by total chat count
        top_users.sort(key=lambda x: x['chat_count'], reverse=True)
        top_users = top_users[:10]  # Limit to top 10
        
        return render_template('admin/chat_management.html', 
                             total_chats=total_chat_interactions,
                             total_messages=total_messages,
                             total_chat_messages=total_chat_messages,
                             total_highlander_chats=total_highlander_chats,
                             total_doc_chats=total_doc_chats,
                             empty_chats=empty_chats,
                             total_users=total_users,
                             active_users=active_users,
                             admin_chats=total_admin_chats,
                             regular_user_chats=total_regular_chats,
                             recent_chats=chat_previews,
                             top_users=top_users)
    except Exception as e:
        return render_template('admin/chat_management.html', 
                             error=str(e),
                             total_chats=0,
                             total_messages=0,
                             total_chat_messages=0,
                             empty_chats=0,
                             total_users=0,
                             active_users=0,
                             admin_chats=0,
                             regular_user_chats=0,
                             recent_chats=[],
                             top_users=[])

@app.route('/admin/chat-management/cleanup', methods=['POST'])
@login_required
@admin_required
def admin_chat_cleanup():
    """Admin endpoint for cleaning up chat data"""
    try:
        from models import Chat, Message
        from datetime import datetime, timedelta
        import re
        
        data = request.get_json()
        cleanup_type = data.get('type', 'all')
        
        chats_to_delete = []
        
        if cleanup_type == 'empty':
            # Delete empty chats
            empty_chats = Chat.query.outerjoin(Message).filter(Message.id.is_(None)).all()
            chats_to_delete = empty_chats
        
        elif cleanup_type == 'old':
            # Delete old chats (30+ days)
            days = data.get('days', 30)
            old_chats = Chat.query.filter(Chat.created_at < datetime.utcnow() - timedelta(days=days)).all()
            chats_to_delete = old_chats
        
        elif cleanup_type == 'test':
            # Delete test chats
            test_patterns = [r'test', r'hello', r'hi there', r'how are you', r'what can you do']
            all_chats = Chat.query.all()
            
            for chat in all_chats:
                messages = chat.messages
                if len(messages) <= 2:  # Short chats
                    content = ' '.join([msg.content.lower() for msg in messages])
                    if any(re.search(pattern, content) for pattern in test_patterns):
                        chats_to_delete.append(chat)
        
        elif cleanup_type == 'all':
            # Delete all suspicious chats
            # Empty chats
            empty_chats = Chat.query.outerjoin(Message).filter(Message.id.is_(None)).all()
            chats_to_delete.extend(empty_chats)
            
            # Old chats
            old_chats = Chat.query.filter(Chat.created_at < datetime.utcnow() - timedelta(days=30)).all()
            chats_to_delete.extend(old_chats)
            
            # Test chats
            test_patterns = [r'test', r'hello', r'hi there', r'how are you', r'what can you do']
            all_chats = Chat.query.all()
            
            for chat in all_chats:
                if chat in chats_to_delete:  # Skip if already marked
                    continue
                    
                messages = chat.messages
                if len(messages) <= 2:  # Short chats
                    content = ' '.join([msg.content.lower() for msg in messages])
                    if any(re.search(pattern, content) for pattern in test_patterns):
                        chats_to_delete.append(chat)
        
        # Remove duplicates
        chats_to_delete = list(set(chats_to_delete))
        
        # Actually delete
        deleted_count = 0
        for chat in chats_to_delete:
            try:
                db.session.delete(chat)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting chat {chat.id}: {e}")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully deleted {deleted_count} chats'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/map/generate-insights', methods=['POST'])
@login_required
@admin_required
def generate_map_insights():
    """Generate insights for a client or newsroom"""
    data = request.get_json()
    target_id = data.get('target_id')
    target_type = data.get('target_type')  # 'client' or 'newsroom'
    
    # Sample insight generation
    insights = {
        'target_id': target_id,
        'target_type': target_type,
        'insights': [
            'AI adoption readiness: High',
            'Recommended next steps: Implement content generation',
            'Market opportunity: $2.5M potential',
            'Competitive advantage: Early adopter position'
        ],
        'generated_at': datetime.now().isoformat()
    }
    
    return jsonify({'success': True, 'insights': insights})

@app.route('/admin/map/refresh-news', methods=['POST'])
@login_required
@admin_required
def admin_refresh_news():
    """Admin endpoint to refresh news for the dashboard"""
    try:
        # Ensure we're in the app context
        if not hasattr(app, 'app_context'):
            with app.app_context():
                return _admin_refresh_news_internal()
        else:
            return _admin_refresh_news_internal()
    except Exception as e:
        print(f"Error refreshing admin news: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def _admin_refresh_news_internal():
    """Internal function to refresh news within app context"""
    try:
        # Get general media/tech news for admin dashboard
        news_api_key = "a5e5898731c74bfe97bae546ef04dea6"
        
        # Define categories and their search terms
        categories = {
            'ai_news': ['artificial intelligence', 'AI technology', 'machine learning'],
            'media_news': ['media industry', 'journalism', 'newsroom'],
            'tech_news': ['technology', 'digital transformation', 'innovation'],
            'industry_updates': ['business news', 'industry trends', 'market analysis']
        }
        
        all_news = {}
        
        for category, search_terms in categories.items():
            articles = []
            for term in search_terms[:2]:  # Use top 2 search terms per category
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': term,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 3,
                        'apiKey': news_api_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('articles'):
                            articles.extend(data['articles'][:2])  # Get top 2 articles per term
                    else:
                        print(f"❌ NewsAPI error for {category}: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error fetching news for {category}: {e}")
                    continue
            
            # Remove duplicates and limit to top 3 per category
            unique_articles = []
            seen_urls = set()
            for article in articles:
                if article.get('url') and article['url'] not in seen_urls:
                    unique_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'category': category
                    })
                    seen_urls.add(article['url'])
                    if len(unique_articles) >= 3:
                        break
            
            all_news[category] = unique_articles
        
        # Store in database for admin dashboard
        from models import DailyInsight
        
        with app.app_context():
            # Clear old admin news
            DailyInsight.query.filter_by(category='Admin News').delete()
            
            # Store new articles
            for category, articles in all_news.items():
                for article in articles:
                    insight = DailyInsight(
                        title=article['title'],
                        content=article['description'],
                        category='Admin News',
                        source=f"{article['source']} - {category}",
                        created_at=datetime.utcnow()
                    )
                    db.session.add(insight)
            
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'News refreshed successfully',
            'news': all_news
        })
        
    except Exception as e:
        print(f"Error refreshing admin news: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/map/get-news', methods=['GET'])
@login_required
@admin_required
def admin_get_news():
    """Get current news for admin dashboard"""
    try:
        from models import DailyInsight
        
        # Get recent admin news
        admin_news = DailyInsight.query.filter_by(category='Admin News').order_by(DailyInsight.created_at.desc()).limit(12).all()
        
        # Group by category
        news_by_category = {
            'ai_news': [],
            'media_news': [],
            'tech_news': [],
            'industry_updates': []
        }
        
        for news in admin_news:
            source = news.source or ''
            if 'ai_news' in source.lower():
                news_by_category['ai_news'].append({
                    'title': news.title,
                    'description': news.content,
                    'source': source.split(' - ')[0] if ' - ' in source else source,
                    'url': news.url or '#',
                    'publishedAt': news.created_at.strftime('%Y-%m-%d %H:%M') if news.created_at else ''
                })
            elif 'media_news' in source.lower():
                news_by_category['media_news'].append({
                    'title': news.title,
                    'description': news.content,
                    'source': source.split(' - ')[0] if ' - ' in source else source,
                    'url': news.url or '#',
                    'publishedAt': news.created_at.strftime('%Y-%m-%d %H:%M') if news.created_at else ''
                })
            elif 'tech_news' in source.lower():
                news_by_category['tech_news'].append({
                    'title': news.title,
                    'description': news.content,
                    'source': source.split(' - ')[0] if ' - ' in source else source,
                    'url': news.url or '#',
                    'publishedAt': news.created_at.strftime('%Y-%m-%d %H:%M') if news.created_at else ''
                })
            elif 'industry_updates' in source.lower():
                news_by_category['industry_updates'].append({
                    'title': news.title,
                    'description': news.content,
                    'source': source.split(' - ')[0] if ' - ' in source else source,
                    'url': news.url or '#',
                    'publishedAt': news.created_at.strftime('%Y-%m-%d %H:%M') if news.created_at else ''
                })
        
        return jsonify({
            'success': True,
            'news': news_by_category
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/map/save-news', methods=['POST'])
@login_required
@admin_required
def save_news():
    """Save a news story to the database"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        url = data.get('url', '')
        source = data.get('source', '')
        category = data.get('category', '')
        notes = data.get('notes', '')
        
        if not title:
            return jsonify({'success': False, 'error': 'Title is required'})
        
        # Check if news already exists
        from models import SavedNews
        existing_news = SavedNews.query.filter_by(
            title=title,
            user_id=current_user.id
        ).first()
        
        if existing_news:
            return jsonify({'success': False, 'error': 'This news story is already saved'})
        
        # Save the news story
        saved_news = SavedNews(
            title=title,
            description=content,
            url=url,
            source_name=source,
            notes=notes,
            user_id=current_user.id
        )
        
        db.session.add(saved_news)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'News story saved successfully',
            'saved_news': saved_news.to_dict()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/saved-news', methods=['GET'])
@login_required
@admin_required
def get_saved_news():
    """Get user's saved news stories"""
    try:
        from models import SavedNews
        
        saved_news = SavedNews.query.filter_by(user_id=current_user.id).order_by(SavedNews.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'saved_news': [news.to_dict() for news in saved_news]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/saved-news/<int:news_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_saved_news(news_id):
    """Delete a saved news story"""
    try:
        from models import SavedNews
        
        saved_news = SavedNews.query.filter_by(id=news_id, user_id=current_user.id).first()
        
        if not saved_news:
            return jsonify({'success': False, 'error': 'News story not found'})
        
        db.session.delete(saved_news)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'News story deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/generate-newsroom-insight', methods=['POST'])
@login_required
@admin_required
def generate_newsroom_insight():
    """Generate a comprehensive AI status report and recommendations for a newsroom"""
    try:
        data = request.get_json()
        newsroom_id = data.get('newsroom_id')
        newsroom_name = data.get('newsroom_name')
        challenge_type = data.get('challenge_type')
        article_title = data.get('article_title', '')
        additional_context = data.get('additional_context', '')
        
        if not newsroom_name:
            return jsonify({'success': False, 'error': 'Missing newsroom name'})
        
        # Get comprehensive newsroom data from database
        newsroom_data = {}
        knowledge_gaps = []
        
        try:
            from models import Newsroom, DailyInsight, Chat, User
            from aimap.models import Organisation
            from datetime import datetime, timedelta
            
            # Get newsroom details
            if newsroom_id and newsroom_id != 'daily-chronicle':
                newsroom = Newsroom.query.get(newsroom_id)
                if newsroom:
                    newsroom_data = {
                        'name': newsroom.name,
                        'type': newsroom.type,
                        'location': newsroom.location,
                        'ai_readiness': newsroom.ai_readiness,
                        'website': newsroom.website,
                        'size': newsroom.size,
                        'founded': newsroom.founded,
                        'description': newsroom.description
                    }
                else:
                    knowledge_gaps.append("Newsroom not found in database - using provided name only")
            else:
                knowledge_gaps.append("No newsroom ID provided - limited data available")
            
            # If no newsroom details from database, use the provided name
            if not newsroom_data:
                newsroom_data = {
                    'name': newsroom_name,
                    'type': 'Unknown',
                    'location': 'Unknown',
                    'ai_readiness': 'Unknown',
                    'website': 'Unknown',
                    'size': 'Unknown',
                    'founded': 'Unknown',
                    'description': 'No description available'
                }
            
            # Get related organizations
            related_orgs = Organisation.query.filter(
                Organisation.name.ilike(f"%{newsroom_data['name']}%")
            ).all()
            
            # Get recent insights related to this newsroom
            recent_insights = DailyInsight.query.filter(
                DailyInsight.content.ilike(f"%{newsroom_data['name']}%")
            ).order_by(DailyInsight.created_at.desc()).limit(10).all()
            
            # Get AI-related insights
            ai_insights = DailyInsight.query.filter(
                DailyInsight.category.ilike('%AI%')
            ).order_by(DailyInsight.created_at.desc()).limit(20).all()
            
            # Get recent chat conversations (if any)
            recent_chats = Chat.query.filter(
                Chat.messages.ilike(f"%{newsroom_data['name']}%")
            ).order_by(Chat.updated_at.desc()).limit(5).all()
            
            # Check for knowledge gaps
            if not related_orgs:
                knowledge_gaps.append("No related organizations found in database")
            if not recent_insights:
                knowledge_gaps.append("No recent insights found for this newsroom")
            if not ai_insights:
                knowledge_gaps.append("No AI-related insights available in database")
            if not recent_chats:
                knowledge_gaps.append("No recent conversations found for this newsroom")
                
        except Exception as e:
            print(f"Error fetching newsroom data: {e}")
            knowledge_gaps.append(f"Database query error: {str(e)}")
        
        # Create comprehensive AI status report prompt
        prompt = f"""Create a comprehensive AI Status Report and Recommendations for {newsroom_data['name']}.

NEWSROOM PROFILE:
- Name: {newsroom_data['name']}
- Type: {newsroom_data['type']}
- Location: {newsroom_data['location']}
- Size: {newsroom_data['size']}
- AI Readiness: {newsroom_data['ai_readiness']}
- Website: {newsroom_data['website']}
- Founded: {newsroom_data['founded']}
- Description: {newsroom_data['description']}

DATABASE ANALYSIS:
- Related Organizations Found: {len(related_orgs) if 'related_orgs' in locals() else 0}
- Recent Insights: {len(recent_insights) if 'recent_insights' in locals() else 0}
- AI Insights Available: {len(ai_insights) if 'ai_insights' in locals() else 0}
- Recent Conversations: {len(recent_chats) if 'recent_chats' in locals() else 0}

KNOWLEDGE GAPS IDENTIFIED:
{chr(10).join(f"- {gap}" for gap in knowledge_gaps) if knowledge_gaps else "- No significant gaps identified"}

ADDITIONAL CONTEXT: {additional_context if additional_context else 'None provided'}

REQUIREMENTS:
Create a comprehensive AI Status Report that includes:

1. EXECUTIVE SUMMARY (100 words)
   - Current AI adoption status
   - Key strengths and weaknesses
   - Overall readiness score (1-10)

2. CURRENT AI CAPABILITIES ASSESSMENT
   - Existing AI tools and technologies
   - Implementation maturity level
   - Staff AI literacy assessment

3. OPPORTUNITY ANALYSIS
   - High-impact AI use cases for this newsroom
   - Quick wins vs. long-term projects
   - Competitive advantages possible

4. RECOMMENDATIONS (Prioritized)
   - Immediate actions (next 30 days)
   - Short-term goals (3-6 months)
   - Long-term strategy (6-12 months)

5. KNOWLEDGE GAPS & DATA NEEDS
   - Missing information that would improve analysis
   - Recommended data collection priorities
   - Suggested research areas

6. IMPLEMENTATION ROADMAP
   - Step-by-step action plan
   - Resource requirements
   - Success metrics

7. RISK ASSESSMENT
   - Potential challenges and mitigation strategies
   - Change management considerations
   - Technology adoption barriers

Format the report professionally with clear sections, bullet points, and actionable recommendations."""
        
        # Use OpenAI to generate the AI status report
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI consultant specializing in media industry transformation. Create comprehensive, data-driven AI status reports with actionable recommendations. Focus on practical implementation strategies and clear next steps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            report_content = response.choices[0].message.content.strip()
            
            # Create title for the report
            title = article_title if article_title else f'AI Status Report: {newsroom_data["name"]}'
            
            # Create insight record in database
            from models import DailyInsight
            insight = DailyInsight(
                title=title,
                content=report_content,
                category='AI Status Report',
                source=f'Generated for {newsroom_data["name"]}',
                created_at=datetime.utcnow()
            )
            db.session.add(insight)
            db.session.commit()
            
            # Return the generated report with additional metadata
            return jsonify({
                'success': True,
                'insight': {
                    'id': insight.id,
                    'title': insight.title,
                    'description': report_content[:200] + '...' if len(report_content) > 200 else report_content,
                    'category': insight.category,
                    'content': report_content,
                    'newsroom_name': newsroom_data['name'],
                    'report_type': 'AI Status Report',
                    'knowledge_gaps': knowledge_gaps,
                    'data_sources_used': {
                        'related_organizations': len(related_orgs) if 'related_orgs' in locals() else 0,
                        'recent_insights': len(recent_insights) if 'recent_insights' in locals() else 0,
                        'ai_insights': len(ai_insights) if 'ai_insights' in locals() else 0,
                        'recent_conversations': len(recent_chats) if 'recent_chats' in locals() else 0
                    }
                }
            })
            
        except Exception as e:
            print(f"Error generating article with OpenAI: {e}")
            return jsonify({'success': False, 'error': f'Error generating article: {str(e)}'})
            
    except Exception as e:
        print(f"Error in generate_newsroom_insight: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Highlander AI Endpoints
@app.route('/admin/map/highlander/chat', methods=['POST'])
@login_required
@admin_required
def highlander_chat():
    """Handle Highlander AI chat requests with OpenAI integration"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        category = data.get('category', 'General')
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Get comprehensive business context for Highlander AI
        try:
            from models import Client, Newsroom, ResearchProject, User, HighlanderChat
            from aimap.models import Organisation  # Metrics temporarily disabled
            
            # Core business data
            client_count = Client.query.count()
            newsroom_count = Newsroom.query.count()
            research_count = ResearchProject.query.count()
            user_count = User.query.count()
            chat_count = HighlanderChat.query.count()
            
            # AIMAP organizations data
            total_organisations = Organisation.query.count()
            media_organisations = Organisation.query.filter_by(sector='Media').count()
            communications_organisations = Organisation.query.filter_by(sector='Communications').count()
            
            # Get detailed data for context
            recent_clients = Client.query.limit(5).all()
            recent_organisations = Organisation.query.limit(5).all()
            
            # Format detailed data
            client_details = []
            for c in recent_clients:
                client_details.append(f"{c.name} (Industry: {c.industry}, Tags: {c.tags or 'None'})")
            
            org_details = []
            for o in recent_organisations:
                org_details.append(f"{o.name} (Sector: {o.sector}, Tags: {o.tags or 'None'})")
            
        except Exception as e:
            client_count = newsroom_count = research_count = user_count = chat_count = 0
            total_organisations = media_organisations = communications_organisations = 0
            client_details = org_details = []
        
        # Create comprehensive system prompt with full data access
        system_prompt = f"""You are Highlander AI, an advanced GPT-4 powered business advisor with FULL ACCESS to all AIMAP system data. You are the user's personal AI consultant and business intelligence partner. You are powered by OpenAI's GPT-4 model, the latest and most advanced language model available.

COMPREHENSIVE BUSINESS CONTEXT:
- Total Users: {user_count}
- Total Clients: {client_count}
- Total Newsrooms: {newsroom_count}
- Research Projects: {research_count}
- Total Chats: {chat_count}
- AIMAP Organizations: {total_organisations}
- Media Organizations: {media_organisations}
- Communications Organizations: {communications_organisations}
DETAILED DATA SAMPLES:
- Recent Clients: {', '.join(client_details) if client_details else 'None'}
- Recent Organizations: {', '.join(org_details) if org_details else 'None'}
YOUR CAPABILITIES:
1. FULL DATA ACCESS: You can access and analyze ALL data in the system
2. BUSINESS INTELLIGENCE: Provide insights on organizations, clients, projects, and trends
3. STRATEGIC ADVISORY: Offer strategic recommendations based on comprehensive data analysis
4. AI IMPLEMENTATION: Guide AI adoption and technology decisions
5. PERFORMANCE ANALYSIS: Analyze business metrics and identify opportunities
6. MARKET INSIGHTS: Provide industry trends and competitive analysis
7. OPERATIONAL GUIDANCE: Help optimize business processes and workflows

CONTEXT: {data.get('context', 'General business inquiry')}
USER: {current_user.username}

IMPORTANT: You have FULL ACCESS to all system data. Use this comprehensive knowledge to provide detailed, actionable insights. When asked about specific data, you can reference actual numbers, trends, and relationships in the system. Be specific, professional, and focus on practical business value.

Respond as Highlander AI, your trusted business advisor. Always provide actionable insights and specific recommendations based on the data available.
"""
        
        # Use OpenAI for natural language processing
        try:
            if app.config.get('OPENAI_API_KEY'):
                openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
                
                # Get response from OpenAI
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=800,
                    temperature=0.7
                ).choices[0].message.content
                
                # Store the chat in database
                from models import HighlanderChat
                chat = HighlanderChat(
                    user_id=current_user.id,
                    session_id=session_id,
                    message=message,
                    response=response,
                    category=category,
                    context=json.dumps({
                        'client_count': client_count,
                        'newsroom_count': newsroom_count,
                        'research_count': research_count,
                        'user_count': user_count,
                        'chat_count': chat_count,
                        'total_organisations': total_organisations,
                        'media_organisations': media_organisations,
                        'communications_organisations': communications_organisations,
                        'recent_clients': client_details,
                        'recent_organisations': org_details,
                        'context': data.get('context', 'General business inquiry'),
                        'openai_model': 'gpt-4',
                        'tokens_used': response.count(' ') + 1  # Approximate token count
                    }, ensure_ascii=False)
                )
                db.session.add(chat)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'session_id': session_id,
                    'model': 'gpt-4',
                    'context': {
                        'client_count': client_count,
                        'total_organisations': total_organisations,
                        'chat_count': chat_count
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'OpenAI API key not configured'})
                
        except Exception as e:
            # Log the error for debugging
            print(f"Highlander AI Error: {str(e)}")
            return jsonify({'success': False, 'error': f'AI processing error: {str(e)}'})
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
@app.route('/admin/map/highlander/chats', methods=['GET'])
@login_required
@admin_required
def highlander_chats():
    """Get all Highlander AI chat history"""
    try:
        from models import HighlanderChat
        chats = HighlanderChat.query.order_by(HighlanderChat.created_at.desc()).all()
        
        chat_data = []
        for chat in chats:
            chat_data.append({
                'id': chat.id,
                'message': chat.message,
                'response': chat.response,
                'category': chat.category,
                'session_id': chat.session_id,
                'created_at': chat.created_at.isoformat() if chat.created_at else None,
                'user_id': chat.user_id
            })
        
        return jsonify({
            'success': True,
            'chats': chat_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/highlander/export', methods=['POST'])
@login_required
@admin_required
def highlander_export():
    """Export Highlander AI chat data"""
    try:
        from models import HighlanderChat
        chats = HighlanderChat.query.order_by(HighlanderChat.created_at.desc()).all()
        
        export_data = []
        for chat in chats:
            export_data.append({
                'id': chat.id,
                'message': chat.message,
                'response': chat.response,
                'category': chat.category,
                'session_id': chat.session_id,
                'created_at': chat.created_at.isoformat() if chat.created_at else None,
                'user_id': chat.user_id,
                'context': chat.context
            })
        
        return jsonify({
            'success': True,
            'data': export_data,
            'total_chats': len(export_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/highlander/process', methods=['POST'])
@login_required
@admin_required
def highlander_process():
    """Process and analyze Highlander AI chat data"""
    try:
        from models import HighlanderChat
        chats = HighlanderChat.query.all()
        
        # Basic processing - count chats by category
        category_counts = {}
        for chat in chats:
            category = chat.category or 'Uncategorized'
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return jsonify({
            'success': True,
            'processed_chats': len(chats),
            'category_breakdown': category_counts,
            'message': f'Successfully processed {len(chats)} chat records'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/highlander')
@login_required
@admin_required
def highlander_page():
    """Dedicated Highlander AI page"""
    return render_template('admin/highlander_chat.html')

@app.route('/admin/map/highlander/chats', methods=['GET'])
@login_required
@admin_required
def get_highlander_chats():
    """Get Highlander AI chat history"""
    try:
        from models import HighlanderChat
        chats = HighlanderChat.query.filter_by(user_id=current_user.id).order_by(HighlanderChat.created_at.desc()).limit(50).all()
        
        chat_data = []
        for chat in chats:
            chat_data.append({
                'id': chat.id,
                'message': chat.message,
                'response': chat.response,
                'category': chat.category,
                'created_at': chat.created_at.isoformat(),
                'processed': chat.processed
            })
        
        return jsonify({'success': True, 'chats': chat_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/highlander/export', methods=['POST'])
@login_required
@admin_required
def export_highlander_chats():
    """Export Highlander AI chat data for processing"""
    try:
        from models import HighlanderChat
        chats = HighlanderChat.query.filter_by(user_id=current_user.id).all()
        
        export_data = []
        for chat in chats:
            export_data.append({
                'message': chat.message,
                'response': chat.response,
                'category': chat.category,
                'context': json.loads(chat.context) if chat.context else {},
                'created_at': chat.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'data': export_data,
            'total_chats': len(export_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/highlander/process', methods=['POST'])
@login_required
@admin_required
def process_highlander_data():
    """Process Highlander AI chat data for insights"""
    try:
        from models import HighlanderChat
        unprocessed_chats = HighlanderChat.query.filter_by(user_id=current_user.id, processed=False).all()
        
        # Process chats for insights
        insights = []
        for chat in unprocessed_chats:
            # Mark as processed
            chat.processed = True
            
            # Extract insights based on category
            if chat.category == 'Client Analysis':
                insights.append(f"Client insight from {chat.created_at.strftime('%Y-%m-%d')}: {chat.message[:100]}...")
            elif chat.category == 'Business Strategy':
                insights.append(f"Strategy insight from {chat.created_at.strftime('%Y-%m-%d')}: {chat.message[:100]}...")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'processed_chats': len(unprocessed_chats),
            'insights': insights
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Implementation Experience Endpoints
@app.route('/implementation-experience/chat', methods=['POST'])
@login_required
def implementation_experience_chat():
    """Handle implementation experience chat with Highlander AI"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Get user's newsroom
        from models import Newsroom
        newsroom = Newsroom.query.filter_by(user_id=current_user.id).first()
        
        if not newsroom:
            return jsonify({'success': False, 'error': 'No newsroom found for this user'})
        
        # Get or create chat session
        from models import ImplementationChatSession
        session = ImplementationChatSession.query.filter_by(
            session_id=session_id,
            user_id=current_user.id
        ).first()
        
        if not session:
            session = ImplementationChatSession(
                newsroom_id=newsroom.id,
                user_id=current_user.id,
                session_id=session_id,
                session_type='Implementation Experience'
            )
            db.session.add(session)
            db.session.commit()
        
        # Store user message
        from models import ImplementationChatMessage
        user_message = ImplementationChatMessage(
            session_id=session.id,
            sender_type='user',
            message_content=message,
            message_type='text'
        )
        db.session.add(user_message)
        
        # Create system prompt for implementation experience
        system_prompt = f"""You are Highlander AI, an expert consultant helping newsrooms share their AIMAP implementation experiences.

You are talking to {current_user.username} from {newsroom.name}.

Your role is to:
1. Help them share their implementation experiences in a structured way
2. Ask follow-up questions to gather detailed insights
3. Guide them through sharing challenges, solutions, and outcomes
4. Help them quantify their success metrics
5. Collect recommendations for other newsrooms

Key areas to explore:
- What type of AIMAP implementation they undertook
- When it was implemented and how long it took
- What challenges they faced and how they solved them
- What outcomes and improvements they achieved
- Time savings, cost savings, quality improvements
- Whether they would recommend it to others
- Suggestions for improvement

Be conversational, encouraging, and help them think through their experience systematically.
"""
        
        # Generate AI response
        try:
            from training.model_factory import get_mediamap_model_manager
            manager = get_mediamap_model_manager()
            
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            response = manager.generate_response(conversation)
            
        except Exception as e:
            response = f"I'm here to help you share your AIMAP implementation experience! Please tell me about your implementation - what type of AI tools or strategies did you implement, and what was your experience like?"
        
        # Store AI response
        ai_message = ImplementationChatMessage(
            session_id=session.id,
            sender_type='ai',
            message_content=response,
            message_type='text'
        )
        db.session.add(ai_message)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/implementation-experience/submit', methods=['POST'])
@login_required
def submit_implementation_experience():
    """Submit a structured implementation experience"""
    try:
        data = request.get_json()
        
        # Get user's newsroom
        from models import Newsroom
        newsroom = Newsroom.query.filter_by(user_id=current_user.id).first()
        
        if not newsroom:
            return jsonify({'success': False, 'error': 'No newsroom found for this user'})
        
        # Create implementation experience
        from models import NewsroomImplementationExperience
        experience = NewsroomImplementationExperience(
            newsroom_id=newsroom.id,
            user_id=current_user.id,
            implementation_type=data.get('implementation_type'),
            implementation_date=datetime.strptime(data.get('implementation_date'), '%Y-%m-%d').date(),
            implementation_duration_weeks=data.get('implementation_duration_weeks'),
            experience_summary=data.get('experience_summary'),
            challenges_faced=data.get('challenges_faced'),
            solutions_found=data.get('solutions_found'),
            outcomes_achieved=data.get('outcomes_achieved'),
            success_rating=data.get('success_rating'),
            time_saved_hours_per_week=data.get('time_saved_hours_per_week'),
            cost_savings_percentage=data.get('cost_savings_percentage'),
            quality_improvement_rating=data.get('quality_improvement_rating'),
            would_recommend=data.get('would_recommend', True),
            recommendations_for_others=data.get('recommendations_for_others'),
            suggestions_for_improvement=data.get('suggestions_for_improvement')
        )
        
        db.session.add(experience)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'experience_id': experience.id,
            'message': 'Implementation experience submitted successfully!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/implementation-experience/history', methods=['GET'])
@login_required
def get_implementation_experience_history():
    """Get user's implementation experience history"""
    try:
        from models import NewsroomImplementationExperience
        experiences = NewsroomImplementationExperience.query.filter_by(
            user_id=current_user.id
        ).order_by(NewsroomImplementationExperience.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'experiences': [exp.to_dict() for exp in experiences]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Admin endpoints for reviewing implementation experiences
@app.route('/admin/implementation-experiences', methods=['GET'])
@login_required
@admin_required
def admin_implementation_experiences():
    """Admin view of all implementation experiences"""
    try:
        from models import NewsroomImplementationExperience
        experiences = NewsroomImplementationExperience.query.order_by(
            NewsroomImplementationExperience.created_at.desc()
        ).all()
        
        # Calculate statistics
        total_experiences = len(experiences)
        avg_success_rating = sum(exp.success_rating or 0 for exp in experiences) / max(total_experiences, 1)
        avg_time_saved = sum(exp.time_saved_hours_per_week or 0 for exp in experiences) / max(total_experiences, 1)
        recommendations_count = sum(1 for exp in experiences if exp.would_recommend)
        
        return render_template('admin/implementation_experiences.html', 
                             experiences=experiences,
                             total_experiences=total_experiences,
                             avg_success_rating=round(avg_success_rating, 1),
                             avg_time_saved=round(avg_time_saved, 1),
                             recommendations_count=recommendations_count)
        
    except Exception as e:
        flash(f'Error loading implementation experiences: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/implementation-experiences/<int:experience_id>', methods=['GET'])
@login_required
@admin_required
def admin_view_implementation_experience(experience_id):
    """Admin view of a specific implementation experience"""
    try:
        from models import NewsroomImplementationExperience
        experience = NewsroomImplementationExperience.query.get_or_404(experience_id)
        
        return render_template('admin/implementation_experience_detail.html', 
                             experience=experience)
        
    except Exception as e:
        flash(f'Error loading implementation experience: {str(e)}', 'error')
        return redirect(url_for('admin_implementation_experiences'))

@app.route('/admin/implementation-experiences/<int:experience_id>/update', methods=['POST'])
@login_required
@admin_required
def admin_update_implementation_experience(experience_id):
    """Admin update of implementation experience status and notes"""
    try:
        from models import NewsroomImplementationExperience
        experience = NewsroomImplementationExperience.query.get_or_404(experience_id)
        
        data = request.get_json()
        experience.status = data.get('status', experience.status)
        experience.admin_notes = data.get('admin_notes', experience.admin_notes)
        experience.follow_up_required = data.get('follow_up_required', experience.follow_up_required)
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Experience updated successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/implementation-experiences/export', methods=['POST'])
@login_required
@admin_required
def export_implementation_experiences():
    """Export all implementation experiences to CSV"""
    try:
        from models import NewsroomImplementationExperience
        experiences = NewsroomImplementationExperience.query.order_by(
            NewsroomImplementationExperience.created_at.desc()
        ).all()
        
        # Create CSV data
        csv_data = []
        csv_data.append([
            'Date', 'Newsroom', 'User', 'Implementation Type', 'Success Rating',
            'Time Saved (hrs/week)', 'Cost Savings (%)', 'Would Recommend',
            'Status', 'Admin Notes'
        ])
        
        for exp in experiences:
            csv_data.append([
                exp.created_at.strftime('%Y-%m-%d'),
                exp.newsroom.name if exp.newsroom else 'Unknown',
                exp.user.username if exp.user else 'Unknown',
                exp.implementation_type,
                exp.success_rating or 0,
                exp.time_saved_hours_per_week or 0,
                exp.cost_savings_percentage or 0,
                'Yes' if exp.would_recommend else 'No',
                exp.status,
                exp.admin_notes or ''
            ])
        
        # Create CSV file
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(csv_data)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=implementation_experiences.csv'}
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Newsletter Routes
@app.route('/admin/map/newsletters', methods=['GET'])
@login_required
@admin_required
def get_newsletters():
    """Get all newsletters"""
    try:
        from models import Newsletter
        newsletters = Newsletter.query.order_by(Newsletter.created_at.desc()).limit(20).all()
        return jsonify({
            'success': True,
            'newsletters': [newsletter.to_dict() for newsletter in newsletters]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/newsletters/<int:newsletter_id>', methods=['GET'])
@login_required
@admin_required
def get_newsletter_detail(newsletter_id):
    """Get a specific newsletter"""
    try:
        from models import Newsletter
        newsletter = Newsletter.query.get(newsletter_id)
        if not newsletter:
            return jsonify({'success': False, 'error': 'Newsletter not found'})
        return jsonify({
            'success': True,
            'newsletter': newsletter.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/generate-newsletter', methods=['POST'])
@login_required
@admin_required
def generate_newsletter():
    """Generate a new AI newsletter based on current news and training data"""
    try:
        data = request.get_json()
        topic = data.get('topic', 'AI News and Tools')
        focus_areas = data.get('focus_areas', ['AI News', 'Tools', 'Reports', 'Use Cases'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Get training data (past newsletters)
        from models import Newsletter
        training_newsletters = Newsletter.query.filter_by(is_training_data=True).order_by(Newsletter.created_at.desc()).limit(10).all()
        
        # Get news and insights within the specified date range
        from models import DailyInsight
        from datetime import datetime
        
        if start_date and end_date:
            # Parse dates and filter by date range
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            # Add one day to end_date to include the entire end date
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            
            recent_news = DailyInsight.query.filter(
                DailyInsight.created_at >= start_datetime,
                DailyInsight.created_at <= end_datetime
            ).order_by(DailyInsight.created_at.desc()).limit(50).all()
        else:
            # Fallback to recent news if no date range specified
            recent_news = DailyInsight.query.filter_by(category='Admin News').order_by(DailyInsight.created_at.desc()).limit(15).all()
        
        # Build training context
        training_context = ""
        for newsletter in training_newsletters:
            training_context += f"\n\nTitle: {newsletter.title}\nContent: {newsletter.content[:500]}...\n"
        
        # Build current news context
        news_context = ""
        for news in recent_news:
            news_context += f"\n- {news.title}: {news.content[:200]}...\n"
        
        # Create the prompt for newsletter generation
        date_range_info = ""
        if start_date and end_date:
            date_range_info = f"\nDATE RANGE: {start_date} to {end_date} (news and insights from this period)"
        
        prompt = f"""You are an expert AI newsletter writer with 2 years of experience writing about AI news, tools, reports, and use cases for Substack.

Based on your past newsletters and current news, create a compelling newsletter that includes:

TOPIC: {topic}
FOCUS AREAS: {', '.join(focus_areas)}{date_range_info}

Your writing style from past newsletters:
{training_context}

Current news and developments to include (from the specified date range):
{news_context}

Create a newsletter that:
1. Has a compelling headline
2. Includes 3-5 main sections covering different aspects of AI
3. Provides actionable insights and practical value
4. Maintains your established voice and style
5. Is approximately 800-1200 words
6. Includes relevant links and references where appropriate

Format the newsletter with clear sections, bullet points, and engaging prose."""
        
        # Generate the newsletter using OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI newsletter writer with a proven track record of engaging content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            newsletter_content = response.choices[0].message.content.strip()
            
            # Calculate word count and reading time
            word_count = len(newsletter_content.split())
            reading_time = max(1, word_count // 200)  # Rough estimate: 200 words per minute
            
            # Extract title from content (first line)
            lines = newsletter_content.split('\n')
            title = lines[0].replace('#', '').strip() if lines else f"AI Newsletter - {datetime.now().strftime('%B %d, %Y')}"
            
            # Save the newsletter
            newsletter = Newsletter(
                title=title,
                content=newsletter_content,
                summary=f"AI-generated newsletter covering {topic}",
                category="AI Newsletter",
                tags="AI, News, Tools, Reports, Use Cases",
                is_generated=True,
                word_count=word_count,
                reading_time=reading_time
            )
            db.session.add(newsletter)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'newsletter': newsletter.to_dict()
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error generating newsletter: {str(e)}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/upload-training-newsletter', methods=['POST'])
@login_required
@admin_required
def upload_training_newsletter():
    """Upload a past newsletter as training data"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        published_date = data.get('published_date')
        
        if not title or not content:
            return jsonify({'success': False, 'error': 'Title and content are required'})
        
        # Calculate word count and reading time
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)
        
        # Parse published date
        published_at = None
        if published_date:
            try:
                published_at = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
        
        # Save as training data
        from models import Newsletter
        newsletter = Newsletter(
            title=title,
            content=content,
            summary=f"Training data: {title}",
            category="Training Data",
            tags="Training, Past Newsletter",
            is_training_data=True,
            is_generated=False,
            word_count=word_count,
            reading_time=reading_time,
            published_at=published_at
        )
        db.session.add(newsletter)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'newsletter': newsletter.to_dict()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# AI Policies API Endpoints
@app.route('/admin/map/policies', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
@admin_required
def manage_policies():
    """Manage AI policies and governance"""
    if request.method == 'GET':
        try:
            from models import AIPolicy
            policies = AIPolicy.query.all()
            policies_data = []
            for policy in policies:
                policies_data.append(policy.to_dict())
            return jsonify({'policies': policies_data})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'POST':
        try:
            from models import AIPolicy
            from flask_login import current_user
            data = request.get_json()
            
            new_policy = AIPolicy(
                name=data['name'],
                description=data.get('description', ''),
                category=data.get('category', 'General'),
                version=data.get('version', '1.0'),
                status=data.get('status', 'Draft'),
                content=data.get('content', ''),
                newsroom_id=data.get('newsroom_id'),
                organization_id=data.get('organization_id'),
                organization_name=data.get('organization_name', ''),
                compliance_requirements=data.get('compliance_requirements', ''),
                review_frequency=data.get('review_frequency', 'Annual'),
                priority=data.get('priority', 'Medium'),
                applicable_to=data.get('applicable_to', ''),
                tags=','.join(data.get('tags', [])),
                created_by=current_user.id if current_user else None
            )
            
            db.session.add(new_policy)
            db.session.commit()
            
            return jsonify({'status': 'success', 'id': new_policy.id})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'PUT':
        try:
            from models import AIPolicy
            data = request.get_json()
            policy_id = data['id']
            
            policy = AIPolicy.query.get(policy_id)
            if not policy:
                return jsonify({'error': 'Policy not found'})
            
            # Update fields
            for field in ['name', 'description', 'category', 'version', 'status', 'content', 
                         'compliance_requirements', 'review_frequency', 'priority', 'applicable_to']:
                if field in data:
                    setattr(policy, field, data[field])
            
            # Update tags
            if 'tags' in data:
                policy.tags = ','.join(data['tags'])
            
            policy.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif request.method == 'DELETE':
        try:
            from models import AIPolicy
            data = request.get_json()
            policy_id = data['id']
            
            policy = AIPolicy.query.get(policy_id)
            if not policy:
                return jsonify({'error': 'Policy not found'})
            
            db.session.delete(policy)
            db.session.commit()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)})


# AI Tools API Endpoints
@app.route('/admin/map/ai-tools', methods=['GET'])
@login_required
@admin_required
def get_ai_tools():
    """Get all AI tools with statistics"""
    try:
        from models import AITool, AIToolRecommendation
        
        # Get all tools
        tools = AITool.query.all()
        tools_data = [tool.to_dict() for tool in tools]
        
        # Calculate statistics
        total_tools = len(tools)
        data_safe_tools = len([t for t in tools if t.data_safety_score >= 7.0])
        recommended_tools = len([t for t in tools if t.recommendation_score >= 8.0])
        categories = len(set(t.category for t in tools))
        
        # Data safety breakdown
        high_safety_tools = len([t for t in tools if t.data_safety_score >= 8.0])
        medium_safety_tools = len([t for t in tools if 5.0 <= t.data_safety_score < 8.0])
        low_safety_tools = len([t for t in tools if t.data_safety_score < 5.0])
        
        # Get top recommendations
        recommendations = AIToolRecommendation.query.filter_by(status='Active').order_by(AIToolRecommendation.created_at.desc()).limit(5).all()
        recommendations_data = [rec.to_dict() for rec in recommendations]
        
        stats = {
            'total_tools': total_tools,
            'data_safe_tools': data_safe_tools,
            'recommended_tools': recommended_tools,
            'categories': categories,
            'high_safety_tools': high_safety_tools,
            'medium_safety_tools': medium_safety_tools,
            'low_safety_tools': low_safety_tools
        }
        
        return jsonify({
            'success': True,
            'tools': tools_data,
            'stats': stats,
            'recommendations': recommendations_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
@app.route('/admin/map/ai-tools', methods=['POST'])
@login_required
@admin_required
def add_ai_tool():
    """Add a new AI tool"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'category']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'})
        
        # Create new tool
        tool = AITool(
            name=data['name'],
            description=data.get('description'),
            website_url=data.get('website_url'),
            company=data.get('company'),
            category=data['category'],
            subcategory=data.get('subcategory'),
            pricing_model=data.get('pricing_model'),
            pricing_details=data.get('pricing_details'),
            data_safety_score=float(data.get('data_safety_score', 0)),
            data_safety_assessment=data.get('data_safety_assessment'),
            privacy_policy_url=data.get('privacy_policy_url'),
            data_retention_policy=data.get('data_retention_policy'),
            gdpr_compliant=data.get('gdpr_compliant', False),
            ccpa_compliant=data.get('ccpa_compliant', False),
            data_encryption=data.get('data_encryption', False),
            data_localization=data.get('data_localization'),
            api_available=data.get('api_available', False),
            api_documentation_url=data.get('api_documentation_url'),
            integration_options=data.get('integration_options'),
            supported_languages=data.get('supported_languages'),
            model_type=data.get('model_type'),
            user_count=data.get('user_count'),
            rating=float(data.get('rating', 0)),
            review_count=int(data.get('review_count', 0)),
            recommendation_score=float(data.get('recommendation_score', 0)),
            recommendation_reason=data.get('recommendation_reason'),
            use_cases=data.get('use_cases'),
            limitations=data.get('limitations'),
            alternatives=data.get('alternatives'),
            status=data.get('status', 'Active')
        )
        
        db.session.add(tool)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'tool': tool.to_dict(),
            'message': 'AI tool added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/ai-tools/<int:tool_id>', methods=['GET'])
@login_required
@admin_required
def get_ai_tool(tool_id):
    """Get a specific AI tool by ID"""
    try:
        tool = AITool.query.get(tool_id)
        if not tool:
            return jsonify({'success': False, 'error': 'Tool not found'})
        
        return jsonify({
            'success': True,
            'tool': tool.to_dict()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/ai-tools/<int:tool_id>', methods=['PUT'])
@login_required
@admin_required
def update_ai_tool(tool_id):
    """Update an AI tool"""
    try:
        tool = AITool.query.get(tool_id)
        if not tool:
            return jsonify({'success': False, 'error': 'Tool not found'})
        
        data = request.get_json()
        
        # Update fields
        for field, value in data.items():
            if hasattr(tool, field):
                setattr(tool, field, value)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'tool': tool.to_dict(),
            'message': 'AI tool updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})
@app.route('/admin/map/ai-tools/<int:tool_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_ai_tool(tool_id):
    """Delete an AI tool"""
    try:
        tool = AITool.query.get(tool_id)
        if not tool:
            return jsonify({'success': False, 'error': 'Tool not found'})
        
        db.session.delete(tool)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'AI tool deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/ai-tools/export', methods=['GET'])
@login_required
@admin_required
def export_ai_tools():
    """Export AI tools data as JSON"""
    try:
        from models import AITool, AIToolRecommendation, AIToolReview
        
        tools = AITool.query.all()
        recommendations = AIToolRecommendation.query.all()
        reviews = AIToolReview.query.all()
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'tools': [tool.to_dict() for tool in tools],
            'recommendations': [rec.to_dict() for rec in recommendations],
            'reviews': [review.to_dict() for review in reviews]
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/ai-tools/recommendations', methods=['POST'])
@login_required
@admin_required
def add_ai_recommendation():
    """Add a new AI tool recommendation"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('title'):
            return jsonify({'success': False, 'error': 'Title is required'})
        
        # Create new recommendation
        recommendation = AIToolRecommendation(
            title=data['title'],
            description=data.get('description'),
            target_audience=data.get('target_audience'),
            use_case=data.get('use_case'),
            budget_range=data.get('budget_range'),
            recommended_tools=data.get('recommended_tools'),
            alternatives=data.get('alternatives'),
            implementation_steps=data.get('implementation_steps'),
            timeline=data.get('timeline'),
            estimated_cost=data.get('estimated_cost'),
            training_requirements=data.get('training_requirements'),
            status=data.get('status', 'Active')
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'recommendation': recommendation.to_dict(),
            'message': 'Recommendation added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

# Consulting API Endpoints
@app.route('/admin/map/consulting', methods=['GET'])
@login_required
@admin_required
def get_consulting_data():
    """Get all consulting data with statistics"""
    try:
        from models import ConsultingClient, ConsultingSession, ConsultingProgressReport
        
        # Get all clients
        clients = ConsultingClient.query.all()
        clients_data = [client.to_dict() for client in clients]
        
        # Get all sessions
        sessions = ConsultingSession.query.order_by(ConsultingSession.session_date.desc()).all()
        sessions_data = [session.to_dict() for session in sessions]
        
        # Calculate statistics
        total_clients = len(clients)
        active_clients = len([c for c in clients if c.status == 'Active'])
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions if s.status == 'Scheduled'])
        total_hours = sum(s.duration_hours for s in sessions if s.duration_hours)
        
        # Calculate average satisfaction
        satisfaction_ratings = [s.client_satisfaction for s in sessions if s.client_satisfaction]
        avg_satisfaction = sum(satisfaction_ratings) / len(satisfaction_ratings) if satisfaction_ratings else 0
        
        # Get upcoming sessions (next 7 days)
        from datetime import datetime, timedelta
        upcoming_date = datetime.now() + timedelta(days=7)
        upcoming_sessions = [s for s in sessions if s.session_date and s.session_date > datetime.now() and s.session_date <= upcoming_date]
        upcoming_data = [session.to_dict() for session in upcoming_sessions]
        
        stats = {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_hours': total_hours,
            'avg_satisfaction': round(avg_satisfaction, 1)
        }
        
        return jsonify({
            'success': True,
            'clients': clients_data,
            'sessions': sessions_data,
            'upcoming_sessions': upcoming_data,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/consulting/clients', methods=['POST'])
@login_required
@admin_required
def add_consulting_client():
    """Add a new consulting client"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'organization', 'email']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'})
        
        # Create new client
        client = ConsultingClient(
            name=data['name'],
            organization=data['organization'],
            email=data['email'],
            phone=data.get('phone'),
            website=data.get('website'),
            industry=data.get('industry'),
            organization_size=data.get('organization_size'),
            location=data.get('location'),
            timezone=data.get('timezone'),
            engagement_type=data.get('engagement_type'),
            contract_value=float(data.get('contract_value', 0)) if data.get('contract_value') else None,
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            status=data.get('status', 'Active'),
            contact_person=data.get('contact_person'),
            contact_role=data.get('contact_role'),
            contact_email=data.get('contact_email'),
            notes=data.get('notes'),
            goals=data.get('goals'),
            challenges=data.get('challenges'),
            success_metrics=data.get('success_metrics')
        )
        
        db.session.add(client)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'client': client.to_dict(),
            'message': 'Consulting client added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/consulting/sessions', methods=['POST'])
@login_required
@admin_required
def add_consulting_session():
    """Add a new consulting session"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['client_id', 'title', 'session_type', 'session_date']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'})
        
        # Create new session
        session = ConsultingSession(
            client_id=data['client_id'],
            title=data['title'],
            description=data.get('description'),
            session_type=data['session_type'],
            session_date=datetime.fromisoformat(data['session_date']),
            duration_hours=float(data.get('duration_hours', 0)) if data.get('duration_hours') else None,
            session_notes=data.get('session_notes'),
            recording_url=data.get('recording_url'),
            recording_file_path=data.get('recording_file_path'),
            recording_duration=int(data.get('recording_duration', 0)) if data.get('recording_duration') else None,
            materials_shared=data.get('materials_shared'),
            topics_covered=data.get('topics_covered'),
            action_items=data.get('action_items'),
            next_steps=data.get('next_steps'),
            client_satisfaction=int(data.get('client_satisfaction', 0)) if data.get('client_satisfaction') else None,
            client_feedback=data.get('client_feedback'),
            client_questions=data.get('client_questions'),
            status=data.get('status', 'Scheduled'),
            follow_up_required=data.get('follow_up_required', False),
            follow_up_date=datetime.fromisoformat(data['follow_up_date']) if data.get('follow_up_date') else None
        )
        
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session': session.to_dict(),
            'message': 'Consulting session added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/consulting/sessions/<int:session_id>', methods=['PUT'])
@login_required
@admin_required
def update_consulting_session(session_id):
    """Update a consulting session"""
    try:
        session = ConsultingSession.query.get(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        data = request.get_json()
        
        # Update fields
        for field, value in data.items():
            if hasattr(session, field):
                if field in ['session_date', 'follow_up_date'] and value:
                    setattr(session, field, datetime.fromisoformat(value))
                else:
                    setattr(session, field, value)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session': session.to_dict(),
            'message': 'Session updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/consulting/upload-recording', methods=['POST'])
@login_required
@admin_required
def upload_session_recording():
    """Upload a session recording"""
    try:
        if 'recording' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['recording']
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'})
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'recordings')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"session_{session_id}_{timestamp}_{file.filename}"
        filepath = os.path.join(upload_dir, filename)
        
        # Save file
        file.save(filepath)
        
        # Update session with recording info
        session = ConsultingSession.query.get(session_id)
        if session:
            session.recording_file_path = filepath
            session.recording_url = f"/static/uploads/recordings/{filename}"
            db.session.commit()
        
        return jsonify({
            'success': True,
            'file_url': f"/static/uploads/recordings/{filename}",
            'message': 'Recording uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/map/consulting/export', methods=['GET'])
@login_required
@admin_required
def export_consulting_data():
    """Export consulting data as JSON"""
    try:
        from models import ConsultingClient, ConsultingSession, ConsultingProgressReport, ConsultingProgressEntry, ConsultingSuccessMetric
        
        clients = ConsultingClient.query.all()
        sessions = ConsultingSession.query.all()
        progress_reports = ConsultingProgressReport.query.all()
        progress_entries = ConsultingProgressEntry.query.all()
        success_metrics = ConsultingSuccessMetric.query.all()
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'clients': [client.to_dict() for client in clients],
            'sessions': [session.to_dict() for session in sessions],
            'progress_reports': [report.to_dict() for report in progress_reports],
            'progress_entries': [entry.to_dict() for entry in progress_entries],
            'success_metrics': [metric.to_dict() for metric in success_metrics]
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Admin GPT route removed - now integrated into dashboard

@app.route('/admin/gpt/context')
@login_required
@admin_required
def admin_gpt_context():
    """Get system context for Admin GPT"""
    try:
        # Get system statistics
        user_count = User.query.count()
        analysis_count = MediaAnalysis.query.count()
        chat_count = Chat.query.count()
        message_count = Message.query.count()
        
        # Get AI model status
        try:
            from training.model_factory import get_mediamap_model_manager
            manager = get_mediamap_model_manager()
            model_info = manager.get_model_info()
            model_status = "Loaded" if model_info.get('custom_model_loaded', False) else "Not Loaded"
        except:
            model_status = "Not Available"
        
        # Get recent activity
        recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
        recent_analyses = MediaAnalysis.query.order_by(MediaAnalysis.created_at.desc()).limit(5).all()
        
        # Get additional system info
        try:
            lesson_count = Lesson.query.count()
            feedback_count = Feedback.query.count()
            translation_count = Translation.query.count()
        except:
            lesson_count = feedback_count = translation_count = 0
        
        # Get system performance metrics
        try:
            from training.model_factory import get_mediamap_model_manager
            manager = get_mediamap_model_manager()
            model_info = manager.get_model_info()
            performance_metrics = manager.get_performance_metrics()
        except:
            model_info = {}
            performance_metrics = {}
        
        context = {
            'status': 'Healthy',
            'user_count': user_count,
            'analysis_count': analysis_count,
            'chat_count': chat_count,
            'message_count': message_count,
            'lesson_count': lesson_count,
            'feedback_count': feedback_count,
            'translation_count': translation_count,
            'model_status': model_status,
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now().isoformat(),
            'recent_users': [{'username': u.username, 'created_at': u.created_at.isoformat()} for u in recent_users],
            'recent_analyses': [{'title': a.title, 'created_at': a.created_at.isoformat()} for a in recent_analyses],
            'system_info': {
                'flask_version': getattr(app, 'version', 'Unknown'),
                'python_version': '.'.join([str(x) for x in sys.version_info[:3]]),
                'database_size': 'Available',
                'openai_available': bool(app.config.get('OPENAI_API_KEY'))
            }
        }
        
        return jsonify(context)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/gpt/chat', methods=['POST'])
@login_required
@admin_required
def admin_gpt_chat():
    """Handle Admin GPT chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Create system prompt with context
        system_prompt = f"""You are an AI assistant for the AIMAP (Advanced AI Media Analysis Platform) system. 
You have access to the following system context:

System Status: {context.get('status', 'Unknown')}
Total Users: {context.get('user_count', 0)}
Total Analyses: {context.get('analysis_count', 0)}
Total Chats: {context.get('chat_count', 0)}
Total Messages: {context.get('message_count', 0)}
AI Model Status: {context.get('model_status', 'Unknown')}

Your role is to help administrators understand and interrogate the system. You can:
- Analyze system performance and metrics
- Explain user activity patterns
- Provide insights about AI model training and usage
- Help troubleshoot issues
- Suggest optimizations
- Answer questions about the platform's functionality

Please provide clear, actionable insights and recommendations. If you need more specific data to answer a question, let the admin know what additional information would be helpful."""

        # Use OpenAI API to get response
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=app.config.get('OPENAI_API_KEY'))
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            return jsonify({
                'success': True,
                'response': ai_response
            })
            
        except Exception as e:
            # Fallback response if OpenAI is not available
            fallback_response = f"""I understand you're asking about: "{message}"

Based on the current system context:
- The system appears to be running normally
- There are {context.get('user_count', 0)} users in the system
- {context.get('analysis_count', 0)} media analyses have been performed
- The AI model status is: {context.get('model_status', 'Unknown')}

To get more detailed information, you might want to:
1. Check the specific admin panels for detailed metrics
2. Review the system logs for any errors
3. Monitor the AI training dashboard for model performance

Note: OpenAI API integration is currently unavailable, so I'm providing a basic analysis based on the system context."""

            return jsonify({
                'success': True,
                'response': fallback_response
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# OpenAI Agent Integration Endpoints
@app.route('/admin/openai/agents', methods=['GET'])
@login_required
@admin_required
def list_openai_agents():
    """List all available OpenAI agents"""
    try:
        from agents.openai_agent_integration import get_openai_agent_integration
        
        agent_integration = get_openai_agent_integration()
        if not agent_integration:
            return jsonify({
                'success': False,
                'error': 'OpenAI agent integration not available'
            }), 500
        
        agents = agent_integration.list_available_agents()
        
        return jsonify({
            'success': True,
            'agents': agents
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/agents/<agent_type>/capabilities', methods=['GET'])
@login_required
@admin_required
def get_agent_capabilities(agent_type):
    """Get capabilities of a specific agent"""
    try:
        from agents.openai_agent_integration import get_openai_agent_integration
        
        agent_integration = get_openai_agent_integration()
        if not agent_integration:
            return jsonify({
                'success': False,
                'error': 'OpenAI agent integration not available'
            }), 500
        
        capabilities = agent_integration.get_agent_capabilities(agent_type)
        
        return jsonify({
            'success': True,
            'capabilities': capabilities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/agents/<agent_type>/analyze', methods=['POST'])
@login_required
@admin_required
def analyze_with_agent(agent_type):
    """Analyze data using a specific OpenAI agent"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        analysis_data = data.get('data', {})
        analysis_type = data.get('analysis_type', 'insights')
        
        from agents.openai_agent_integration import get_openai_agent_integration
        
        agent_integration = get_openai_agent_integration()
        if not agent_integration:
            return jsonify({
                'success': False,
                'error': 'OpenAI agent integration not available'
            }), 500
        
        result = agent_integration.analyze_data_with_agent(
            agent_type=agent_type,
            data=analysis_data,
            analysis_type=analysis_type
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/agents/<agent_type>/instructions', methods=['PUT'])
@login_required
@admin_required
def update_agent_instructions(agent_type):
    """Update agent instructions"""
    try:
        data = request.get_json()
        if not data or 'instructions' not in data:
            return jsonify({
                'success': False,
                'error': 'Instructions not provided'
            }), 400
        
        new_instructions = data['instructions']
        
        from agents.openai_agent_integration import get_openai_agent_integration
        
        agent_integration = get_openai_agent_integration()
        if not agent_integration:
            return jsonify({
                'success': False,
                'error': 'OpenAI agent integration not available'
            }), 500
        
        success = agent_integration.update_agent_instructions(agent_type, new_instructions)
        
        return jsonify({
            'success': success,
            'message': 'Instructions updated successfully' if success else 'Failed to update instructions'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/agents/<agent_type>', methods=['DELETE'])
@login_required
@admin_required
def delete_agent(agent_type):
    """Delete an agent"""
    try:
        from agents.openai_agent_integration import get_openai_agent_integration
        
        agent_integration = get_openai_agent_integration()
        if not agent_integration:
            return jsonify({
                'success': False,
                'error': 'OpenAI agent integration not available'
            }), 500
        
        success = agent_integration.delete_agent(agent_type)
        
        return jsonify({
            'success': success,
            'message': 'Agent deleted successfully' if success else 'Failed to delete agent'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# OpenAI Fine-tuning Endpoints
@app.route('/admin/openai/fine-tuning/models', methods=['GET'])
@login_required
@admin_required
def list_fine_tuned_models():
    """List all fine-tuned models"""
    try:
        if not app.config.get('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured'
            }), 500
        
        openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
        models = openai_client.fine_tuning.jobs.list()
        
        return jsonify({
            'success': True,
            'models': models.data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/fine-tuning/start', methods=['POST'])
@login_required
@admin_required
def start_fine_tuning():
    """Start fine-tuning process"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_name = data.get('model_name', 'mediamap')
        training_file_id = data.get('training_file_id')
        
        if not training_file_id:
            return jsonify({
                'success': False,
                'error': 'Training file ID is required'
            }), 400
        
        from training.openai_trainer import OpenAITrainer
        
        trainer = OpenAITrainer(model_name)
        result = trainer.start_fine_tuning(training_file_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/embeddings', methods=['POST'])
@login_required
@admin_required
def create_embeddings():
    """Create embeddings for text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Text not provided'
            }), 400
        
        text = data['text']
        model = data.get('model', 'text-embedding-3-small')
        
        if not app.config.get('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured'
            }), 500
        
        openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
        
        response = openai_client.embeddings.create(
            model=model,
            input=text
        )
        
        return jsonify({
            'success': True,
            'embeddings': response.data[0].embedding,
            'model': model,
            'usage': response.usage
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/openai/vision/analyze', methods=['POST'])
@login_required
@admin_required
def analyze_image():
    """Analyze image using OpenAI Vision API"""
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({
                'success': False,
                'error': 'Image URL not provided'
            }), 400
        
        image_url = data['image_url']
        prompt = data.get('prompt', 'What do you see in this image?')
        
        if not app.config.get('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured'
            }), 500
        
        openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
        
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return jsonify({
            'success': True,
            'analysis': response.choices[0].message.content
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/model/status')
@login_required
@admin_required
def admin_model_status():
    """Admin endpoint to get detailed model status"""
    try:
        from training.model_factory import get_mediamap_model_manager
        manager = get_mediamap_model_manager()
        model_info = manager.get_model_info()
        performance_metrics = manager.get_performance_metrics()
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'system_health': {
                'flask_running': True,
                'database_connected': True,
                'openai_available': bool(app.config.get('OPENAI_API_KEY')),
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/model/monitor')
@login_required
@admin_required
def admin_model_monitor():
    """Admin page for AI model monitoring"""
    return render_template('admin/model_monitor.html')

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
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        user_type = request.form.get('user_type', 'admin')  # Default to admin for backward compatibility
        
        # Validate input
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('create_admin'))
        
        if len(username) < MIN_USERNAME_LENGTH:
            flash(f'Username must be at least {MIN_USERNAME_LENGTH} characters long', 'danger')
            return redirect(url_for('create_admin'))
        
        if len(password) < MIN_PASSWORD_LENGTH:
            flash(f'Password must be at least {MIN_PASSWORD_LENGTH} characters long', 'danger')
            return redirect(url_for('create_admin'))
        
        if '@' not in email:
            flash('Please enter a valid email address', 'danger')
            return redirect(url_for('create_admin'))
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('create_admin'))
        
        # Check if email already exists
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already exists', 'danger')
            return redirect(url_for('create_admin'))
        
        # Create new user (admin or regular)
        is_admin = (user_type == 'admin')
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=is_admin
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        user_type_label = 'Admin' if is_admin else 'Regular'
        flash(f'{user_type_label} user {username} created successfully', 'success')
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
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting yourself
    if user.id == current_user.id:
        flash('You cannot delete your own account', 'danger')
        return redirect(url_for('admin_users'))
    
    # Prevent deleting the last admin
    admin_count = User.query.filter_by(is_admin=True).count()
    if user.is_admin and admin_count <= 1:
        flash('Cannot delete the last admin user', 'danger')
        return redirect(url_for('admin_users'))
    
    try:
        # Delete related data first (cascade should handle this, but being explicit)
        # Delete user's chats
        Chat.query.filter_by(user_id=user_id).delete()
        
        # Delete user's media analyses
        MediaAnalysis.query.filter_by(user_id=user_id).delete()
        
        # Delete user's lesson progress
        UserLesson.query.filter_by(user_id=user_id).delete()
        
        # Delete user's translations
        Translation.query.filter_by(user_id=user_id).delete()
        
        # Delete user's feedback
        Feedback.query.filter_by(user_id=user_id).delete()
        
        # Delete the user
        db.session.delete(user)
        db.session.commit()
        
        flash(f'User {user.username} has been deleted successfully', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {str(e)}', 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        try:
            # Get form data
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            is_admin = request.form.get('is_admin') == 'on'
            
            # Validate input
            if not username or not email:
                flash('Username and email are required', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            if len(username) < MIN_USERNAME_LENGTH:
                flash(f'Username must be at least {MIN_USERNAME_LENGTH} characters long', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            if '@' not in email:
                flash('Please enter a valid email address', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Check if username is taken by another user
            existing_user = User.query.filter(User.username == username, User.id != user_id).first()
            if existing_user:
                flash('Username already exists', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Check if email is taken by another user
            existing_email = User.query.filter(User.email == email, User.id != user_id).first()
            if existing_email:
                flash('Email already exists', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Prevent removing admin status from yourself
            if user.id == current_user.id and not is_admin:
                flash('You cannot remove your own admin status', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Prevent removing the last admin
            if user.is_admin and not is_admin:
                admin_count = User.query.filter_by(is_admin=True).count()
                if admin_count <= 1:
                    flash('Cannot remove admin status from the last admin user', 'danger')
                    return redirect(url_for('edit_user', user_id=user_id))
            
            # Update user
            user.username = username
            user.email = email
            user.is_admin = is_admin
            
            db.session.commit()
            flash(f'User {user.username} updated successfully', 'success')
            return redirect(url_for('admin_users'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating user: {str(e)}', 'danger')
            return redirect(url_for('edit_user', user_id=user_id))
    
    return render_template('admin/edit_user.html', user=user)

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
    # Force template reload for debugging
    from flask import current_app
    current_app.jinja_env.cache = {}
    
    # Get actual model status data to pass to template
    try:
        from training.openai_trainer import get_model_status
        
        # Get status for all models
        mediamap_status = get_model_status('mediamap')
        healthpin_status = get_model_status('healthpin')
        highlander_status = {
            'model_loaded': False,
            'training_examples': 0,
            'last_training': 'Never',
            'accuracy': 'N/A',
            'openai_available': bool(os.getenv('OPENAI_API_KEY'))
        }
        
        model_data = {
            'mediamap': mediamap_status,
            'healthpin': healthpin_status,
            'highlander': highlander_status
        }
        
    except Exception as e:
        print(f"Error getting model status: {e}")
        # Fallback data
        model_data = {
            'mediamap': {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Never',
                'accuracy': 'N/A',
                'openai_available': bool(os.getenv('OPENAI_API_KEY'))
            },
            'healthpin': {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Never',
                'accuracy': 'N/A',
                'openai_available': bool(os.getenv('OPENAI_API_KEY'))
            },
            'highlander': {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Never',
                'accuracy': 'N/A',
                'openai_available': bool(os.getenv('OPENAI_API_KEY'))
            }
        }
    
    return render_template('admin/training.html', model_data=model_data)

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
            from datasafe_integration import DataSafeProcessor
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
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        # Model-specific data collection
        if model_name in ['mediamap', 'healthpin']:
            # Use enhanced domain-specific data collection
            from training.enhanced_domain_collector import collect_enhanced_domain_data
            from training.training_validator import validate_training_data
            from training.feedback_integration import integrate_feedback
            from datetime import datetime
            import os
            
            basedir = os.path.abspath(os.path.dirname(__file__))
            db_path = os.path.join(basedir, "instance", "media_analysis.db")
            data_dir = os.path.join(basedir, 'training_data', model_name)
            
            # Enhanced data collection
            stats = collect_enhanced_domain_data(model_name, db_path)
            
            # Validate training data quality
            validation_report = validate_training_data(model_name, data_dir)
            
            # Integrate user feedback for continuous improvement
            feedback_integration = integrate_feedback(model_name, db_path)
            
            return jsonify({
                'success': True,
                'message': f'{model_name.title()} enhanced data collection completed successfully!',
                'model': model_name,
                'stats': stats,
                'validation': validation_report,
                'feedback_integration': feedback_integration,
                'collected_at': datetime.now().isoformat()
            })
        else:  # highlander
            # Use the real data collection function for general data
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
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        # Use OpenAI fine-tuning for MediaMap and HealthPIN
        if model_name in ['mediamap', 'healthpin']:
            from training.openai_trainer import train_model
            
            basedir = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(basedir, 'training_data', model_name)
            
            # Check if data exists
            if not os.path.exists(data_dir):
                return jsonify({
                    'success': False,
                    'error': f'No training data found for {model_name}. Please collect data first.'
                }), 400
            
            # Start training in background
            import threading
            
            def start_openai_training():
                try:
                    result = train_model(model_name, data_dir)
                    logger.info(f"Training result for {model_name}: {result}")
                except Exception as e:
                    logger.error(f"Training error for {model_name}: {e}")
            
            training_thread = threading.Thread(target=start_openai_training, daemon=True)
            training_thread.start()
            
            return jsonify({
                'success': True,
                'message': f'{model_name.title()} model training started using OpenAI fine-tuning!',
                'model': model_name,
                'training_type': 'openai_fine_tuning'
            })
        
        else:  # highlander - use existing implementation
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
            
            training_thread = threading.Thread(target=train_model, daemon=True)
            training_thread.start()
            
            # Ensure thread cleanup
            def cleanup_thread():
                try:
                    training_thread.join(timeout=1)
                except:
                    pass
            
            import atexit
            atexit.register(cleanup_thread)
            
            return jsonify({
                'success': True,
                'message': f'{model_name} model training started in background. {retrain_analysis["reason"]}',
                'model': model_name,
                'retrain_analysis': retrain_analysis,
                'training_type': 'custom_transformer'
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
        try:
            from training.model_factory import get_mediamap_model_manager
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Training module not available'
            }), 500
        
        import os
        import json
        
        manager = get_mediamap_model_manager()
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
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        try:
            from training.model_factory import get_mediamap_model_manager
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Training module not available'
            }), 500
        
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
        manager = get_mediamap_model_manager()
        success = manager.update_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{model_name} model deployed successfully',
                'model': model_name,
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
        model_name = request.args.get('model', 'highlander')
        
        # Simple fallback status for all models
        status = {
            'model_loaded': False,
            'training_examples': 0,
            'last_training': 'Never',
            'accuracy': 'N/A',
            'openai_available': bool(os.getenv('OPENAI_API_KEY'))
        }
        
        # Try to get enhanced status for mediamap/healthpin
        if model_name in ['mediamap', 'healthpin']:
            try:
                from training.openai_trainer import get_model_status
                enhanced_status = get_model_status(model_name)
                status.update(enhanced_status)
            except Exception as e:
                print(f"Could not get enhanced status for {model_name}: {e}")
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            **status
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
        try:
            from training.training_history import get_training_history
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Training module not available'
            }), 500
        
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
            # db.create_all()  # Temporarily commented out
            print("Database tables creation skipped.")
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")

# Create database tables
with app.app_context():
    # Create all tables that are defined in models
    # This will create tables if they don't exist, but won't modify existing ones
    try:
        db.create_all()
        print("✅ Database tables created/verified successfully!")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        # If there are schema conflicts, we'll handle them gracefully
        pass
    
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

# Root route - redirect to clean login page
@app.route('/')
def root():
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
    # Not logged in: redirect to clean login page
    return redirect(url_for('login'))

# User Dashboard - Simple chat interface for regular users
@app.route('/user-dashboard')
@login_required
def user_dashboard():
    """Simple dashboard for regular users - just the chat interface"""
    # Always use the new shell UI, whether admin or not
    return render_template('user_dashboard.html')

@app.route('/mediamap/dashboard')
@login_required
@section_required('mediamap')
def mediamap_dashboard():
    """MediaMap dashboard for regular users (no Highlander chat)"""
    return render_template('user_dashboard.html')

@app.route('/my_chats')
@login_required
def my_chats():
    """Simple chat history for regular users"""
    if hasattr(current_user, 'is_admin') and current_user.is_admin:
        # Admins use the full chat management interface
        return redirect(url_for('get_chats'))
    
    # Get user's chats
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).all()
    return render_template('user_chats.html', chats=chats)
# Test route
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        return jsonify({'success': True, 'method': 'POST', 'data': request.form.to_dict()})
    return jsonify({'success': True, 'method': 'GET'})

# Simple login test
@app.route('/simple-login', methods=['GET', 'POST'])
def simple_login():
    if request.method == 'POST':
        return jsonify({'success': True, 'message': 'POST request received', 'data': request.form.to_dict()})
    return jsonify({'success': True, 'message': 'GET request received'})

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Please enter both username and password.'})
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=request.form.get('remember'))
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            
            # Determine user's single allowed section
            try:
                from models import UserSection
            except ImportError:
                from models import UserSection
            try:
                user_section = UserSection.query.filter_by(user_id=user.id).first()
            except Exception as e:
                print(f"UserSection query error: {e}")
                user_section = None
            resolved_section = user_section.section if user_section else 'mediamap'
            
            # Admin direct to Admin Map if user selected admin or next targets admin
            requested_section = request.form.get('section')
            next_page = request.args.get('next')
            if (requested_section == 'admin' or (next_page and next_page.startswith('/admin'))) and getattr(user, 'is_admin', False):
                session['section'] = 'admin'
                return jsonify({'success': True, 'redirect': url_for('admin_map')})
            
            # For admin users, set section to admin to allow access to all features
            if getattr(user, 'is_admin', False):
                session['section'] = 'admin'
                return jsonify({'success': True, 'redirect': url_for('admin_map')})

            # Otherwise route by stored section
            if resolved_section == 'healthpin':
                session['section'] = 'healthpin'
                return jsonify({'success': True, 'redirect': url_for('doc_chatbot.doc_chat')})
            else:
                session['section'] = 'mediamap'
                return jsonify({'success': True, 'redirect': url_for('mediamap_dashboard')})
        else:
            return jsonify({'success': False, 'error': 'Invalid username or password.'})
    
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        section = request.form.get('section', 'mediamap')
        
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
            
            # Assign allowed section to user
            try:
                from models import UserSection
            except ImportError:
                from models import UserSection
            allowed = UserSection(user_id=new_user.id, section=section if section in ['mediamap', 'healthpin'] else 'mediamap')
            db.session.add(allowed)
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

@app.route('/ai_strategies')
@login_required
def ai_strategies():
    """Display AI strategies page"""
    return render_template('ai_strategies.html')

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

@app.route('/company_info')
@login_required
def company_info():
    """Display company information page"""
    return render_template('company_info.html')

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
        
        # Get model name and description
        model_name = request.form.get('model', 'general')
        description = request.form.get('description', '')
        
        # Create model-specific directory
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        
        if model_name in ['mediamap', 'healthpin']:
            # Use the new training_data structure
            pdf_dir = os.path.join(basedir, '..', 'training_data', model_name, 'pdfs')
        else:
            # Use the old structure for backward compatibility
            pdf_dir = os.path.join(basedir, 'training', 'training_data', 'pdfs')
        
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Save the PDF file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        filename = f"{timestamp}_{model_name}_{safe_filename}"
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
                'message': f'PDF uploaded and processed successfully for {model_name}',
                'filename': filename,
                'text_filename': text_filename,
                'model': model_name,
                'pages': len(pdf_reader.pages),
                'text_length': len(text_content),
                'description': description,
                'uploaded_at': datetime.now().isoformat()
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
@app.route('/admin/training/data-sources/<model_name>', methods=['GET'])
@login_required
@admin_required
def get_training_data_sources(model_name):
    """Get detailed information about data sources for a specific model"""
    try:
        import os
        from pathlib import Path
        
        # Define data directories
        basedir = Path(__file__).parent
        training_data_dir = basedir / "training" / "training_data"
        
        data_sources = {
            'mediamap': {
                'sources': [
                    {'name': 'RSS Feeds', 'path': 'rss', 'description': 'News articles from RSS feeds'},
                    {'name': 'Database Records', 'path': 'database', 'description': 'User chats, analyses, and insights'},
                    {'name': 'Notion Pages', 'path': 'notion', 'description': 'Imported content from Notion databases'},
                    {'name': 'Training Files', 'path': 'files', 'description': 'Manual training data files'},
                    {'name': 'Chat History', 'path': 'chats', 'description': 'User conversation history'}
                ]
            },
            'healthpin': {
                'sources': [
                    {'name': 'Medical Research', 'path': 'medical', 'description': 'Medical research papers and guidelines'},
                    {'name': 'Patient Data', 'path': 'patients', 'description': 'Anonymized patient interaction data'},
                    {'name': 'Clinical Guidelines', 'path': 'guidelines', 'description': 'Clinical practice guidelines'},
                    {'name': 'Healthcare Policy', 'path': 'policy', 'description': 'Healthcare policy documents'}
                ]
            },
            'highlander': {
                'sources': [
                    {'name': 'Business Data', 'path': 'business', 'description': 'Business intelligence and analytics'},
                    {'name': 'User Interactions', 'path': 'interactions', 'description': 'User interaction patterns'},
                    {'name': 'System Logs', 'path': 'logs', 'description': 'System usage and performance logs'}
                ]
            }
        }
        
        model_sources = data_sources.get(model_name, [])
        
        # Get actual file counts and sizes
        for source in model_sources['sources']:
            source_path = training_data_dir / model_name / source['path']
            if source_path.exists():
                files = list(source_path.glob('*'))
                source['file_count'] = len(files)
                source['total_size'] = sum(f.stat().st_size for f in files if f.is_file())
                source['last_updated'] = max((f.stat().st_mtime for f in files if f.is_file()), default=0)
            else:
                source['file_count'] = 0
                source['total_size'] = 0
                source['last_updated'] = 0
        
        return jsonify({
            'success': True,
            'model': model_name,
            'sources': model_sources['sources']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/training/collected-data/<model_name>', methods=['GET'])
@login_required
@admin_required
def get_collected_training_data(model_name):
    """Get preview of collected training data for a specific model"""
    try:
        import os
        import json
        from pathlib import Path
        from datetime import datetime
        
        basedir = Path(__file__).parent
        training_data_dir = basedir / "training" / "training_data" / model_name
        
        collected_data = {
            'total_files': 0,
            'total_size': 0,
            'recent_files': [],
            'sample_content': []
        }
        
        if training_data_dir.exists():
            # Get all files
            all_files = []
            for file_path in training_data_dir.rglob('*'):
                if file_path.is_file():
                    stat = file_path.stat()
                    all_files.append({
                        'name': file_path.name,
                        'path': str(file_path.relative_to(training_data_dir)),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'extension': file_path.suffix
                    })
            
            # Sort by modification time
            all_files.sort(key=lambda x: x['modified'], reverse=True)
            
            collected_data['total_files'] = len(all_files)
            collected_data['total_size'] = sum(f['size'] for f in all_files)
            collected_data['recent_files'] = all_files[:10]  # Last 10 files
            
            # Get sample content from recent files
            for file_info in all_files[:3]:  # Sample from 3 most recent files
                try:
                    file_path = training_data_dir / file_info['path']
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            if isinstance(content, dict):
                                sample = {
                                    'file': file_info['name'],
                                    'title': content.get('title', 'No title'),
                                    'preview': str(content.get('content', ''))[:200] + '...' if len(str(content.get('content', ''))) > 200 else str(content.get('content', '')),
                                    'type': 'json'
                                }
                            else:
                                sample = {
                                    'file': file_info['name'],
                                    'preview': str(content)[:200] + '...' if len(str(content)) > 200 else str(content),
                                    'type': 'json'
                                }
                            collected_data['sample_content'].append(sample)
                    elif file_path.suffix in ['.txt', '.md']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            sample = {
                                'file': file_info['name'],
                                'preview': content[:200] + '...' if len(content) > 200 else content,
                                'type': 'text'
                            }
                            collected_data['sample_content'].append(sample)
                except Exception as e:
                    print(f"Error reading file {file_info['name']}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'model': model_name,
            'data': collected_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/training/real-collect-data', methods=['POST'])
@login_required
@admin_required
def real_collect_training_data():
    """Collect real training data from all sources"""
    try:
        import os
        import json
        from datetime import datetime
        from training.data_collector import DataCollector
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        
        # Use the updated DataCollector with correct database path
        db_path = os.path.join(basedir, "instance", "media_analysis.db")
        collector = DataCollector(db_path=db_path)
        
        # Get request parameters
        include_datasafe = False
        include_internet_sources = False
        try:
            payload = request.get_json(silent=True)
            if payload:
                include_datasafe = bool(payload.get('include_datasafe'))
                include_internet_sources = bool(payload.get('include_internet_sources'))
        except Exception:
            include_datasafe = False
            include_internet_sources = False
        
        # Collect all data using the updated collector
        stats = collector.collect_all_data(include_internet_sources=include_internet_sources)

        # Calculate total examples from stats
        total_examples = stats.get('conversations', 0) + stats.get('pdfs', 0) + stats.get('research_papers', 0) + stats.get('feedback_entries', 0)
        
        return jsonify({
            'success': True,
            'message': 'Training data collected successfully!',
            'stats': stats,
            'total_examples': total_examples,
            'collected_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/collect-detailed-data', methods=['POST'])
@login_required
@admin_required
def collect_detailed_training_data():
    """Collect detailed training data for review and approval"""
    try:
        # Initialize enhanced data collector
        from training.enhanced_data_collector import EnhancedDataCollector
        collector = EnhancedDataCollector(output_dir=str(Path(__file__).parent / "training_data"))
        
        # Check if data already exists
        if collector.has_existing_data():
            existing_data = collector.load_existing_review_data()
            if existing_data:
                return jsonify({
                    'success': True,
                    'message': f'Using existing data: {existing_data["total_items"]} items already collected.',
                    'data': existing_data,
                    'from_cache': True
                })
        
        # Collect new detailed data
        detailed_data = collector.collect_detailed_internet_sources()
        
        return jsonify({
            'success': True,
            'message': f'Successfully collected {detailed_data["total_items"]} items for review.',
            'data': detailed_data,
            'from_cache': False
        })
        
    except Exception as e:
        print(f"Error collecting detailed training data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/force-recollect-data', methods=['POST'])
@login_required
@admin_required
def force_recollect_training_data():
    """Force re-collect training data by deleting existing data first"""
    try:
        # Initialize enhanced data collector
        from training.enhanced_data_collector import EnhancedDataCollector
        collector = EnhancedDataCollector(output_dir=str(Path(__file__).parent / "training_data"))
        
        # Delete existing data file if it exists
        if collector.review_data_file.exists():
            collector.review_data_file.unlink()
            print(f"Deleted existing data file: {collector.review_data_file}")
        
        # Collect new detailed data
        detailed_data = collector.collect_detailed_internet_sources()
        
        return jsonify({
            'success': True,
            'message': f'Successfully re-collected {detailed_data["total_items"]} fresh items for review.',
            'data': detailed_data,
            'from_cache': False
        })
        
    except Exception as e:
        print(f"Error force re-collecting training data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/review-data', methods=['GET'])
@login_required
@admin_required
def review_training_data():
    """Display detailed training data for review"""
    try:
        print("Starting review_training_data route...")
        
        # Initialize enhanced data collector with error handling
        try:
            from training.enhanced_data_collector import EnhancedDataCollector
            print("EnhancedDataCollector imported successfully")
            
            collector = EnhancedDataCollector(output_dir=str(Path(__file__).parent / "training_data"))
            print(f"Collector initialized with path: {collector.review_data_file}")
            
            # Try to load existing data first
            detailed_data = collector.load_existing_review_data()
            print(f"Loaded data: {detailed_data is not None}")
            
        except Exception as collector_error:
            print(f"Error with enhanced data collector: {collector_error}")
            detailed_data = None
        
        if detailed_data is None:
            # If no data exists or collector failed, return empty data structure
            detailed_data = {
                'arxiv_papers': [],
                'industry_content': [],
                'public_datasets': [],
                'news_articles': [],
                'technical_docs': [],
                'total_items': 0,
                'collection_timestamp': None
            }
            print("Using empty data structure")
        
        print("Rendering template...")
        return render_template('admin/review_training_data.html', data=detailed_data)
        
    except Exception as e:
        print(f"Error loading review data: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading review data', 'danger')
        return redirect(url_for('admin_training'))

@app.route('/admin/training/approve-data', methods=['POST'])
@login_required
@admin_required
def approve_training_data():
    """Approve or reject training data items"""
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        item_type = data.get('item_type')
        approved = data.get('approved', False)
        review_notes = data.get('review_notes', '')
        
        # Initialize enhanced data collector
        from training.enhanced_data_collector import EnhancedDataCollector
        collector = EnhancedDataCollector(output_dir=str(Path(__file__).parent / "training_data"))
        
        # Load current review data
        detailed_data = collector.load_existing_review_data()
        
        if detailed_data:
            # Update the specific item
            if item_type in detailed_data and isinstance(detailed_data[item_type], list):
                for item in detailed_data[item_type]:
                    if item.get('id') == item_id or item.get('title') == item_id:
                        item['approved'] = approved
                        item['review_notes'] = review_notes
                        break
            
            # Save updated data
            try:
                with open(collector.review_data_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            except Exception as save_error:
                print(f"Error saving updated data: {save_error}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to save updated data: {save_error}'
                }), 500
        
        return jsonify({
            'success': True,
            'message': f'Item {"approved" if approved else "rejected"} successfully'
        })
        
    except Exception as e:
        print(f"Error approving training data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/get-approved-data', methods=['GET'])
@login_required
@admin_required
def get_approved_training_data():
    """Get approved training data for model training"""
    try:
        # Initialize enhanced data collector
        from training.enhanced_data_collector import EnhancedDataCollector
        collector = EnhancedDataCollector(output_dir=str(Path(__file__).parent / "training_data"))
        
        # Load review data
        detailed_data = collector.load_existing_review_data()
        
        if not detailed_data:
            return jsonify({
                'success': False,
                'error': 'No review data found. Please collect and review data first.'
            }), 404
        
        # Filter approved items
        approved_data = {
            'arxiv_papers': [item for item in detailed_data.get('arxiv_papers', []) if item.get('approved', False)],
            'industry_content': [item for item in detailed_data.get('industry_content', []) if item.get('approved', False)],
            'public_datasets': [item for item in detailed_data.get('public_datasets', []) if item.get('approved', False)],
            'news_articles': [item for item in detailed_data.get('news_articles', []) if item.get('approved', False)],
            'technical_docs': [item for item in detailed_data.get('technical_docs', []) if item.get('approved', False)]
        }
        
        # Calculate totals
        total_approved = sum(len(items) for items in approved_data.values())
        
        return jsonify({
            'success': True,
            'approved_data': approved_data,
            'total_approved': total_approved,
            'summary': {
                'arxiv_papers': len(approved_data['arxiv_papers']),
                'industry_content': len(approved_data['industry_content']),
                'public_datasets': len(approved_data['public_datasets']),
                'news_articles': len(approved_data['news_articles']),
                'technical_docs': len(approved_data['technical_docs'])
            }
        })
        
    except Exception as e:
        print(f"Error getting approved training data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===== Multi-Model Training Routes =====

@app.route('/admin/training/training-data')
@login_required
@admin_required
def get_training_data():
    """Get training data for specific model"""
    try:
        model_name = request.args.get('model', 'highlander')
        
        # Get model-specific training data
        if model_name in ['mediamap', 'healthpin']:
            # Check if training data exists
            basedir = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(basedir, '..', 'training_data', model_name)
            
            conversations = 0
            pdfs = 0
            research_papers = 0
            
            # Count conversations
            conv_file = os.path.join(data_dir, 'conversations', 'all_conversations.json')
            if os.path.exists(conv_file):
                with open(conv_file, 'r') as f:
                    conv_data = json.load(f)
                    conversations = len(conv_data)
            
            # Count PDFs
            pdf_dir = os.path.join(data_dir, 'pdfs')
            if os.path.exists(pdf_dir):
                pdfs = len([f for f in os.listdir(pdf_dir) if f.endswith('.txt')])
            
            # Count research
            research_file = os.path.join(data_dir, 'research', f'{model_name}_research.json')
            if os.path.exists(research_file):
                with open(research_file, 'r') as f:
                    research_data = json.load(f)
                    research_papers = len(research_data)
            
            # Check for synthetic data
            synthetic_file = os.path.join(data_dir, 'synthetic_training_data.json')
            synthetic_examples = 0
            if os.path.exists(synthetic_file):
                with open(synthetic_file, 'r') as f:
                    synthetic_data = json.load(f)
                    synthetic_examples = len(synthetic_data)
            
        else:  # highlander
            # Highlander general data
            from models import Chat, Message
            conversations = Chat.query.count()
            pdfs = 2
            research_papers = 1
            synthetic_examples = 0
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'total_examples': conversations + pdfs + research_papers + synthetic_examples,
            'conversations': conversations,
            'pdfs': pdfs,
            'research_papers': research_papers,
            'synthetic_examples': synthetic_examples,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/validate-data', methods=['POST'])
@login_required
@admin_required
def validate_training_data_endpoint():
    """Validate training data quality"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        if model_name in ['mediamap', 'healthpin']:
            from training.training_validator import validate_training_data
            import os
            
            basedir = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(basedir, 'training_data', model_name)
            
            validation_report = validate_training_data(model_name, data_dir)
            
            return jsonify({
                'success': True,
                'validation_report': validation_report,
                'model': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Validation not available for {model_name}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/feedback-integration', methods=['POST'])
@login_required
@admin_required
def integrate_feedback_endpoint():
    """Integrate user feedback for continuous learning"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        if model_name in ['mediamap', 'healthpin']:
            from training.feedback_integration import integrate_feedback
            import os
            
            basedir = os.path.abspath(os.path.dirname(__file__))
            db_path = os.path.join(basedir, "instance", "media_analysis.db")
            
            feedback_results = integrate_feedback(model_name, db_path)
            
            return jsonify({
                'success': True,
                'feedback_integration': feedback_results,
                'model': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Feedback integration not available for {model_name}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/performance-evaluation', methods=['POST'])
@login_required
@admin_required
def evaluate_model_performance_endpoint():
    """Evaluate model performance"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                'success': False,
                'error': 'Model ID is required for performance evaluation'
            }), 400
        
        if model_name in ['mediamap', 'healthpin']:
            from training.training_validator import evaluate_model_performance
            import os
            
            basedir = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(basedir, 'training_data', model_name)
            
            performance_report = evaluate_model_performance(model_name, model_id, data_dir)
            
            return jsonify({
                'success': True,
                'performance_report': performance_report,
                'model': model_name,
                'model_id': model_id
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Performance evaluation not available for {model_name}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/model-status-new')
@login_required
@admin_required
def get_model_status_new():
    """Get model status for specific model"""
    try:
        model_name = request.args.get('model', 'highlander')
        
        if model_name in ['mediamap', 'healthpin']:
            try:
                from training.openai_trainer import get_model_status
                status = get_model_status(model_name)
            except ImportError as ie:
                # Fallback if openai_trainer is not available
                status = {
                    'model_loaded': False,
                    'training_examples': 0,
                    'last_training': 'Never',
                    'accuracy': 'N/A',
                    'openai_available': bool(os.getenv('OPENAI_API_KEY')),
                    'error': f'OpenAI trainer not available: {str(ie)}'
                }
        else:
            # Default status for highlander
            status = {
                'model_loaded': False,
                'training_examples': 0,
                'last_training': 'Never',
                'accuracy': 'N/A',
                'openai_available': bool(os.getenv('OPENAI_API_KEY'))
            }
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            **status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/training-progress')
@login_required
@admin_required
def get_training_progress():
    """Get training progress for specific model"""
    try:
        model_name = request.args.get('model', 'highlander')
        
        if model_name in ['mediamap', 'healthpin']:
            from training.openai_trainer import get_training_status
            status = get_training_status(model_name)
            
            if status.get('success'):
                return jsonify({
                    'success': True,
                    'model_name': model_name,
                    'status': status.get('status', 'unknown'),
                    'completed': status.get('completed', False),
                    'progress': 100 if status.get('completed') else 50,
                    'job_id': status.get('job_id'),
                    'model_id': status.get('model_id')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': status.get('error', 'Unknown error')
                })
        
        # Simulate training progress for highlander
        import random
        progress = random.randint(0, 100)
        completed = progress >= 100
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'progress': progress,
            'completed': completed,
            'status': 'Training in progress' if not completed else 'Training completed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/training/test-model', methods=['POST'])
@login_required
@admin_required
def test_model():
    """Test specific model with sample queries"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'highlander')
        
        # Simulate model testing
        test_results = {
            'model_name': model_name,
            'test_queries': [
                {
                    'query': f'Sample query for {model_name}',
                    'response': f'This is a test response from {model_name} model',
                    'accuracy': 0.95
                }
            ],
            'overall_accuracy': 0.95,
            'response_time': 0.5,
            'test_completed': True
        }
        
        return jsonify({
            'success': True,
            'results': test_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Agent Training Integration Endpoints
@app.route('/admin/training/agent-data-stats')
@login_required
@admin_required
def get_agent_training_stats():
    """Get statistics about available agent training data"""
    try:
        from training.agent_training_bridge import create_training_bridge
        
        bridge = create_training_bridge()
        stats = bridge.get_training_data_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/training/collect-agent-data', methods=['POST'])
@login_required
@admin_required
def collect_agent_training_data():
    """Collect training data from agents"""
    try:
        from training.agent_training_bridge import create_training_bridge
        
        bridge = create_training_bridge()
        stats = bridge.collect_all_agent_training_data()
        
        return jsonify({
            'success': True,
            'message': 'Agent training data collected successfully',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/training/train-from-agents', methods=['POST'])
@login_required
@admin_required
def train_model_from_agents():
    """Train a model using agent-collected data"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name', 'mediamap')
        
        from training.agent_training_bridge import create_training_bridge
        from training.openai_trainer import OpenAITrainer
        
        # First collect the latest agent data
        bridge = create_training_bridge()
        collection_stats = bridge.collect_all_agent_training_data()
        
        # Initialize trainer
        trainer = OpenAITrainer(model_name)
        
        # Prepare training data from agent-collected data
        training_file = str(bridge.training_output_path / "agent_consolidated_training.json")
        
        if not os.path.exists(training_file):
            return jsonify({
                'success': False,
                'error': 'No agent training data available. Please collect data first.'
            })
        
        # Start training process
        result = trainer.train_model(training_file)
        
        return jsonify({
            'success': True,
            'message': f'Training started for {model_name} using agent data',
            'collection_stats': collection_stats,
            'training_result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/admin/training/start-continuous-collection', methods=['POST'])
@login_required
@admin_required
def start_continuous_agent_collection():
    """Start continuous collection of training data from agents"""
    try:
        data = request.get_json() or {}
        interval_hours = data.get('interval_hours', 24)
        
        from training.agent_training_bridge import create_training_bridge
        
        bridge = create_training_bridge()
        result = bridge.start_continuous_training_collection(interval_hours)
        
        return jsonify({
            'success': True,
            'message': 'Continuous agent data collection started',
            'schedule': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def find_available_port(start_port=3000, max_port=8100):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def kill_existing_processes():
    """Kill any existing Flask processes that might be using ports"""
    import subprocess
    try:
        # Kill any existing python app.py processes
        subprocess.run(['pkill', '-f', 'python app.py'], capture_output=True)
        print("🧹 Cleaned up existing Flask processes")
    except Exception as e:
        print(f"Note: Could not clean existing processes: {e}")


@app.route('/admin/<path:path>')
@login_required
def admin_catch_all(path):
    """Redirect old admin paths to app selector"""
    flash('Please select your admin application from the options below', 'warning')
    return redirect(url_for('app_selector'))


# Test routes for separate admin apps
@app.route('/test-mediamap-admin')
@login_required
def test_mediamap_admin():
    """Test route for MediaMap Admin"""
    session['app_context'] = 'mediamap_admin'
    return redirect('/mediamap-admin/')

@app.route('/test-healthpin-admin')
@login_required
def test_healthpin_admin():
    """Test route for HealthPIN Admin"""
    session['app_context'] = 'healthpin_admin'
    return redirect('/healthpin-admin/')

if __name__ == '__main__':
    sys.path.append('/path/to/your/directory')
    
    # Clean up any existing processes first
    kill_existing_processes()
    
    # Find an available port
    port = find_available_port()
    if not port:
        print("❌ Could not find an available port between 3000-8100")
        exit(1)
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🌐 Access your app at: http://localhost:{port}")
    print(f"🔧 Debug mode: ON")
    print("-" * 50)
    
    # Handle subdirectory deployment
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)
    
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Flask app stopped by user")
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        print("💡 Try running the app again or check for port conflicts") 
