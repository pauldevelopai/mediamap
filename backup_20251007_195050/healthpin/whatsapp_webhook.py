"""
WhatsApp Webhook Handler for HealthPIN
=====================================

Handles incoming WhatsApp messages and integrates with Doc chatbot
"""

import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from healthpin.doc_chatbot import doc_manager
from healthpin.twilio_service import twilio_service
from models import db, ChatMessage, User

logger = logging.getLogger(__name__)

# Create Blueprint
whatsapp_webhook_bp = Blueprint('whatsapp_webhook', __name__, url_prefix='/healthpin/webhooks')

@whatsapp_webhook_bp.route('/whatsapp', methods=['POST'])
def handle_whatsapp_message():
    """
    Handle incoming WhatsApp messages from Twilio
    """
    try:
        # Get message data from Twilio
        message_body = request.form.get('Body', '').strip()
        from_number = request.form.get('From', '').replace('whatsapp:', '')
        to_number = request.form.get('To', '').replace('whatsapp:', '')
        message_sid = request.form.get('MessageSid', '')
        
        logger.info(f"WhatsApp message received from {from_number}: {message_body[:100]}...")
        
        if not message_body:
            return _create_twiml_response("I didn't receive your message. Please try again.")
        
        # Check if this is a sandbox join message
        if message_body.lower() == 'join clay-surprise':
            return _create_twiml_response(
                "🎉 Welcome to HealthPIN WhatsApp! You're now connected to Doc, your healthcare AI assistant. "
                "Ask me about patient care, healthcare technology, analytics, or any medical topic!"
            )
        
        # Get or create user based on phone number
        user = _get_or_create_whatsapp_user(from_number)
        
        # Generate response using Doc chatbot
        result = doc_manager.generate_response(
            message=message_body,
            conversation_history=_get_whatsapp_conversation_history(user.id),
            user_context={'phone': from_number, 'platform': 'whatsapp'}
        )
        
        # Save the conversation
        chat_message = ChatMessage(
            user_id=user.id,
            message=message_body,
            response=result['response'],
            chatbot_type='doc_healthpin_whatsapp',
            metadata=json.dumps({
                'context_type': result['context_type'],
                'response_time_ms': result['response_time_ms'],
                'platform': 'whatsapp',
                'from_number': from_number,
                'message_sid': message_sid
            })
        )
        
        db.session.add(chat_message)
        db.session.commit()
        
        # Format response for WhatsApp
        whatsapp_response = _format_whatsapp_response(result['response'], result['context_type'])
        
        logger.info(f"WhatsApp response sent to {from_number}: {whatsapp_response[:100]}...")
        
        return _create_twiml_response(whatsapp_response)
        
    except Exception as e:
        logger.error(f"Error handling WhatsApp message: {str(e)}")
        return _create_twiml_response(
            "I'm sorry, I encountered an error processing your message. Please try again later."
        )

@whatsapp_webhook_bp.route('/whatsapp/status', methods=['POST'])
def handle_whatsapp_status():
    """
    Handle WhatsApp message status updates from Twilio
    """
    try:
        message_sid = request.form.get('MessageSid', '')
        message_status = request.form.get('MessageStatus', '')
        error_code = request.form.get('ErrorCode', '')
        
        logger.info(f"WhatsApp message status update: {message_sid} - {message_status}")
        
        # You can update message status in database here if needed
        # For now, just log it
        
        return jsonify({'status': 'received'})
        
    except Exception as e:
        logger.error(f"Error handling WhatsApp status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@whatsapp_webhook_bp.route('/sms', methods=['POST'])
def handle_sms_message():
    """
    Handle incoming SMS messages from Twilio
    """
    try:
        message_body = request.form.get('Body', '').strip()
        from_number = request.form.get('From', '')
        to_number = request.form.get('To', '')
        message_sid = request.form.get('MessageSid', '')
        
        logger.info(f"SMS message received from {from_number}: {message_body[:100]}...")
        
        if not message_body:
            return _create_twiml_response("I didn't receive your message. Please try again.")
        
        # Get or create user based on phone number
        user = _get_or_create_whatsapp_user(from_number)
        
        # Generate response using Doc chatbot
        result = doc_manager.generate_response(
            message=message_body,
            conversation_history=_get_whatsapp_conversation_history(user.id),
            user_context={'phone': from_number, 'platform': 'sms'}
        )
        
        # Save the conversation
        chat_message = ChatMessage(
            user_id=user.id,
            message=message_body,
            response=result['response'],
            chatbot_type='doc_healthpin_sms',
            metadata=json.dumps({
                'context_type': result['context_type'],
                'response_time_ms': result['response_time_ms'],
                'platform': 'sms',
                'from_number': from_number,
                'message_sid': message_sid
            })
        )
        
        db.session.add(chat_message)
        db.session.commit()
        
        # Format response for SMS (shorter than WhatsApp)
        sms_response = _format_sms_response(result['response'], result['context_type'])
        
        logger.info(f"SMS response sent to {from_number}: {sms_response[:100]}...")
        
        return _create_twiml_response(sms_response)
        
    except Exception as e:
        logger.error(f"Error handling SMS message: {str(e)}")
        return _create_twiml_response(
            "I'm sorry, I encountered an error processing your message. Please try again later."
        )

def _get_or_create_whatsapp_user(phone_number):
    """Get or create user based on WhatsApp phone number"""
    # Try to find existing user by phone number
    user = User.query.filter_by(phone=phone_number).first()
    
    if not user:
        # Create new user for WhatsApp
        username = f"whatsapp_{phone_number.replace('+', '').replace(' ', '')}"
        user = User(
            username=username,
            email=f"{username}@whatsapp.healthpin.local",
            phone=phone_number,
            is_active=True,
            is_verified=True  # Assume WhatsApp numbers are verified
        )
        
        db.session.add(user)
        db.session.commit()
        
        logger.info(f"Created new WhatsApp user: {username}")
    
    return user

def _get_whatsapp_conversation_history(user_id, limit=5):
    """Get recent conversation history for WhatsApp user"""
    messages = ChatMessage.query.filter(
        ChatMessage.user_id == user_id,
        ChatMessage.chatbot_type.in_(['doc_healthpin_whatsapp', 'doc_healthpin_sms'])
    ).order_by(ChatMessage.created_at.desc()).limit(limit * 2).all()
    
    # Convert to conversation format
    history = []
    for msg in messages:
        history.append({
            'role': 'user',
            'content': msg.message
        })
        history.append({
            'role': 'assistant', 
            'content': msg.response
        })
    
    # Reverse to get chronological order
    return list(reversed(history))

def _format_whatsapp_response(response, context_type):
    """Format response for WhatsApp with appropriate emojis and formatting"""
    context_emojis = {
        'main': '🏥',
        'patient_care': '👩‍⚕️',
        'technology': '💻',
        'analytics': '📊'
    }
    
    emoji = context_emojis.get(context_type, '🏥')
    
    # Add context indicator
    formatted_response = f"{emoji} *Doc (HealthPIN)*\n\n{response}"
    
    # Add helpful footer
    formatted_response += "\n\n---\n💬 Reply with any healthcare questions!"
    
    return formatted_response

def _format_sms_response(response, context_type):
    """Format response for SMS (shorter, no special formatting)"""
    # Truncate if too long for SMS
    if len(response) > 140:
        response = response[:137] + "..."
    
    return f"Doc: {response}"

def _create_twiml_response(message):
    """Create TwiML response for Twilio"""
    response = MessagingResponse()
    response.message(message)
    return str(response)




