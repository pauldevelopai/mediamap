"""
HealthPIN Webhook Handlers
=========================

Handles incoming WhatsApp and SMS messages from Twilio for HealthPIN platform.
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required
import logging
from datetime import datetime
import json

from .models import Patient, Doctor, HealthRecord, FamilyNotification
from .twilio_service import twilio_service
from models import db

# Create Blueprint
webhooks_bp = Blueprint('healthpin_webhooks', __name__, url_prefix='/healthpin/webhooks')

logger = logging.getLogger(__name__)

@webhooks_bp.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """
    Handle incoming WhatsApp messages from Twilio
    
    This endpoint receives messages when patients send WhatsApp messages
    to the HealthPIN WhatsApp number.
    """
    try:
        # Get message data from Twilio
        from_number = request.form.get('From', '').replace('whatsapp:', '')
        message_body = request.form.get('Body', '')
        message_sid = request.form.get('MessageSid', '')
        
        logger.info(f"Received WhatsApp message from {from_number}: {message_body}")
        
        # Find patient by phone number
        patient = Patient.query.filter_by(phone_number=from_number, is_active=True).first()
        
        if not patient:
            # Unknown number - send welcome message
            response_message = """🏥 *Welcome to HealthPIN!*

I'm your AI health companion. To get started, please register with us first.

Visit our website or contact our support team to create your HealthPIN account.

*HealthPIN - Your health companion*"""
            
            result = twilio_service.send_whatsapp_message(from_number, response_message)
            
            return jsonify({
                'success': True,
                'message': 'Welcome message sent to unknown number',
                'twilio_result': result
            })
        
        # Process patient message
        response = process_patient_message(patient, message_body)
        
        # Send response back to patient
        if response:
            result = twilio_service.send_whatsapp_message(from_number, response)
            logger.info(f"Sent response to {from_number}: {result}")
        
        return jsonify({
            'success': True,
            'message': 'WhatsApp message processed',
            'patient_id': patient.id,
            'response_sent': bool(response)
        })
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@webhooks_bp.route('/sms', methods=['POST'])
def sms_webhook():
    """
    Handle incoming SMS messages from Twilio
    
    This endpoint receives SMS messages and can be used for
    family member responses or emergency communications.
    """
    try:
        from_number = request.form.get('From', '')
        message_body = request.form.get('Body', '')
        message_sid = request.form.get('MessageSid', '')
        
        logger.info(f"Received SMS from {from_number}: {message_body}")
        
        # For now, just acknowledge receipt
        # In a full implementation, you might want to:
        # - Check if it's a family member responding to a notification
        # - Handle emergency responses
        # - Process appointment confirmations
        
        return jsonify({
            'success': True,
            'message': 'SMS received and acknowledged'
        })
        
    except Exception as e:
        logger.error(f"Error processing SMS webhook: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@webhooks_bp.route('/status', methods=['POST'])
def message_status_webhook():
    """
    Handle message status updates from Twilio
    
    This endpoint receives status updates when messages are delivered,
    read, or fail to deliver.
    """
    try:
        message_sid = request.form.get('MessageSid', '')
        message_status = request.form.get('MessageStatus', '')
        error_code = request.form.get('ErrorCode', '')
        error_message = request.form.get('ErrorMessage', '')
        
        logger.info(f"Message status update: {message_sid} - {message_status}")
        
        # Update message status in database if needed
        # This could be used to track delivery success rates
        
        return jsonify({
            'success': True,
            'message': 'Status update received'
        })
        
    except Exception as e:
        logger.error(f"Error processing status webhook: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_patient_message(patient, message: str) -> str:
    """
    Process incoming patient message and generate appropriate response
    
    Args:
        patient: Patient object
        message: Incoming message text
    
    Returns:
        Response message to send back to patient
    """
    message_lower = message.lower().strip()
    
    # Greeting responses
    if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'sawubona', 'molo']):
        return f"""Sawubona {patient.first_name}! 👋

I'm your HealthPIN health companion. How can I help you today?

You can ask me about:
• Finding a doctor (type "find doctor")
• Your health records (type "my records")
• Health tips (type "health tips")
• Emergency help (type "emergency")

*HealthPIN - Your health companion*"""
    
    # Doctor finding
    elif any(keyword in message_lower for keyword in ['find doctor', 'doctor', 'need doctor', 'funa udokotela']):
        return f"""🔍 *Finding a Doctor*

I can help you find the right doctor, {patient.first_name}!

What type of doctor do you need?
• General doctor
• Heart specialist
• Women's health
• Children's doctor
• Emergency

Or tell me your symptoms and I'll suggest the right specialist.

*HealthPIN - Your health companion*"""
    
    # Health records
    elif any(keyword in message_lower for keyword in ['my records', 'records', 'health history', 'amarekhodi ami']):
        return f"""📋 *Your Health Records*

Here's a summary of your recent health activity, {patient.first_name}:

• Last visit: [Would show actual date from database]
• Current medications: [Would show actual medications]
• Next appointment: [Would show actual appointment]

For detailed records, please visit our website or contact your doctor.

*HealthPIN - Your health companion*"""
    
    # Health tips
    elif any(keyword in message_lower for keyword in ['health tips', 'tips', 'advice', 'amacebiso']):
        return f"""💡 *Health Tips*

Here are some health tips for you, {patient.first_name}:

• Drink 8 glasses of water daily
• Get 7-8 hours of sleep
• Exercise for 30 minutes daily
• Eat fruits and vegetables
• Wash hands regularly

Would you like tips for a specific health condition?

*HealthPIN - Your health companion*"""
    
    # Emergency
    elif any(keyword in message_lower for keyword in ['emergency', 'help', 'urgent', 'sos', 'ngosizo']):
        return f"""🚨 *Emergency Help*

{patient.first_name}, if this is a medical emergency:

• Call 10177 (Emergency Services)
• Go to your nearest hospital
• Contact your emergency contact: {patient.emergency_contact_name or 'Not set'}

For non-emergency health questions, I'm here to help!

*HealthPIN - Your health companion*"""
    
    # Medication reminders
    elif any(keyword in message_lower for keyword in ['medication', 'medicine', 'pills', 'imithi']):
        return f"""💊 *Medication Help*

I can help you with medication reminders, {patient.first_name}!

Your current medications:
• [Would show actual medications from database]

Need a reminder? I can send you daily medication alerts.

*HealthPIN - Your health companion*"""
    
    # Default response
    else:
        return f"""I understand you said: "{message}"

I'm still learning, {patient.first_name}! For now, I can help you with:

• Finding doctors (type "find doctor")
• Health records (type "my records") 
• Health tips (type "health tips")
• Emergency help (type "emergency")

*HealthPIN - Your health companion*"""

@webhooks_bp.route('/test', methods=['GET'])
@login_required
def test_webhook():
    """Test endpoint to verify webhook functionality"""
    return jsonify({
        'success': True,
        'message': 'HealthPIN webhooks are working',
        'timestamp': datetime.utcnow().isoformat(),
        'twilio_configured': twilio_service.is_configured()
    })
