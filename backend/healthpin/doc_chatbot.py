"""
Doc Chatbot for HealthPIN - Healthcare AI Assistant
Specialized chatbot for healthcare providers and medical professionals
"""

import json
import time
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session
from flask_login import login_required, current_user
# Import section_required locally to avoid circular import
from functools import wraps
from flask import session, redirect, url_for, flash

def section_required(required_section):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Allow admin users to bypass section requirements
            if hasattr(current_user, 'is_admin') and current_user.is_admin:
                return f(*args, **kwargs)
            
            current_section = session.get('section')
            if current_section != required_section:
                flash(f'You need to be in the {required_section} section to access this feature.', 'warning')
                return redirect(url_for('user_dashboard'))
            return f(*args, **kwargs)
        return wrapped
    return decorator
from models import db, ChatMessage, User, PromptTemplate
from prompt_manager import get_prompt as get_prompt_from_db
from prompt_version_manager import performance_tracker
from healthpin.models import Patient, HealthRecord, Doctor

# Create Blueprint
doc_chatbot_bp = Blueprint('doc_chatbot', __name__, url_prefix='/healthpin/doc')

class DocChatbotManager:
    """Manages the Doc chatbot for HealthPIN"""
    
    def __init__(self):
        self.name = "Doc"
        self.description = "Healthcare AI Assistant"
        self.specialization = "Medical information, patient care, and healthcare technology"
    
    def get_system_prompt(self, context_type="main"):
        """Get the appropriate system prompt based on context"""
        prompt_map = {
            "main": "DOC_SYSTEM_PROMPT_MAIN",
            "patient_care": "DOC_SYSTEM_PROMPT_PATIENT_CARE", 
            "technology": "DOC_SYSTEM_PROMPT_TECHNOLOGY",
            "analytics": "DOC_SYSTEM_PROMPT_ANALYTICS",
            "patient_lookup": "DOC_SYSTEM_PROMPT_PATIENT_LOOKUP",
            "patient_analysis": "DOC_SYSTEM_PROMPT_PATIENT_ANALYSIS",
            "patient_care_plan": "DOC_SYSTEM_PROMPT_PATIENT_CARE_PLAN"
        }
        
        prompt_name = prompt_map.get(context_type, "DOC_SYSTEM_PROMPT_MAIN")
        return get_prompt_from_db(prompt_name)
    
    def analyze_message_context(self, message):
        """Analyze message to determine appropriate context type"""
        message_lower = message.lower()
        
        # Patient-specific keywords (highest priority)
        patient_specific_keywords = [
            'patient', 'look up', 'find patient', 'patient record', 'patient data',
            'patient history', 'patient info', 'patient details', 'patient summary',
            'patient analysis', 'care plan', 'treatment plan', 'patient care plan'
        ]
        
        # Patient care keywords
        patient_care_keywords = [
            'care', 'clinical', 'diagnosis', 'treatment', 'symptoms',
            'medication', 'therapy', 'outcome', 'safety', 'quality', 'protocol'
        ]
        
        # Technology keywords  
        technology_keywords = [
            'ehr', 'electronic health record', 'system', 'integration', 'software',
            'technology', 'digital', 'automation', 'workflow', 'platform'
        ]
        
        # Analytics keywords
        analytics_keywords = [
            'data', 'analytics', 'reporting', 'metrics', 'kpi', 'dashboard',
            'insights', 'performance', 'benchmark', 'population health'
        ]
        
        # Count keyword matches
        patient_specific_score = sum(1 for keyword in patient_specific_keywords if keyword in message_lower)
        patient_care_score = sum(1 for keyword in patient_care_keywords if keyword in message_lower)
        technology_score = sum(1 for keyword in technology_keywords if keyword in message_lower)
        analytics_score = sum(1 for keyword in analytics_keywords if keyword in message_lower)
        
        # Return context with highest score (patient-specific has priority)
        if patient_specific_score > 0:
            return "patient_lookup"
        elif patient_care_score > technology_score and patient_care_score > analytics_score:
            return "patient_care"
        elif technology_score > analytics_score:
            return "technology"
        elif analytics_score > 0:
            return "analytics"
        else:
            return "main"
    
    def generate_response(self, message, conversation_history=None, user_context=None):
        """Generate response using appropriate system prompt"""
        start_time = time.time()
        
        try:
            # Determine context type
            context_type = self.analyze_message_context(message)
            
            # Get appropriate system prompt
            system_prompt = self.get_system_prompt(context_type)
            
            # Build conversation context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call OpenAI API for real AI responses
            response = self._call_openai_api(messages)
            
            # Track performance
            response_time_ms = int((time.time() - start_time) * 1000)
            self._track_usage(context_type, response_time_ms, len(message), len(response))
            
            return {
                "response": response,
                "context_type": context_type,
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            print(f"❌ Error generating Doc response: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "context_type": "main",
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _call_openai_api(self, messages):
        """Call OpenAI API or HealthPIN model to generate AI response"""
        try:
            # Try to use HealthPIN model first
            try:
                from training.model_factory import get_healthpin_model_manager
                
                healthpin_manager = get_healthpin_model_manager()
                
                if healthpin_manager and healthpin_manager.is_model_loaded:
                    # Use custom HealthPIN model
                    response, source = healthpin_manager.generate_response(
                        messages[-1]["content"] if messages else "Hello",
                        conversation_history=messages[:-1] if len(messages) > 1 else []
                    )
                    return response
            except Exception as model_error:
                print(f"❌ HealthPIN model error: {model_error}")
            
            # Fallback to OpenAI
            try:
                import openai
                from backend.app import client
                
                if client:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                else:
                    raise Exception("OpenAI client not available")
                    
            except Exception as openai_error:
                print(f"❌ OpenAI API error in Doc chatbot: {openai_error}")
                raise openai_error
            
        except Exception as e:
            print(f"❌ AI API error in Doc chatbot: {e}")
            # Fallback to simple response
            return self._generate_simple_response(
                messages[-1]["content"] if messages else "Hello",
                "main",
                messages[0]["content"] if messages else ""
            )
    
    def _generate_simple_response(self, message, context_type, system_prompt):
        """Generate a simple response based on context (placeholder for full AI integration)"""
        
        # Patient-specific responses
        if context_type == "patient_lookup":
            return self._handle_patient_lookup_query(message)
        
        elif context_type == "patient_analysis":
            return self._handle_patient_analysis_query(message)
        
        elif context_type == "patient_care_plan":
            return self._handle_patient_care_plan_query(message)
        
        # Healthcare-focused responses based on context
        elif context_type == "patient_care":
            return f"I understand you're asking about patient care. Based on your message: '{message[:100]}...', I'd recommend focusing on evidence-based practices and patient safety protocols. What specific aspect of patient care would you like to explore further?"
        
        elif context_type == "technology":
            return f"I see you're interested in healthcare technology. Regarding: '{message[:100]}...', I can help you with EHR optimization, system integration, or digital health solutions. What specific technology challenge are you facing?"
        
        elif context_type == "analytics":
            return f"You're asking about healthcare analytics. For: '{message[:100]}...', I can assist with data insights, reporting, or performance metrics. What specific analytics or reporting needs do you have?"
        
        else:  # main context
            return f"Hello! I'm Doc, your healthcare AI assistant. I'm here to help with medical information, patient care, and healthcare technology. You mentioned: '{message[:100]}...'. How can I assist you with your healthcare challenges today?"
    
    def _handle_patient_lookup_query(self, message):
        """Handle patient lookup queries"""
        message_lower = message.lower()
        
        # Extract potential patient identifiers
        identifiers = []
        
        # Look for phone numbers
        import re
        phone_pattern = r'\+?\d{10,15}'
        phones = re.findall(phone_pattern, message)
        identifiers.extend(phones)
        
        # Look for names (simple heuristic)
        words = message.split()
        potential_names = []
        for word in words:
            if word.isalpha() and len(word) > 2 and word.lower() not in ['patient', 'look', 'up', 'find', 'show', 'me', 'the', 'for', 'with', 'about']:
                potential_names.append(word)
        
        if potential_names:
            identifiers.extend([' '.join(potential_names)])
        
        # Try to find patients
        found_patients = []
        for identifier in identifiers:
            patient = self.lookup_patient(identifier)
            if patient:
                found_patients.append(patient)
        
        if found_patients:
            patient = found_patients[0]
            summary = self.get_patient_summary(patient)
            
            response = f"✅ **Patient Found: {patient.first_name} {patient.last_name}**\n\n"
            response += f"**Patient ID:** {patient.id}\n"
            response += f"**Phone:** {patient.phone_number}\n"
            response += f"**Age:** {self._calculate_age(patient.date_of_birth) if patient.date_of_birth else 'Not specified'}\n"
            response += f"**Location:** {patient.city}, {patient.province}\n"
            
            if summary and summary['recent_conditions']:
                response += f"\n**Recent Conditions:**\n"
                for condition in summary['recent_conditions'][:3]:
                    response += f"• {condition['condition']} ({condition['date']})\n"
            
            if summary and summary['current_medications']:
                response += f"\n**Current Medications:**\n"
                for med in summary['current_medications'][:3]:
                    response += f"• {med}\n"
            
            response += f"\nWould you like me to provide a detailed analysis or care plan for this patient?"
            
            return response
        
        else:
            return f"I'd be happy to help you look up a patient. Please provide:\n\n• Patient name (first and last name)\n• Phone number\n• Patient ID\n\nFor example: 'Look up patient John Smith' or 'Find patient +27821234567'"
    
    def _handle_patient_analysis_query(self, message):
        """Handle patient analysis queries"""
        return "I can provide comprehensive patient analysis including risk factors, care recommendations, and clinical insights. Please first identify the patient you'd like me to analyze, then I'll provide detailed clinical assessment."
    
    def _handle_patient_care_plan_query(self, message):
        """Handle patient care plan queries"""
        return "I can develop personalized care plans for specific patients. Please identify the patient first, then I'll create a comprehensive care plan including treatment goals, medication management, and follow-up schedules."
    
    def _calculate_age(self, date_of_birth):
        """Calculate age from date of birth"""
        if not date_of_birth:
            return None
        
        from datetime import date
        today = date.today()
        return today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    
    def _track_usage(self, context_type, response_time_ms, input_length, output_length):
        """Track usage for performance metrics"""
        try:
            # Get the appropriate prompt for tracking
            prompt_map = {
                "main": "DOC_SYSTEM_PROMPT_MAIN",
                "patient_care": "DOC_SYSTEM_PROMPT_PATIENT_CARE",
                "technology": "DOC_SYSTEM_PROMPT_TECHNOLOGY", 
                "analytics": "DOC_SYSTEM_PROMPT_ANALYTICS"
            }
            
            prompt_name = prompt_map.get(context_type, "DOC_SYSTEM_PROMPT_MAIN")
            prompt = PromptTemplate.query.filter_by(name=prompt_name).first()
            
            if prompt:
                performance_tracker.record_usage(
                    prompt_id=prompt.id,
                    version_number=prompt.version,
                    response_time_ms=response_time_ms,
                    token_count_input=input_length // 4,  # Rough token estimate
                    token_count_output=output_length // 4,
                    cost_estimate=0.001,  # Placeholder cost
                    user_id=current_user.id if current_user.is_authenticated else None,
                    session_id=session.get('session_id'),
                    usage_context=f'Doc chatbot - {context_type}',
                    variables_used={'context_type': context_type}
                )
        except Exception as e:
            print(f"⚠️ Error tracking Doc usage: {e}")
    
    def lookup_patient(self, identifier):
        """
        Look up a patient by various identifiers
        
        Args:
            identifier: Patient name, phone number, or ID
            
        Returns:
            Patient object if found, None otherwise
        """
        try:
            # Try by phone number first
            if identifier.startswith('+') or identifier.replace(' ', '').isdigit():
                patient = Patient.query.filter_by(phone_number=identifier).first()
                if patient:
                    return patient
            
            # Try by ID
            if identifier.isdigit():
                patient = Patient.query.get(int(identifier))
                if patient:
                    return patient
            
            # Try by name (first name, last name, or full name)
            name_parts = identifier.lower().split()
            if len(name_parts) >= 1:
                # Search by first name
                patients = Patient.query.filter(
                    db.func.lower(Patient.first_name).like(f'%{name_parts[0]}%')
                ).all()
                
                if len(name_parts) >= 2:
                    # Search by first and last name
                    patients = [p for p in patients if 
                              db.func.lower(p.last_name).like(f'%{name_parts[1]}%')]
                
                if patients:
                    return patients[0]  # Return first match
            
            return None
            
        except Exception as e:
            print(f"❌ Error looking up patient: {e}")
            return None
    
    def get_patient_summary(self, patient):
        """
        Get a comprehensive summary of patient information
        
        Args:
            patient: Patient object
            
        Returns:
            Dictionary with patient summary
        """
        try:
            # Get health records
            health_records = HealthRecord.query.filter_by(patient_id=patient.id).order_by(
                HealthRecord.record_date.desc()
            ).limit(10).all()
            
            # Get recent records summary
            recent_conditions = []
            current_medications = []
            recent_lab_results = []
            
            for record in health_records:
                if record.record_type == 'diagnosis':
                    recent_conditions.append({
                        'condition': record.title,
                        'date': record.record_date.isoformat(),
                        'doctor': record.doctor_name
                    })
                elif record.record_type == 'prescription':
                    if record.medications:
                        current_medications.extend(record.medications)
                elif record.record_type == 'lab_result':
                    recent_lab_results.append({
                        'test': record.title,
                        'date': record.record_date.isoformat(),
                        'results': record.lab_results
                    })
            
            return {
                'patient_info': patient.to_dict(),
                'recent_conditions': recent_conditions,
                'current_medications': list(set(current_medications)),  # Remove duplicates
                'recent_lab_results': recent_lab_results,
                'total_records': len(health_records),
                'last_visit': health_records[0].record_date.isoformat() if health_records else None
            }
            
        except Exception as e:
            print(f"❌ Error getting patient summary: {e}")
            return None
    
    def analyze_patient_data(self, patient_summary):
        """
        Analyze patient data and provide clinical insights
        
        Args:
            patient_summary: Patient summary dictionary
            
        Returns:
            Dictionary with clinical insights
        """
        try:
            insights = {
                'risk_factors': [],
                'care_recommendations': [],
                'follow_up_needed': [],
                'medication_review': [],
                'preventive_care': []
            }
            
            patient_info = patient_summary['patient_info']
            recent_conditions = patient_summary['recent_conditions']
            current_medications = patient_summary['current_medications']
            
            # Age-based risk assessment
            if patient_info.get('date_of_birth'):
                from datetime import date
                birth_date = datetime.strptime(patient_info['date_of_birth'], '%Y-%m-%d').date()
                age = (date.today() - birth_date).days // 365
                
                if age >= 65:
                    insights['risk_factors'].append('Advanced age - increased risk for chronic conditions')
                    insights['preventive_care'].append('Annual comprehensive health assessment recommended')
                elif age >= 50:
                    insights['preventive_care'].append('Regular cancer screenings and cardiovascular assessment')
            
            # Condition-based insights
            for condition in recent_conditions:
                condition_name = condition['condition'].lower()
                if 'diabetes' in condition_name:
                    insights['care_recommendations'].append('Regular blood glucose monitoring and HbA1c testing')
                    insights['follow_up_needed'].append('Endocrinology consultation if not already scheduled')
                elif 'hypertension' in condition_name or 'high blood pressure' in condition_name:
                    insights['care_recommendations'].append('Daily blood pressure monitoring and lifestyle modifications')
                elif 'heart' in condition_name or 'cardiac' in condition_name:
                    insights['risk_factors'].append('Cardiovascular condition - monitor closely')
                    insights['follow_up_needed'].append('Cardiology follow-up appointment')
            
            # Medication review
            if current_medications:
                insights['medication_review'].append('Review medication adherence and potential interactions')
                if len(current_medications) > 5:
                    insights['risk_factors'].append('Polypharmacy - review medication necessity and interactions')
            
            return insights
            
        except Exception as e:
            print(f"❌ Error analyzing patient data: {e}")
            return None

# Global Doc chatbot manager
doc_manager = DocChatbotManager()

@doc_chatbot_bp.route('/')
@login_required
@section_required('healthpin')
def doc_chat():
    """Main Doc chatbot interface"""
    return render_template('healthpin/doc_chat.html', 
                         chatbot_name=doc_manager.name,
                         chatbot_description=doc_manager.description)

@doc_chatbot_bp.route('/whatsapp-setup')
@login_required
@section_required('healthpin')
def whatsapp_setup():
    """WhatsApp setup guide"""
    return render_template('healthpin/twilio_setup.html')

@doc_chatbot_bp.route('/chat', methods=['POST'])
@login_required
@section_required('healthpin')
def doc_chat_message():
    """Handle Doc chatbot messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Get conversation history
        conversation_history = data.get('conversation_history', [])
        
        # Generate response
        try:
            result = doc_manager.generate_response(message, conversation_history)
        except Exception as e:
            print(f"❌ Error in Doc manager generate_response: {e}")
            # Fallback response
            result = {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                'context_type': 'main',
                'response_time_ms': 0
            }
        
        # Save message to database
        chat_message = ChatMessage(
            sender_id=str(current_user.id),
            recipient_id='doc_healthpin',
            message_text=message,
            is_user_message=True
        )
        
        db.session.add(chat_message)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'context_type': result['context_type'],
            'response_time_ms': result['response_time_ms'],
            'message_id': chat_message.id
        })
        
    except Exception as e:
        print(f"❌ Error in Doc chat: {e}")
        return jsonify({'success': False, 'error': str(e)})

@doc_chatbot_bp.route('/history')
@login_required
@section_required('healthpin')
def doc_chat_history():
    """Get Doc chatbot conversation history"""
    try:
        messages = ChatMessage.query.filter_by(
            user_id=current_user.id,
            chatbot_type='doc_healthpin'
        ).order_by(ChatMessage.created_at.desc()).limit(50).all()
        
        history = []
        for msg in messages:
            history.append({
                'id': msg.id,
                'message': msg.message,
                'response': msg.response,
                'created_at': msg.created_at.isoformat(),
                'metadata': json.loads(msg.metadata) if msg.metadata else {}
            })
        
        return jsonify({'success': True, 'history': history})
        
    except Exception as e:
        print(f"❌ Error getting Doc history: {e}")
        return jsonify({'success': False, 'error': str(e)})

@doc_chatbot_bp.route('/feedback', methods=['POST'])
@login_required
@section_required('healthpin')
def doc_chat_feedback():
    """Record feedback for Doc chatbot responses"""
    try:
        data = request.get_json()
        message_id = data.get('message_id')
        rating = data.get('rating')  # 1-5 stars
        feedback = data.get('feedback', '')
        was_helpful = data.get('was_helpful')
        
        if not message_id:
            return jsonify({'success': False, 'error': 'Message ID required'})
        
        # Get the chat message
        chat_message = ChatMessage.query.get(message_id)
        if not chat_message or chat_message.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Message not found'})
        
        # Update message with feedback
        metadata = json.loads(chat_message.metadata) if chat_message.metadata else {}
        metadata.update({
            'user_rating': rating,
            'user_feedback': feedback,
            'was_helpful': was_helpful,
            'feedback_date': datetime.utcnow().isoformat()
        })
        
        chat_message.metadata = json.dumps(metadata)
        db.session.commit()
        
        # Also record in performance tracking
        if metadata.get('context_type'):
            prompt_map = {
                "main": "DOC_SYSTEM_PROMPT_MAIN",
                "patient_care": "DOC_SYSTEM_PROMPT_PATIENT_CARE",
                "technology": "DOC_SYSTEM_PROMPT_TECHNOLOGY",
                "analytics": "DOC_SYSTEM_PROMPT_ANALYTICS"
            }
            
            prompt_name = prompt_map.get(metadata['context_type'])
            if prompt_name:
                prompt = PromptTemplate.query.filter_by(name=prompt_name).first()
                if prompt:
                    performance_tracker.record_user_feedback(
                        prompt_id=prompt.id,
                        version_number=prompt.version,
                        user_rating=rating,
                        user_feedback=feedback,
                        was_helpful=was_helpful,
                        user_id=current_user.id,
                        session_id=session.get('session_id')
                    )
        
        return jsonify({'success': True, 'message': 'Feedback recorded successfully'})
        
    except Exception as e:
        print(f"❌ Error recording Doc feedback: {e}")
        return jsonify({'success': False, 'error': str(e)})

@doc_chatbot_bp.route('/stats')
@login_required
@section_required('healthpin')
def doc_chat_stats():
    """Get Doc chatbot usage statistics"""
    try:
        # Get user's Doc chat statistics
        total_messages = ChatMessage.query.filter_by(
            user_id=current_user.id,
            chatbot_type='doc_healthpin'
        ).count()
        
        # Get recent messages with context types
        recent_messages = ChatMessage.query.filter_by(
            user_id=current_user.id,
            chatbot_type='doc_healthpin'
        ).order_by(ChatMessage.created_at.desc()).limit(20).all()
        
        context_stats = {}
        for msg in recent_messages:
            if msg.metadata:
                metadata = json.loads(msg.metadata)
                context_type = metadata.get('context_type', 'main')
                context_stats[context_type] = context_stats.get(context_type, 0) + 1
        
        return jsonify({
            'success': True,
            'stats': {
                'total_messages': total_messages,
                'context_breakdown': context_stats,
                'chatbot_name': doc_manager.name,
                'specialization': doc_manager.specialization
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting Doc stats: {e}")
        return jsonify({'success': False, 'error': str(e)})
