"""
HealthPIN Database Models
========================

Core models for the HealthPIN platform including patients, doctors, 
health records, and family connections.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.models import db

class Patient(db.Model):
    """Patient model for HealthPIN platform"""
    __tablename__ = 'healthpin_patients'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Link to existing user
    phone_number = db.Column(db.String(20), unique=True, nullable=False)
    whatsapp_id = db.Column(db.String(50), unique=True, nullable=True)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    language_preference = db.Column(db.String(10), default='en')  # en, zu, xh, sn
    
    # Location
    city = db.Column(db.String(100), nullable=True)
    province = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), default='South Africa')
    
    # Health Preferences
    preferred_specialties = db.Column(JSON, nullable=True)  # List of preferred medical specialties
    cultural_preferences = db.Column(JSON, nullable=True)  # Cultural/religious preferences
    accessibility_needs = db.Column(Text, nullable=True)
    
    # Family Connections
    emergency_contact_name = db.Column(db.String(200), nullable=True)
    emergency_contact_phone = db.Column(db.String(20), nullable=True)
    family_members = db.Column(JSON, nullable=True)  # List of family member contacts
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships - using fully qualified module paths
    health_records = db.relationship('backend.healthpin.models.HealthRecord', cascade='all, delete-orphan', overlaps="health_records")
    doctor_matches = db.relationship('backend.healthpin.models.DoctorMatch', cascade='all, delete-orphan', overlaps="doctor_matches")
    family_notifications = db.relationship('backend.healthpin.models.FamilyNotification', cascade='all, delete-orphan', overlaps="family_notifications")
    
    def to_dict(self):
        return {
            'id': self.id,
            'phone_number': self.phone_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name} {self.last_name}",
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'language_preference': self.language_preference,
            'city': self.city,
            'province': self.province,
            'country': self.country,
            'preferred_specialties': self.preferred_specialties or [],
            'cultural_preferences': self.cultural_preferences or [],
            'accessibility_needs': self.accessibility_needs,
            'emergency_contact_name': self.emergency_contact_name,
            'emergency_contact_phone': self.emergency_contact_phone,
            'family_members': self.family_members or [],
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Doctor(db.Model):
    """Doctor model for HealthPIN platform"""
    __tablename__ = 'healthpin_doctors'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(50), nullable=True)  # Dr., Prof., etc.
    
    # Professional Information
    medical_license = db.Column(db.String(100), unique=True, nullable=False)
    specialties = db.Column(JSON, nullable=False)  # List of medical specialties
    qualifications = db.Column(JSON, nullable=True)  # List of qualifications
    years_experience = db.Column(db.Integer, nullable=True)
    
    # Practice Information
    practice_name = db.Column(db.String(200), nullable=True)
    practice_type = db.Column(db.String(50), nullable=True)  # Private, Public, Clinic, Hospital
    languages_spoken = db.Column(JSON, nullable=True)  # List of languages
    
    # Location
    city = db.Column(db.String(100), nullable=False)
    province = db.Column(db.String(100), nullable=False)
    address = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    
    # Contact Information
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    whatsapp_available = db.Column(db.Boolean, default=False)
    
    # Availability and Pricing
    consultation_fee = db.Column(db.Float, nullable=True)
    accepts_medical_aid = db.Column(db.Boolean, default=True)
    availability_schedule = db.Column(JSON, nullable=True)  # Weekly schedule
    
    # AI Matching Data
    patient_ratings = db.Column(JSON, nullable=True)  # Patient feedback and ratings
    cultural_sensitivity_score = db.Column(db.Float, default=0.0)
    accessibility_score = db.Column(db.Float, default=0.0)
    communication_style = db.Column(db.String(50), nullable=True)  # Formal, Casual, etc.
    
    # Status
    is_verified = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships will be defined on child models using backref

    def to_dict(self):
        return {
            'id': self.id,
            'full_name': f"{self.title or ''} {self.first_name} {self.last_name}".strip(),
            'first_name': self.first_name,
            'last_name': self.last_name,
            'title': self.title,
            'medical_license': self.medical_license,
            'specialties': self.specialties or [],
            'qualifications': self.qualifications or [],
            'years_experience': self.years_experience,
            'practice_name': self.practice_name,
            'practice_type': self.practice_type,
            'languages_spoken': self.languages_spoken or [],
            'city': self.city,
            'province': self.province,
            'address': self.address,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'phone': self.phone,
            'email': self.email,
            'whatsapp_available': self.whatsapp_available,
            'consultation_fee': self.consultation_fee,
            'accepts_medical_aid': self.accepts_medical_aid,
            'availability_schedule': self.availability_schedule or {},
            'patient_ratings': self.patient_ratings or {},
            'cultural_sensitivity_score': self.cultural_sensitivity_score,
            'accessibility_score': self.accessibility_score,
            'communication_style': self.communication_style,
            'is_verified': self.is_verified,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class HealthRecord(db.Model):
    """Health record model for HealthBank functionality"""
    __tablename__ = 'healthpin_health_records'
    __table_args__ = {'extend_existing': True}
    __mapper_args__ = {'polymorphic_identity': 'healthpin_health_record'}
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('healthpin_patients.id'), nullable=False)
    
    # Record Information
    record_type = db.Column(db.String(50), nullable=False)  # diagnosis, prescription, lab_result, scan, note
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Medical Details
    diagnosis_code = db.Column(db.String(20), nullable=True)  # ICD-10 code
    symptoms = db.Column(JSON, nullable=True)  # List of symptoms
    medications = db.Column(JSON, nullable=True)  # List of medications
    dosages = db.Column(JSON, nullable=True)  # Medication dosages
    lab_results = db.Column(JSON, nullable=True)  # Lab test results
    
    # Provider Information
    doctor_name = db.Column(db.String(200), nullable=True)
    facility_name = db.Column(db.String(200), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('healthpin_doctors.id'), nullable=True)
    
    # Dates
    record_date = db.Column(db.Date, nullable=False)
    follow_up_date = db.Column(db.Date, nullable=True)
    
    # Attachments
    attachments = db.Column(JSON, nullable=True)  # List of file paths/URLs
    
    # Privacy
    is_private = db.Column(db.Boolean, default=False)
    share_with_family = db.Column(db.Boolean, default=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    doctor = db.relationship(
        'backend.healthpin.models.Doctor',
        foreign_keys=[doctor_id]
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'record_type': self.record_type,
            'title': self.title,
            'description': self.description,
            'diagnosis_code': self.diagnosis_code,
            'symptoms': self.symptoms or [],
            'medications': self.medications or [],
            'dosages': self.dosages or [],
            'lab_results': self.lab_results or {},
            'doctor_name': self.doctor_name,
            'facility_name': self.facility_name,
            'doctor_id': self.doctor_id,
            'record_date': self.record_date.isoformat() if self.record_date else None,
            'follow_up_date': self.follow_up_date.isoformat() if self.follow_up_date else None,
            'attachments': self.attachments or [],
            'is_private': self.is_private,
            'share_with_family': self.share_with_family,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class DoctorMatch(db.Model):
    """AI-powered doctor matching results"""
    __tablename__ = 'healthpin_doctor_matches'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('healthpin_patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('healthpin_doctors.id'), nullable=False)
    
    # Matching Scores
    overall_score = db.Column(db.Float, nullable=False)  # 0-100
    specialty_match = db.Column(db.Float, nullable=False)  # 0-100
    location_score = db.Column(db.Float, nullable=False)  # 0-100
    cultural_fit = db.Column(db.Float, nullable=False)  # 0-100
    accessibility_score = db.Column(db.Float, nullable=False)  # 0-100
    communication_style = db.Column(db.Float, nullable=False)  # 0-100
    
    # AI Analysis
    ai_reasoning = db.Column(Text, nullable=True)  # AI explanation for the match
    confidence_level = db.Column(db.Float, nullable=False)  # 0-100
    
    # Patient Interaction
    patient_viewed = db.Column(db.Boolean, default=False)
    patient_selected = db.Column(db.Boolean, default=False)
    consultation_scheduled = db.Column(db.Boolean, default=False)
    
    # Feedback
    patient_feedback = db.Column(db.Text, nullable=True)
    patient_rating = db.Column(db.Integer, nullable=True)  # 1-5 stars
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    doctor = db.relationship(
        'backend.healthpin.models.Doctor',
        foreign_keys=[doctor_id]
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'overall_score': self.overall_score,
            'specialty_match': self.specialty_match,
            'location_score': self.location_score,
            'cultural_fit': self.cultural_fit,
            'accessibility_score': self.accessibility_score,
            'communication_style': self.communication_style,
            'ai_reasoning': self.ai_reasoning,
            'confidence_level': self.confidence_level,
            'patient_viewed': self.patient_viewed,
            'patient_selected': self.patient_selected,
            'consultation_scheduled': self.consultation_scheduled,
            'patient_feedback': self.patient_feedback,
            'patient_rating': self.patient_rating,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class FamilyNotification(db.Model):
    """Family notification system for FamilyHealth"""
    __tablename__ = 'healthpin_family_notifications'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('healthpin_patients.id'), nullable=False)
    
    # Notification Details
    notification_type = db.Column(db.String(50), nullable=False)  # appointment, medication, emergency, update
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    
    # AI Generated Content
    ai_summary = db.Column(db.Text, nullable=True)  # AI-generated summary of health update
    urgency_level = db.Column(db.String(20), default='normal')  # low, normal, high, emergency
    
    # Recipients
    family_members = db.Column(JSON, nullable=False)  # List of family member contacts
    sent_to = db.Column(JSON, nullable=True)  # Track who received the notification
    
    # Delivery
    delivery_method = db.Column(db.String(20), default='whatsapp')  # whatsapp, sms, email
    delivery_status = db.Column(db.String(20), default='pending')  # pending, sent, delivered, failed
    
    # Related Records
    health_record_id = db.Column(db.Integer, db.ForeignKey('healthpin_health_records.id'), nullable=True)
    consultation_id = db.Column(db.Integer, db.ForeignKey('healthpin_consultations.id'), nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    health_record = db.relationship('backend.healthpin.models.HealthRecord', overlaps="health_record")
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'notification_type': self.notification_type,
            'title': self.title,
            'message': self.message,
            'ai_summary': self.ai_summary,
            'urgency_level': self.urgency_level,
            'family_members': self.family_members or [],
            'sent_to': self.sent_to or [],
            'delivery_method': self.delivery_method,
            'delivery_status': self.delivery_status,
            'health_record_id': self.health_record_id,
            'consultation_id': self.consultation_id,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Consultation(db.Model):
    """Consultation/appointment model"""
    __tablename__ = 'healthpin_consultations'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('healthpin_patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('healthpin_doctors.id'), nullable=False)
    
    # Appointment Details
    appointment_type = db.Column(db.String(50), nullable=False)  # in_person, video, phone
    scheduled_date = db.Column(db.DateTime, nullable=False)
    duration_minutes = db.Column(db.Integer, default=30)
    
    # Status
    status = db.Column(db.String(20), default='scheduled')  # scheduled, confirmed, completed, cancelled, no_show
    
    # Consultation Notes
    chief_complaint = db.Column(db.Text, nullable=True)
    diagnosis = db.Column(db.Text, nullable=True)
    treatment_plan = db.Column(db.Text, nullable=True)
    prescriptions = db.Column(JSON, nullable=True)
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.Date, nullable=True)
    
    # Feedback
    patient_satisfaction = db.Column(db.Integer, nullable=True)  # 1-5 rating
    doctor_notes = db.Column(db.Text, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    family_notifications = db.relationship('backend.healthpin.models.FamilyNotification', overlaps="family_notifications")
    doctor = db.relationship(
        'backend.healthpin.models.Doctor',
        foreign_keys=[doctor_id]
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'appointment_type': self.appointment_type,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'duration_minutes': self.duration_minutes,
            'status': self.status,
            'chief_complaint': self.chief_complaint,
            'diagnosis': self.diagnosis,
            'treatment_plan': self.treatment_plan,
            'prescriptions': self.prescriptions or [],
            'follow_up_required': self.follow_up_required,
            'follow_up_date': self.follow_up_date.isoformat() if self.follow_up_date else None,
            'patient_satisfaction': self.patient_satisfaction,
            'doctor_notes': self.doctor_notes,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class HealthNews(db.Model):
    """HealthNews content for monetization"""
    __tablename__ = 'healthpin_health_news'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Content Information
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    
    # Categorization
    category = db.Column(db.String(50), nullable=False)  # general, specialty, medication, lifestyle
    target_audience = db.Column(db.String(50), nullable=False)  # general, diabetic, cardiac, etc.
    language = db.Column(db.String(10), default='en')
    
    # Personalization
    age_group = db.Column(db.String(20), nullable=True)  # child, adult, senior
    gender_specific = db.Column(db.String(10), nullable=True)  # male, female, all
    location_specific = db.Column(db.String(100), nullable=True)  # city, province, country
    
    # Monetization
    is_sponsored = db.Column(db.Boolean, default=False)
    sponsor_name = db.Column(db.String(200), nullable=True)
    sponsor_type = db.Column(db.String(50), nullable=True)  # pharmaceutical, clinic, insurance
    
    # Engagement
    views_count = db.Column(db.Integer, default=0)
    shares_count = db.Column(db.Integer, default=0)
    likes_count = db.Column(db.Integer, default=0)
    
    # Status
    is_published = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'category': self.category,
            'target_audience': self.target_audience,
            'language': self.language,
            'age_group': self.age_group,
            'gender_specific': self.gender_specific,
            'location_specific': self.location_specific,
            'is_sponsored': self.is_sponsored,
            'sponsor_name': self.sponsor_name,
            'sponsor_type': self.sponsor_type,
            'views_count': self.views_count,
            'shares_count': self.shares_count,
            'likes_count': self.likes_count,
            'is_published': self.is_published,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
