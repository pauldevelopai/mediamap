"""
HealthPIN Sample Data Seeder
===========================

Creates sample data for HealthPIN demonstration.
"""

import sys
import os
from datetime import datetime, date, timedelta

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON, Date
from sqlalchemy.orm import relationship

# Define HealthPIN models inline to avoid import issues
class Patient(db.Model):
    """Patient model for HealthPIN platform"""
    __tablename__ = 'healthpin_patients'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)
    whatsapp_id = db.Column(db.String(50), unique=True, nullable=True)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    language_preference = db.Column(db.String(10), default='en')
    
    # Location
    city = db.Column(db.String(100), nullable=True)
    province = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), default='South Africa')
    
    # Health Preferences
    preferred_specialties = db.Column(JSON, nullable=True)
    cultural_preferences = db.Column(JSON, nullable=True)
    accessibility_needs = db.Column(Text, nullable=True)
    
    # Family Connections
    emergency_contact_name = db.Column(db.String(200), nullable=True)
    emergency_contact_phone = db.Column(db.String(20), nullable=True)
    family_members = db.Column(JSON, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Doctor(db.Model):
    """Doctor model for HealthPIN platform"""
    __tablename__ = 'healthpin_doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(50), nullable=True)
    
    # Professional Information
    medical_license = db.Column(db.String(100), unique=True, nullable=False)
    specialties = db.Column(JSON, nullable=False)
    qualifications = db.Column(JSON, nullable=True)
    years_experience = db.Column(db.Integer, nullable=True)
    
    # Practice Information
    practice_name = db.Column(db.String(200), nullable=True)
    practice_type = db.Column(db.String(50), nullable=True)
    languages_spoken = db.Column(JSON, nullable=True)
    
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
    availability_schedule = db.Column(JSON, nullable=True)
    
    # AI Matching Data
    patient_ratings = db.Column(JSON, nullable=True)
    cultural_sensitivity_score = db.Column(db.Float, default=0.0)
    accessibility_score = db.Column(db.Float, default=0.0)
    communication_style = db.Column(db.String(50), nullable=True)
    
    # Status
    is_verified = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class HealthRecord(db.Model):
    """Health record model for HealthBank functionality"""
    __tablename__ = 'healthpin_health_records'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('healthpin_patients.id'), nullable=False)
    
    # Record Information
    record_type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Medical Details
    diagnosis_code = db.Column(db.String(20), nullable=True)
    symptoms = db.Column(JSON, nullable=True)
    medications = db.Column(JSON, nullable=True)
    dosages = db.Column(JSON, nullable=True)
    lab_results = db.Column(JSON, nullable=True)
    
    # Provider Information
    doctor_name = db.Column(db.String(200), nullable=True)
    facility_name = db.Column(db.String(200), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('healthpin_doctors.id'), nullable=True)
    
    # Dates
    record_date = db.Column(db.Date, nullable=False)
    follow_up_date = db.Column(db.Date, nullable=True)
    
    # Attachments
    attachments = db.Column(JSON, nullable=True)
    
    # Privacy
    is_private = db.Column(db.Boolean, default=False)
    share_with_family = db.Column(db.Boolean, default=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class HealthNews(db.Model):
    """HealthNews content for monetization"""
    __tablename__ = 'healthpin_health_news'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Content Information
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    
    # Categorization
    category = db.Column(db.String(50), nullable=False)
    target_audience = db.Column(db.String(50), nullable=False)
    language = db.Column(db.String(10), default='en')
    
    # Personalization
    age_group = db.Column(db.String(20), nullable=True)
    gender_specific = db.Column(db.String(10), nullable=True)
    location_specific = db.Column(db.String(100), nullable=True)
    
    # Monetization
    is_sponsored = db.Column(db.Boolean, default=False)
    sponsor_name = db.Column(db.String(200), nullable=True)
    sponsor_type = db.Column(db.String(50), nullable=True)
    
    # Engagement
    views_count = db.Column(db.Integer, default=0)
    shares_count = db.Column(db.Integer, default=0)
    likes_count = db.Column(db.Integer, default=0)
    
    # Status
    is_published = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def seed_healthpin_data():
    """Seed HealthPIN with sample data"""
    
    print("🌱 Seeding HealthPIN sample data...")
    
    # Sample patients
    patients_data = [
        {
            'phone_number': '+27821234567',
            'first_name': 'Thabo',
            'last_name': 'Mthembu',
            'date_of_birth': date(1985, 3, 15),
            'gender': 'male',
            'language_preference': 'zu',
            'city': 'Durban',
            'province': 'KwaZulu-Natal',
            'preferred_specialties': ['Cardiology', 'General Practice'],
            'cultural_preferences': ['Traditional Medicine', 'Family Involvement'],
            'emergency_contact_name': 'Nomsa Mthembu',
            'emergency_contact_phone': '+27821234568',
            'family_members': [
                {'name': 'Nomsa Mthembu', 'phone': '+27821234568', 'relationship': 'Wife'},
                {'name': 'Sipho Mthembu', 'phone': '+27821234569', 'relationship': 'Son'}
            ]
        },
        {
            'phone_number': '+27831234567',
            'first_name': 'Sarah',
            'last_name': 'Johnson',
            'date_of_birth': date(1992, 7, 22),
            'gender': 'female',
            'language_preference': 'en',
            'city': 'Cape Town',
            'province': 'Western Cape',
            'preferred_specialties': ['Gynecology', 'Endocrinology'],
            'cultural_preferences': ['Western Medicine', 'Privacy'],
            'emergency_contact_name': 'Michael Johnson',
            'emergency_contact_phone': '+27831234568',
            'family_members': [
                {'name': 'Michael Johnson', 'phone': '+27831234568', 'relationship': 'Husband'},
                {'name': 'Emma Johnson', 'phone': '+27831234569', 'relationship': 'Mother'}
            ]
        },
        {
            'phone_number': '+27841234567',
            'first_name': 'Mandla',
            'last_name': 'Ndlovu',
            'date_of_birth': date(1978, 11, 8),
            'gender': 'male',
            'language_preference': 'xh',
            'city': 'Port Elizabeth',
            'province': 'Eastern Cape',
            'preferred_specialties': ['Orthopedics', 'Sports Medicine'],
            'cultural_preferences': ['Traditional Healing', 'Community Support'],
            'emergency_contact_name': 'Nolwazi Ndlovu',
            'emergency_contact_phone': '+27841234568',
            'family_members': [
                {'name': 'Nolwazi Ndlovu', 'phone': '+27841234568', 'relationship': 'Sister'},
                {'name': 'Bongani Ndlovu', 'phone': '+27841234569', 'relationship': 'Brother'}
            ]
        }
    ]
    
    # Sample doctors
    doctors_data = [
        {
            'first_name': 'Dr. James',
            'last_name': 'Mitchell',
            'title': 'Dr.',
            'medical_license': 'MP123456',
            'specialties': ['Cardiology', 'Internal Medicine'],
            'qualifications': ['MBChB', 'FCP(SA)', 'Cardiology Fellowship'],
            'years_experience': 15,
            'practice_name': 'Heart Care Clinic',
            'practice_type': 'Private',
            'languages_spoken': ['English', 'Afrikaans'],
            'city': 'Durban',
            'province': 'KwaZulu-Natal',
            'address': '123 Medical Drive, Durban',
            'latitude': -29.8587,
            'longitude': 31.0218,
            'phone': '+27821234000',
            'email': 'james.mitchell@heartcare.co.za',
            'whatsapp_available': True,
            'consultation_fee': 800.00,
            'accepts_medical_aid': True,
            'cultural_sensitivity_score': 85.0,
            'accessibility_score': 90.0,
            'communication_style': 'Professional',
            'is_verified': True
        },
        {
            'first_name': 'Dr. Nomsa',
            'last_name': 'Dlamini',
            'title': 'Dr.',
            'medical_license': 'MP234567',
            'specialties': ['Gynecology', 'Obstetrics'],
            'qualifications': ['MBChB', 'MMed(Obstetrics & Gynecology)'],
            'years_experience': 12,
            'practice_name': 'Women\'s Health Center',
            'practice_type': 'Private',
            'languages_spoken': ['English', 'isiZulu', 'isiXhosa'],
            'city': 'Cape Town',
            'province': 'Western Cape',
            'address': '456 Health Street, Cape Town',
            'latitude': -33.9249,
            'longitude': 18.4241,
            'phone': '+27831234000',
            'email': 'nomsa.dlamini@womenshealth.co.za',
            'whatsapp_available': True,
            'consultation_fee': 750.00,
            'accepts_medical_aid': True,
            'cultural_sensitivity_score': 95.0,
            'accessibility_score': 85.0,
            'communication_style': 'Warm',
            'is_verified': True
        },
        {
            'first_name': 'Dr. Peter',
            'last_name': 'Van Der Merwe',
            'title': 'Dr.',
            'medical_license': 'MP345678',
            'specialties': ['Orthopedics', 'Sports Medicine'],
            'qualifications': ['MBChB', 'MMed(Orthopedics)', 'Sports Medicine Diploma'],
            'years_experience': 18,
            'practice_name': 'Sports & Orthopedic Clinic',
            'practice_type': 'Private',
            'languages_spoken': ['English', 'Afrikaans'],
            'city': 'Port Elizabeth',
            'province': 'Eastern Cape',
            'address': '789 Sports Avenue, Port Elizabeth',
            'latitude': -33.9608,
            'longitude': 25.6022,
            'phone': '+27841234000',
            'email': 'peter.vandermerwe@sportsclinic.co.za',
            'whatsapp_available': False,
            'consultation_fee': 900.00,
            'accepts_medical_aid': True,
            'cultural_sensitivity_score': 75.0,
            'accessibility_score': 95.0,
            'communication_style': 'Direct',
            'is_verified': True
        },
        {
            'first_name': 'Dr. Thandi',
            'last_name': 'Mthembu',
            'title': 'Dr.',
            'medical_license': 'MP456789',
            'specialties': ['General Practice', 'Family Medicine'],
            'qualifications': ['MBChB', 'Dip(Obstetrics)', 'Family Medicine Certificate'],
            'years_experience': 10,
            'practice_name': 'Community Health Center',
            'practice_type': 'Public',
            'languages_spoken': ['English', 'isiZulu', 'isiXhosa'],
            'city': 'Durban',
            'province': 'KwaZulu-Natal',
            'address': '321 Community Road, Durban',
            'latitude': -29.8587,
            'longitude': 31.0218,
            'phone': '+27821234001',
            'email': 'thandi.mthembu@communityhealth.gov.za',
            'whatsapp_available': True,
            'consultation_fee': 0.00,  # Public health
            'accepts_medical_aid': True,
            'cultural_sensitivity_score': 98.0,
            'accessibility_score': 80.0,
            'communication_style': 'Compassionate',
            'is_verified': True
        }
    ]
    
    # Create patients
    patients = []
    for patient_data in patients_data:
        patient = Patient(**patient_data)
        db.session.add(patient)
        patients.append(patient)
    
    # Create doctors
    doctors = []
    for doctor_data in doctors_data:
        doctor = Doctor(**doctor_data)
        db.session.add(doctor)
        doctors.append(doctor)
    
    db.session.commit()
    print(f"✅ Created {len(patients)} patients and {len(doctors)} doctors")
    
    # Create sample health records
    health_records_data = [
        {
            'patient_id': patients[0].id,
            'record_type': 'diagnosis',
            'title': 'Hypertension Diagnosis',
            'description': 'Patient diagnosed with stage 1 hypertension',
            'diagnosis_code': 'I10',
            'symptoms': ['High blood pressure', 'Headaches', 'Dizziness'],
            'medications': ['Amlodipine 5mg', 'Lisinopril 10mg'],
            'dosages': ['Once daily', 'Once daily'],
            'doctor_name': 'Dr. James Mitchell',
            'facility_name': 'Heart Care Clinic',
            'record_date': date.today() - timedelta(days=30),
            'follow_up_date': date.today() + timedelta(days=30)
        },
        {
            'patient_id': patients[1].id,
            'record_type': 'prescription',
            'title': 'Birth Control Prescription',
            'description': 'Prescribed oral contraceptive for family planning',
            'medications': ['Ethinylestradiol + Levonorgestrel'],
            'dosages': ['One tablet daily'],
            'doctor_name': 'Dr. Nomsa Dlamini',
            'facility_name': 'Women\'s Health Center',
            'record_date': date.today() - timedelta(days=15),
            'follow_up_date': date.today() + timedelta(days=90)
        },
        {
            'patient_id': patients[2].id,
            'record_type': 'lab_result',
            'title': 'Blood Test Results',
            'description': 'Routine blood work showing elevated cholesterol',
            'lab_results': {
                'total_cholesterol': '6.2 mmol/L',
                'hdl_cholesterol': '1.1 mmol/L',
                'ldl_cholesterol': '4.8 mmol/L',
                'triglycerides': '2.1 mmol/L'
            },
            'doctor_name': 'Dr. Thandi Mthembu',
            'facility_name': 'Community Health Center',
            'record_date': date.today() - timedelta(days=7)
        }
    ]
    
    for record_data in health_records_data:
        record = HealthRecord(**record_data)
        db.session.add(record)
    
    db.session.commit()
    print(f"✅ Created {len(health_records_data)} health records")
    
    # Create sample health news
    health_news_data = [
        {
            'title': 'Managing Diabetes During Pregnancy',
            'content': 'Pregnancy with diabetes requires careful monitoring and management. Here are key strategies for maintaining healthy blood sugar levels...',
            'summary': 'Essential tips for managing diabetes during pregnancy to ensure both mother and baby stay healthy.',
            'category': 'specialty',
            'target_audience': 'diabetic',
            'language': 'en',
            'age_group': 'adult',
            'gender_specific': 'female',
            'is_published': True
        },
        {
            'title': 'Ukulawula i-Diabetes Ngesikhathi Sokukhulelwa',
            'content': 'Ukukhulelwa nge-diabetes kudinga ukuqapha nokulawula ngokucophelela. Nazi izindlela ezibalulekile zokugcina amazinga e-blood sugar aphilayo...',
            'summary': 'Amathiphu abalulekile okulawula i-diabetes ngesikhathi sokukhulelwa ukuze kuqinisekiswe ukuthi umama nengane bahlala bephilile.',
            'category': 'specialty',
            'target_audience': 'diabetic',
            'language': 'zu',
            'age_group': 'adult',
            'gender_specific': 'female',
            'is_published': True
        },
        {
            'title': 'Heart Health: Exercise for Cardiovascular Fitness',
            'content': 'Regular exercise is crucial for maintaining heart health. Learn about the best exercises for cardiovascular fitness and how to start safely...',
            'summary': 'Discover effective exercises to improve heart health and cardiovascular fitness.',
            'category': 'lifestyle',
            'target_audience': 'general',
            'language': 'en',
            'age_group': 'adult',
            'gender_specific': 'all',
            'is_published': True
        }
    ]
    
    for news_data in health_news_data:
        news = HealthNews(**news_data)
        db.session.add(news)
    
    db.session.commit()
    print(f"✅ Created {len(health_news_data)} health news articles")
    
    print("🎉 HealthPIN sample data seeding completed!")
    
    return {
        'patients': len(patients),
        'doctors': len(doctors),
        'health_records': len(health_records_data),
        'health_news': len(health_news_data)
    }

if __name__ == '__main__':
    from app import app
    with app.app_context():
        seed_healthpin_data()
