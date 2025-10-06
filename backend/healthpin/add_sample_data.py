#!/usr/bin/env python3
"""
HealthPIN Sample Data Seeder
============================

Adds sample data to the HealthPIN system for demonstration purposes.
Run this script to populate the database with sample patients, doctors, and health records.
"""

import sys
import os
from datetime import datetime, date, timedelta
import random

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import db
from healthpin.models import (
    Patient, Doctor, HealthRecord, DoctorMatch, 
    FamilyNotification, Consultation, HealthNews
)

def add_sample_data():
    """Add sample data to HealthPIN"""
    
    print("🌱 Adding HealthPIN sample data...")
    
    # Sample patients
    patients_data = [
        {
            'phone_number': '+27821234567',
            'first_name': 'Thabo',
            'last_name': 'Mthembu',
            'date_of_birth': date(1985, 3, 15),
            'gender': 'Male',
            'language_preference': 'zu',
            'city': 'Johannesburg',
            'province': 'Gauteng',
            'country': 'South Africa',
            'preferred_specialties': ['General Practice', 'Cardiology'],
            'cultural_preferences': ['Traditional Medicine', 'Family Involvement'],
            'accessibility_needs': 'Wheelchair accessible',
            'emergency_contact_name': 'Nomsa Mthembu',
            'emergency_contact_phone': '+27821234568',
            'family_members': [
                {'name': 'Nomsa Mthembu', 'phone': '+27821234568', 'relationship': 'Wife'},
                {'name': 'Sipho Mthembu', 'phone': '+27821234569', 'relationship': 'Son'}
            ]
        },
        {
            'phone_number': '+27831234567',
            'first_name': 'Nomsa',
            'last_name': 'Dlamini',
            'date_of_birth': date(1990, 7, 22),
            'gender': 'Female',
            'language_preference': 'zu',
            'city': 'Durban',
            'province': 'KwaZulu-Natal',
            'country': 'South Africa',
            'preferred_specialties': ['Gynecology', 'Pediatrics'],
            'cultural_preferences': ['Traditional Medicine'],
            'accessibility_needs': None,
            'emergency_contact_name': 'Sipho Dlamini',
            'emergency_contact_phone': '+27831234568',
            'family_members': [
                {'name': 'Sipho Dlamini', 'phone': '+27831234568', 'relationship': 'Husband'},
                {'name': 'Thandi Dlamini', 'phone': '+27831234569', 'relationship': 'Mother'}
            ]
        },
        {
            'phone_number': '+27841234567',
            'first_name': 'John',
            'last_name': 'Smith',
            'date_of_birth': date(1978, 11, 8),
            'gender': 'Male',
            'language_preference': 'en',
            'city': 'Cape Town',
            'province': 'Western Cape',
            'country': 'South Africa',
            'preferred_specialties': ['Cardiology', 'Internal Medicine'],
            'cultural_preferences': ['Western Medicine'],
            'accessibility_needs': None,
            'emergency_contact_name': 'Mary Smith',
            'emergency_contact_phone': '+27841234568',
            'family_members': [
                {'name': 'Mary Smith', 'phone': '+27841234568', 'relationship': 'Wife'},
                {'name': 'David Smith', 'phone': '+27841234569', 'relationship': 'Son'}
            ]
        }
    ]
    
    # Sample doctors
    doctors_data = [
        {
            'first_name': 'Dr. Sarah',
            'last_name': 'Johnson',
            'title': 'Dr.',
            'specialties': ['General Practice', 'Family Medicine'],
            'phone_number': '+27851234567',
            'email': 'sarah.johnson@healthpin.co.za',
            'address': '123 Medical Centre, Johannesburg',
            'city': 'Johannesburg',
            'province': 'Gauteng',
            'country': 'South Africa',
            'languages': ['English', 'Afrikaans'],
            'cultural_competence': 'Experience with diverse cultural backgrounds',
            'accessibility_options': 'Wheelchair accessible, sign language interpreter available',
            'rating': 4.8,
            'is_verified': True
        },
        {
            'first_name': 'Dr. Themba',
            'last_name': 'Mkhize',
            'title': 'Dr.',
            'specialties': ['Cardiology', 'Internal Medicine'],
            'phone_number': '+27861234567',
            'email': 'themba.mkhize@healthpin.co.za',
            'address': '456 Heart Clinic, Durban',
            'city': 'Durban',
            'province': 'KwaZulu-Natal',
            'country': 'South Africa',
            'languages': ['English', 'isiZulu', 'isiXhosa'],
            'cultural_competence': 'Traditional medicine integration, family-centered care',
            'accessibility_options': 'Wheelchair accessible',
            'rating': 4.9,
            'is_verified': True
        },
        {
            'first_name': 'Dr. Aisha',
            'last_name': 'Hassan',
            'title': 'Dr.',
            'specialties': ['Gynecology', 'Obstetrics'],
            'phone_number': '+27871234567',
            'email': 'aisha.hassan@healthpin.co.za',
            'address': '789 Women\'s Health, Cape Town',
            'city': 'Cape Town',
            'province': 'Western Cape',
            'country': 'South Africa',
            'languages': ['English', 'Afrikaans', 'Arabic'],
            'cultural_competence': 'Muslim-friendly care, cultural sensitivity training',
            'accessibility_options': 'Wheelchair accessible, private consultation rooms',
            'rating': 4.7,
            'is_verified': True
        }
    ]
    
    # Add patients
    patients = []
    for patient_data in patients_data:
        # Check if patient already exists
        existing_patient = Patient.query.filter_by(phone_number=patient_data['phone_number']).first()
        if not existing_patient:
            patient = Patient(**patient_data)
            db.session.add(patient)
            patients.append(patient)
            print(f"✅ Added patient: {patient.first_name} {patient.last_name}")
        else:
            patients.append(existing_patient)
            print(f"ℹ️  Patient already exists: {existing_patient.first_name} {existing_patient.last_name}")
    
    # Add doctors
    doctors = []
    for doctor_data in doctors_data:
        # Check if doctor already exists
        existing_doctor = Doctor.query.filter_by(phone_number=doctor_data['phone_number']).first()
        if not existing_doctor:
            doctor = Doctor(**doctor_data)
            db.session.add(doctor)
            doctors.append(doctor)
            print(f"✅ Added doctor: {doctor.title} {doctor.first_name} {doctor.last_name}")
        else:
            doctors.append(existing_doctor)
            print(f"ℹ️  Doctor already exists: {existing_doctor.title} {existing_doctor.first_name} {existing_doctor.last_name}")
    
    # Commit patients and doctors first
    db.session.commit()
    
    # Add health records
    health_records_data = [
        {
            'patient_id': patients[0].id,
            'record_type': 'consultation',
            'title': 'Annual Checkup',
            'description': 'Routine annual health checkup',
            'content': 'Patient in good health. Blood pressure normal. Recommended regular exercise and healthy diet.',
            'recorded_at': datetime.utcnow() - timedelta(days=30)
        },
        {
            'patient_id': patients[1].id,
            'record_type': 'diagnosis',
            'title': 'Pregnancy Checkup',
            'description': 'Regular pregnancy monitoring',
            'content': 'Pregnancy progressing normally. Fetal heartbeat strong. Next appointment in 2 weeks.',
            'recorded_at': datetime.utcnow() - timedelta(days=15)
        },
        {
            'patient_id': patients[2].id,
            'record_type': 'lab_result',
            'title': 'Blood Test Results',
            'description': 'Comprehensive blood panel',
            'content': 'Cholesterol levels slightly elevated. Recommended dietary changes and follow-up in 3 months.',
            'recorded_at': datetime.utcnow() - timedelta(days=7)
        }
    ]
    
    for record_data in health_records_data:
        # Check if record already exists
        existing_record = HealthRecord.query.filter_by(
            patient_id=record_data['patient_id'],
            title=record_data['title']
        ).first()
        if not existing_record:
            health_record = HealthRecord(**record_data)
            db.session.add(health_record)
            print(f"✅ Added health record: {health_record.title}")
        else:
            print(f"ℹ️  Health record already exists: {existing_record.title}")
    
    # Add doctor matches
    if patients and doctors:
        match_data = {
            'patient_id': patients[0].id,
            'doctor_id': doctors[1].id,  # Cardiologist for patient with heart concerns
            'match_score': 0.95,
            'match_reasoning': 'Patient has heart-related concerns and doctor specializes in cardiology. Both located in same region.',
            'consultation_scheduled': True,
            'scheduled_at': datetime.utcnow() + timedelta(days=7)
        }
        
        existing_match = DoctorMatch.query.filter_by(
            patient_id=match_data['patient_id'],
            doctor_id=match_data['doctor_id']
        ).first()
        if not existing_match:
            doctor_match = DoctorMatch(**match_data)
            db.session.add(doctor_match)
            print(f"✅ Added doctor match: Patient {patients[0].first_name} matched with Dr. {doctors[1].last_name}")
        else:
            print(f"ℹ️  Doctor match already exists")
    
    # Add health news
    news_data = [
        {
            'title': 'Managing Diabetes in South Africa',
            'summary': 'Tips for managing diabetes with traditional and modern medicine approaches',
            'content': 'Diabetes management in South Africa requires a combination of modern medical treatment and cultural considerations...',
            'category': 'diabetes',
            'tags': ['diabetes', 'management', 'traditional medicine'],
            'language': 'en',
            'is_sponsored': False
        },
        {
            'title': 'Ukulungisa i-Diabetes eNingizimu Afrika',
            'summary': 'Amacebo okulungisa i-diabetes ngendlela yezokwelapha zendabuko nezamuhla',
            'content': 'Ukulungisa i-diabetes eNingizimu Afrika kudinga ukuhlanganiswa kwezokwelapha zanamuhla kanye nezinto zamasiko...',
            'category': 'diabetes',
            'tags': ['diabetes', 'ukulungisa', 'izokwelapha zendabuko'],
            'language': 'zu',
            'is_sponsored': False
        }
    ]
    
    for news_item in news_data:
        existing_news = HealthNews.query.filter_by(title=news_item['title']).first()
        if not existing_news:
            health_news = HealthNews(**news_item)
            db.session.add(health_news)
            print(f"✅ Added health news: {health_news.title}")
        else:
            print(f"ℹ️  Health news already exists: {existing_news.title}")
    
    # Commit all changes
    db.session.commit()
    
    print("🎉 HealthPIN sample data added successfully!")
    print(f"📊 Summary:")
    print(f"   - Patients: {len(patients)}")
    print(f"   - Doctors: {len(doctors)}")
    print(f"   - Health Records: {len(health_records_data)}")
    print(f"   - Doctor Matches: 1")
    print(f"   - Health News: {len(news_data)}")

if __name__ == '__main__':
    # This script should be run from the backend directory
    # or with the proper Flask app context
    print("⚠️  This script should be run with Flask app context")
    print("   Use: python -c \"from healthpin.add_sample_data import add_sample_data; add_sample_data()\"")

