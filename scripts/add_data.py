#!/usr/bin/env python3
"""
AIMAP Data Addition Tool
CLI tool for adding people, companies, leads, research reports, and custom data
"""
import sys
import os
import json
from datetime import datetime, date
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

try:
    from backend.aimap.models import (
        Organisation, Person, Lead, ResearchReport, CustomData, 
        ConsultingProject, Metrics
    )
    from backend.app import app, db
except ImportError as e:
    print(f"❌ AIMAP modules not found: {e}")
    print("Please ensure the backend is properly set up.")
    sys.exit(1)

def init_database():
    """Initialize the database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")

def add_organisation():
    """Add a new organization"""
    print("\n🏢 Adding New Organization")
    print("=" * 40)
    
    name = input("Organization name: ").strip()
    if not name:
        print("❌ Organization name is required")
        return
    
    sector = input("Sector (Media/Communications/Finance/Healthcare): ").strip() or "Media"
    region = input("Region: ").strip()
    country = input("Country: ").strip()
    size_band = input("Size band (startup/small/medium/large/enterprise): ").strip()
    website_url = input("Website URL: ").strip()
    notes = input("Notes: ").strip()
    
    # Create organization
    org = Organisation(
        name=name,
        sector=sector,
        region=region,
        country=country,
        size_band=size_band,
        website_url=website_url,
        notes=notes,
        ai_tools=[]
    )
    
    db.session.add(org)
    db.session.commit()
    
    print(f"✅ Organization '{name}' added with ID: {org.id}")
    
    # Add initial metrics
    add_initial_metrics = input("\nAdd initial AI adoption metrics? (y/n): ").strip().lower()
    if add_initial_metrics == 'y':
        score = float(input("AI Adoption Score (0-100): ").strip() or "30")
        stage = input("Maturity Stage (Exploring/Piloting/Scaling/Optimizing): ").strip() or "Exploring"
        
        metrics = Metrics(
            organisation_id=org.id,
            ai_adoption_score=score,
            maturity_stage=stage,
            period=datetime.now().strftime("%Y-%m"),
            signals={},
            benchmark_bucket=f"{sector}:{region}:{size_band}"
        )
        
        db.session.add(metrics)
        db.session.commit()
        print(f"✅ Initial metrics added with score: {score}")

def add_person():
    """Add a new person/contact"""
    print("\n👤 Adding New Person/Contact")
    print("=" * 40)
    
    # List organizations
    orgs = Organisation.query.all()
    if not orgs:
        print("❌ No organizations found. Please add an organization first.")
        return
    
    print("\nAvailable organizations:")
    for org in orgs:
        print(f"  {org.id}: {org.name} ({org.sector})")
    
    org_id = input("\nOrganization ID: ").strip()
    try:
        org_id = int(org_id)
        org = Organisation.query.get(org_id)
        if not org:
            print("❌ Organization not found")
            return
    except ValueError:
        print("❌ Invalid organization ID")
        return
    
    first_name = input("First name: ").strip()
    last_name = input("Last name: ").strip()
    email = input("Email: ").strip()
    phone = input("Phone: ").strip()
    title = input("Title: ").strip()
    department = input("Department: ").strip()
    role = input("Role (Decision maker/Influencer/User): ").strip()
    linkedin_url = input("LinkedIn URL: ").strip()
    notes = input("Notes: ").strip()
    is_primary = input("Primary contact? (y/n): ").strip().lower() == 'y'
    
    person = Person(
        organisation_id=org_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone,
        title=title,
        department=department,
        role=role,
        linkedin_url=linkedin_url,
        notes=notes,
        is_primary_contact=is_primary
    )
    
    db.session.add(person)
    db.session.commit()
    
    print(f"✅ Person '{first_name} {last_name}' added with ID: {person.id}")

def add_lead():
    """Add a new lead/prospect"""
    print("\n🎯 Adding New Lead/Prospect")
    print("=" * 40)
    
    # List organizations
    orgs = Organisation.query.all()
    if not orgs:
        print("❌ No organizations found. Please add an organization first.")
        return
    
    print("\nAvailable organizations:")
    for org in orgs:
        print(f"  {org.id}: {org.name} ({org.sector})")
    
    org_id = input("\nOrganization ID: ").strip()
    try:
        org_id = int(org_id)
        org = Organisation.query.get(org_id)
        if not org:
            print("❌ Organization not found")
            return
    except ValueError:
        print("❌ Invalid organization ID")
        return
    
    status = input("Status (Prospect/Contacted/Qualified/Proposal/Negotiation): ").strip() or "Prospect"
    source = input("Source (Website/Referral/Cold outreach): ").strip()
    priority = input("Priority (High/Medium/Low): ").strip() or "Medium"
    estimated_value = input("Estimated value (USD): ").strip()
    probability = input("Probability (0-100%): ").strip()
    expected_close_date = input("Expected close date (YYYY-MM-DD): ").strip()
    assigned_to = input("Assigned to: ").strip()
    notes = input("Notes: ").strip()
    
    lead = Lead(
        organisation_id=org_id,
        status=status,
        source=source,
        priority=priority,
        estimated_value=float(estimated_value) if estimated_value else None,
        probability=float(probability) if probability else None,
        expected_close_date=datetime.strptime(expected_close_date, '%Y-%m-%d').date() if expected_close_date else None,
        assigned_to=assigned_to,
        notes=notes
    )
    
    db.session.add(lead)
    db.session.commit()
    
    print(f"✅ Lead added with ID: {lead.id}")

def add_research_report():
    """Add a new research report"""
    print("\n📊 Adding New Research Report")
    print("=" * 40)
    
    title = input("Report title: ").strip()
    if not title:
        print("❌ Report title is required")
        return
    
    description = input("Description: ").strip()
    report_type = input("Report type (Industry analysis/Case study/White paper): ").strip()
    file_path = input("File path: ").strip()
    author = input("Author: ").strip()
    publication_date = input("Publication date (YYYY-MM-DD): ").strip()
    tags_input = input("Tags (comma-separated): ").strip()
    summary = input("Summary: ").strip()
    
    # Parse tags
    tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
    
    # Optional organization association
    org_id = input("Organization ID (optional): ").strip()
    org_id = int(org_id) if org_id else None
    
    report = ResearchReport(
        organisation_id=org_id,
        title=title,
        description=description,
        report_type=report_type,
        file_path=file_path,
        file_size=0,  # Will be calculated if file exists
        file_type=Path(file_path).suffix if file_path else None,
        tags=tags,
        ai_insights={},
        summary=summary,
        author=author,
        publication_date=datetime.strptime(publication_date, '%Y-%m-%d').date() if publication_date else None
    )
    
    db.session.add(report)
    db.session.commit()
    
    print(f"✅ Research report '{title}' added with ID: {report.id}")

def add_custom_data():
    """Add custom data"""
    print("\n📝 Adding Custom Data")
    print("=" * 40)
    
    data_types = [
        'competitor', 'market_trend', 'tool_review', 'case_study',
        'industry_analysis', 'technology_assessment', 'vendor_evaluation',
        'best_practice', 'regulatory_update', 'event_notes'
    ]
    
    print("\nAvailable data types:")
    for i, dt in enumerate(data_types, 1):
        print(f"  {i}: {dt}")
    
    type_choice = input("\nSelect data type (1-10): ").strip()
    try:
        type_idx = int(type_choice) - 1
        data_type = data_types[type_idx]
    except (ValueError, IndexError):
        print("❌ Invalid choice")
        return
    
    title = input("Title: ").strip()
    if not title:
        print("❌ Title is required")
        return
    
    print("\nEnter content (JSON format, e.g., {\"key\": \"value\"}):")
    content_input = input("Content: ").strip()
    
    try:
        content = json.loads(content_input) if content_input else {}
    except json.JSONDecodeError:
        print("❌ Invalid JSON format")
        return
    
    tags_input = input("Tags (comma-separated): ").strip()
    tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
    
    metadata_input = input("Metadata (JSON format, optional): ").strip()
    try:
        metadata = json.loads(metadata_input) if metadata_input else {}
    except json.JSONDecodeError:
        print("❌ Invalid JSON format for metadata")
        return
    
    custom_data = CustomData(
        data_type=data_type,
        title=title,
        content=content,
        tags=tags,
        metadata=metadata
    )
    
    db.session.add(custom_data)
    db.session.commit()
    
    print(f"✅ Custom data '{title}' added with ID: {custom_data.id}")

def list_data():
    """List existing data"""
    print("\n📋 Data Summary")
    print("=" * 40)
    
    orgs = Organisation.query.count()
    people = Person.query.count()
    leads = Lead.query.count()
    reports = ResearchReport.query.count()
    custom_data = CustomData.query.count()
    
    print(f"Organizations: {orgs}")
    print(f"People/Contacts: {people}")
    print(f"Leads/Prospects: {leads}")
    print(f"Research Reports: {reports}")
    print(f"Custom Data: {custom_data}")
    
    if orgs > 0:
        print(f"\nRecent Organizations:")
        recent_orgs = Organisation.query.order_by(Organisation.created_at.desc()).limit(5).all()
        for org in recent_orgs:
            print(f"  {org.id}: {org.name} ({org.sector})")

def main():
    """Main CLI interface"""
    print("🎯 AIMAP Data Management Tool")
    print("=" * 40)
    
    # Initialize database
    init_database()
    
    while True:
        print("\nOptions:")
        print("1. Add Organization")
        print("2. Add Person/Contact")
        print("3. Add Lead/Prospect")
        print("4. Add Research Report")
        print("5. Add Custom Data")
        print("6. List Data Summary")
        print("0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            add_organisation()
        elif choice == '2':
            add_person()
        elif choice == '3':
            add_lead()
        elif choice == '4':
            add_research_report()
        elif choice == '5':
            add_custom_data()
        elif choice == '6':
            list_data()
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
