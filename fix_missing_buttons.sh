#!/bin/bash
echo "🔧 FIXING MISSING BUTTONS - Direct Template Edit"
cd /opt/mediamap

echo "1. Checking current dashboard template..."
grep -n "View" backend/templates/healthpin/dashboard.html | head -3

echo ""
echo "2. Adding buttons directly to the template HTML..."
python3 << 'EOF'
# Read the dashboard template
with open('backend/templates/healthpin/dashboard.html', 'r') as f:
    content = f.read()

# Find the colorful boxes section and add buttons after each one
# Look for the card structure with "Total Patients"
if 'Total Patients' in content and 'View Clinical Cases' not in content:
    # Add button after Total Patients card
    patients_card = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">44</h4>
                                <p class="card-text">Total Patients</p>
                            </div>'''
    
    patients_with_button = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">44</h4>
                                <p class="card-text">Total Patients</p>
                                <a href="/healthpin/patients" class="btn btn-sm btn-light mt-2">👥 View Clinical Cases</a>
                            </div>'''
    
    content = content.replace(patients_card, patients_with_button)
    
    # Add button after Verified Doctors card
    doctors_card = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">2</h4>
                                <p class="card-text">Verified Doctors</p>
                            </div>'''
    
    doctors_with_button = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">2</h4>
                                <p class="card-text">Verified Doctors</p>
                                <a href="/healthpin/doctors" class="btn btn-sm btn-light mt-2">👨‍⚕️ View Sources</a>
                            </div>'''
    
    content = content.replace(doctors_card, doctors_with_button)
    
    # Add button after Health Records card
    records_card = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">121</h4>
                                <p class="card-text">Health Records</p>
                            </div>'''
    
    records_with_button = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">121</h4>
                                <p class="card-text">Health Records</p>
                                <a href="/healthpin/records" class="btn btn-sm btn-light mt-2">📋 View Records</a>
                            </div>'''
    
    content = content.replace(records_card, records_with_button)
    
    # Add button after AI Matches card
    matches_card = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">4</h4>
                                <p class="card-text">AI Matches</p>
                            </div>'''
    
    matches_with_button = '''                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">4</h4>
                                <p class="card-text">AI Matches</p>
                                <a href="/healthpin/matches" class="btn btn-sm btn-light mt-2">🤖 View Matches</a>
                            </div>'''
    
    content = content.replace(matches_card, matches_with_button)
    
    # Write back
    with open('backend/templates/healthpin/dashboard.html', 'w') as f:
        f.write(content)
    
    print("✅ Added buttons directly to template HTML")
else:
    print("❌ Could not find card structure or buttons already exist")
EOF

echo ""
echo "3. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "4. Testing if buttons appear..."
curl -s http://localhost/healthpin/ | grep -c "View Clinical Cases"

echo ""
echo "🔧 BUTTON FIX COMPLETE!"
echo ""
echo "Visit: http://35.177.61.112/healthpin/"
echo "You should now see small buttons under each colorful box!"
echo ""
echo "If still not working, the template structure might be different."
echo "Let me know and I'll create a different approach."
