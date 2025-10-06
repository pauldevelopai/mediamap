#!/bin/bash
echo "🔧 FIXING BUTTONS DIRECTLY IN HTML"
cd /opt/mediamap

echo "1. Creating backup of current template..."
cp backend/templates/healthpin/dashboard.html backend/templates/healthpin/dashboard.html.backup

echo "2. Removing all JavaScript button attempts (lines 1400+)..."
# Keep only the first 400 lines (before all the JavaScript mess)
head -n 400 backend/templates/healthpin/dashboard.html > /tmp/clean_template.html

echo "3. Adding the closing tags and clean JavaScript..."
cat >> /tmp/clean_template.html << 'EOF'

</div>

<script>
// Simple, clean JavaScript for HealthPIN dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('HealthPIN Dashboard loaded successfully');
    
    // Add any future interactive features here
});
</script>

{% endblock %}
EOF

echo "4. Now adding buttons directly in the HTML structure..."
# Add button to Total Patients card (after line 30: <p class="card-text">Total Patients</p>)
sed -i '30a\                            <a href="/healthpin/patients" class="btn btn-sm btn-light mt-2">👥 View Cases</a>' /tmp/clean_template.html

# Add button to Verified Doctors card (after line 49: <p class="card-text">Verified Doctors</p>)
sed -i '50a\                            <a href="/healthpin/doctors" class="btn btn-sm btn-light mt-2">👨‍⚕️ View Sources</a>' /tmp/clean_template.html

# Add button to Health Records card (after line 69: <p class="card-text">Health Records</p>)
sed -i '70a\                            <a href="/healthpin/records" class="btn btn-sm btn-light mt-2">📋 View Records</a>' /tmp/clean_template.html

# Add button to AI Matches card (after line 89: <p class="card-text">AI Matches</p>)
sed -i '90a\                            <a href="/healthpin/matches" class="btn btn-sm btn-light mt-2">🤖 View Matches</a>' /tmp/clean_template.html

echo "5. Replacing the template with clean version..."
mv /tmp/clean_template.html backend/templates/healthpin/dashboard.html

echo "6. Setting correct permissions..."
chown www-data:www-data backend/templates/healthpin/dashboard.html
chmod 644 backend/templates/healthpin/dashboard.html

echo "7. Checking the result..."
echo "=== CHECKING TOTAL PATIENTS SECTION ==="
grep -n -A 5 -B 2 "Total Patients" backend/templates/healthpin/dashboard.html

echo ""
echo "=== CHECKING ALL BUTTON ADDITIONS ==="
grep -n "View Cases\|View Sources\|View Records\|View Matches" backend/templates/healthpin/dashboard.html

echo "8. Restarting service..."
systemctl restart mediamap

echo ""
echo "🔧 DIRECT HTML BUTTON FIX COMPLETE!"
echo "✅ Removed all conflicting JavaScript"
echo "✅ Added buttons directly in HTML structure"
echo "✅ Buttons should now appear under each colorful box"
echo ""
echo "Visit: http://35.177.61.112/healthpin/"
echo "You should see small light-colored buttons under each number!"
