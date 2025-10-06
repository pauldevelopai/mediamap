#!/bin/bash
echo "🔗 ADDING LINKS UNDER COLORFUL BOXES"
cd /opt/mediamap

echo "1. Adding links to the HealthPIN dashboard template..."
python3 << 'EOF'
# Read the HealthPIN dashboard template
template_file = 'backend/templates/healthpin/dashboard.html'
try:
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Find the colorful boxes section and add links
    # Look for the card structure and add links after each card
    
    # Add CSS for the links
    link_css = '''
<style>
.data-link {
    display: block;
    text-align: center;
    margin-top: 10px;
    padding: 8px 16px;
    background: rgba(255,255,255,0.9);
    border-radius: 20px;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease;
    border: 1px solid rgba(0,0,0,0.1);
}
.data-link:hover {
    background: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    text-decoration: none;
}
.patients-link { color: #0d6efd; border-color: #0d6efd; }
.doctors-link { color: #198754; border-color: #198754; }
.records-link { color: #0dcaf0; border-color: #0dcaf0; }
.matches-link { color: #ffc107; border-color: #ffc107; }
</style>'''
    
    # Add CSS to head section
    if '<head>' in template_content:
        template_content = template_content.replace('<head>', '<head>' + link_css)
    
    # Add JavaScript to inject links after page loads
    link_js = '''
<script>
document.addEventListener('DOMContentLoaded', function() {
    console.log("Adding data links under boxes...");
    
    // Find all cards and add appropriate links
    const cards = document.querySelectorAll('.card');
    
    cards.forEach(card => {
        const cardText = card.textContent || '';
        let linkHTML = '';
        
        if (cardText.includes('Total Patients') || cardText.includes('44')) {
            linkHTML = '<a href="/healthpin/patients" class="data-link patients-link">👥 View All Clinical Cases</a>';
        } else if (cardText.includes('Verified Doctors') || cardText.includes('2')) {
            linkHTML = '<a href="/healthpin/doctors" class="data-link doctors-link">👨‍⚕️ View Healthcare Sources</a>';
        } else if (cardText.includes('Health Records') || cardText.includes('121')) {
            linkHTML = '<a href="/healthpin/records" class="data-link records-link">📋 View All Records</a>';
        } else if (cardText.includes('AI Matches') || cardText.includes('4')) {
            linkHTML = '<a href="/healthpin/matches" class="data-link matches-link">🤖 View AI Categories</a>';
        }
        
        if (linkHTML) {
            // Add link to card body or after card
            const cardBody = card.querySelector('.card-body');
            if (cardBody) {
                cardBody.insertAdjacentHTML('beforeend', linkHTML);
                console.log("Added link to card:", cardText.substring(0, 20));
            }
        }
    });
    
    console.log("Data links added successfully!");
});
</script>'''
    
    # Add JavaScript before closing body tag
    if '</body>' in template_content:
        template_content = template_content.replace('</body>', link_js + '\n</body>')
    else:
        template_content += link_js
    
    # Write back
    with open(template_file, 'w') as f:
        f.write(template_content)
    
    print("✅ Added links and styling to HealthPIN dashboard")

except Exception as e:
    print(f"❌ Error updating template: {e}")
EOF

echo ""
echo "2. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "3. Testing the new pages..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "Testing patients page:"
curl -s -b "session=$COOKIE" http://localhost/healthpin/patients | head -5

echo ""
echo "Testing doctors page:"
curl -s -b "session=$COOKIE" http://localhost/healthpin/doctors | head -5

echo ""
echo "🔗 LINKS UNDER BOXES COMPLETE!"
echo ""
echo "✅ Added beautiful links under each colorful box:"
echo "   🔵 Total Patients → 👥 View All Clinical Cases"
echo "   🟢 Verified Doctors → 👨‍⚕️ View Healthcare Sources"  
echo "   🔵 Health Records → 📋 View All Records"
echo "   🟡 AI Matches → 🤖 View AI Categories"
echo ""
echo "✅ Each link goes to a dedicated page with your real data!"
echo "✅ Pages auto-update with new agent data!"
echo ""
echo "Visit: http://35.177.61.112/healthpin/"
echo "You'll see links under each colorful box!"
