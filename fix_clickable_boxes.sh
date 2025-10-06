#!/bin/bash
echo "🔧 FIXING CLICKABLE BOXES - Direct JavaScript Injection"
cd /opt/mediamap

echo "1. Checking if JavaScript was added to template..."
grep -n "showPatients" backend/templates/healthpin/dashboard.html | head -2

echo ""
echo "2. Adding JavaScript directly to the template..."
python3 << 'EOF'
# Read the HealthPIN template
template_file = 'backend/templates/healthpin/dashboard.html'
try:
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Check if JavaScript already exists
    if 'showPatients' in template_content:
        print("✅ JavaScript already exists in template")
    else:
        print("❌ JavaScript not found, adding it now...")
        
        # Add JavaScript before closing body tag
        interactive_js = '''
<script>
console.log("HealthPIN Interactive Script Loading...");

// Interactive functions
function showPatients() {
    console.log("Patients clicked!");
    fetch('/healthpin/api/patients')
        .then(response => response.json())
        .then(data => {
            console.log("Patients data:", data);
            if (data.success) {
                showDataModal('Clinical Care Patients (' + data.total + ' total)', data.data, 'patients');
            }
        })
        .catch(error => console.error('Error:', error));
}

function showDoctors() {
    console.log("Doctors clicked!");
    fetch('/healthpin/api/doctors')
        .then(response => response.json())
        .then(data => {
            console.log("Doctors data:", data);
            if (data.success) {
                showDataModal('Healthcare Sources (' + data.total + ' sources)', data.data, 'doctors');
            }
        })
        .catch(error => console.error('Error:', error));
}

function showRecords() {
    console.log("Records clicked!");
    fetch('/healthpin/api/records')
        .then(response => response.json())
        .then(data => {
            console.log("Records data:", data);
            if (data.success) {
                showDataModal('Health Records (' + data.total + ' records)', data.data, 'records');
            }
        })
        .catch(error => console.error('Error:', error));
}

function showMatches() {
    console.log("Matches clicked!");
    fetch('/healthpin/api/matches')
        .then(response => response.json())
        .then(data => {
            console.log("Matches data:", data);
            if (data.success) {
                showDataModal('AI Matches (' + data.total + ' categories)', data.data, 'matches');
            }
        })
        .catch(error => console.error('Error:', error));
}

function showDataModal(title, data, type) {
    console.log("Showing modal:", title, type);
    
    let tableHeaders = '';
    if (type === 'patients') {
        tableHeaders = '<th>Clinical Case</th><th>Description</th><th>Source</th><th>Date</th>';
    } else if (type === 'doctors') {
        tableHeaders = '<th>Expert</th><th>Specialty</th><th>Entries</th><th>Latest</th>';
    } else if (type === 'records') {
        tableHeaders = '<th>Record</th><th>Category</th><th>Content</th><th>Date</th>';
    } else if (type === 'matches') {
        tableHeaders = '<th>Category</th><th>Matches</th><th>Description</th><th>Confidence</th>';
    }
    
    let tableRows = '';
    data.forEach(item => {
        tableRows += '<tr>';
        if (type === 'patients') {
            tableRows += `<td><strong>${item.name}</strong></td><td>${item.description.substring(0, 100)}...</td><td>${item.source.includes('who.int') ? 'WHO' : 'Medical Source'}</td><td>${item.date}</td>`;
        } else if (type === 'doctors') {
            tableRows += `<td><strong>${item.name}</strong></td><td>${item.specialty}</td><td>${item.entries_count || 'N/A'}</td><td>${(item.latest_entry || '').substring(0, 50)}...</td>`;
        } else if (type === 'records') {
            tableRows += `<td><strong>${item.title}</strong></td><td>${item.type || item.category}</td><td>${item.content.substring(0, 100)}...</td><td>${item.date}</td>`;
        } else if (type === 'matches') {
            tableRows += `<td><strong>${item.category}</strong></td><td>${item.match_count}</td><td>${item.description}</td><td>${((item.confidence || 0.85) * 100).toFixed(1)}%</td>`;
        }
        tableRows += '</tr>';
    });
    
    const modalHTML = `
        <div class="modal fade" id="dataModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title">🏥 ${title}</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead class="table-dark">
                                    <tr>${tableHeaders}</tr>
                                </thead>
                                <tbody>${tableRows}</tbody>
                            </table>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>`;
    
    // Remove existing modal
    const existingModal = document.getElementById('dataModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add new modal
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('dataModal'));
    modal.show();
}

// Add click handlers when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("Adding click handlers to boxes...");
    
    // Find all cards and add click handlers
    const cards = document.querySelectorAll('.card');
    console.log("Found", cards.length, "cards");
    
    cards.forEach((card, index) => {
        const cardText = card.textContent || '';
        console.log("Card", index, "text:", cardText.substring(0, 50));
        
        if (cardText.includes('Total Patients') || cardText.includes('44')) {
            console.log("Adding patients click handler");
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s';
            card.title = 'Click to view clinical cases';
            card.addEventListener('click', function(e) {
                e.preventDefault();
                console.log("Patients card clicked!");
                showPatients();
            });
            card.addEventListener('mouseenter', () => card.style.transform = 'scale(1.05)');
            card.addEventListener('mouseleave', () => card.style.transform = 'scale(1)');
        } else if (cardText.includes('Verified Doctors') || cardText.includes('2')) {
            console.log("Adding doctors click handler");
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s';
            card.title = 'Click to view healthcare sources';
            card.addEventListener('click', function(e) {
                e.preventDefault();
                console.log("Doctors card clicked!");
                showDoctors();
            });
            card.addEventListener('mouseenter', () => card.style.transform = 'scale(1.05)');
            card.addEventListener('mouseleave', () => card.style.transform = 'scale(1)');
        } else if (cardText.includes('Health Records') || cardText.includes('121')) {
            console.log("Adding records click handler");
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s';
            card.title = 'Click to view health records';
            card.addEventListener('click', function(e) {
                e.preventDefault();
                console.log("Records card clicked!");
                showRecords();
            });
            card.addEventListener('mouseenter', () => card.style.transform = 'scale(1.05)');
            card.addEventListener('mouseleave', () => card.style.transform = 'scale(1)');
        } else if (cardText.includes('AI Matches') || cardText.includes('4')) {
            console.log("Adding matches click handler");
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s';
            card.title = 'Click to view AI matches';
            card.addEventListener('click', function(e) {
                e.preventDefault();
                console.log("Matches card clicked!");
                showMatches();
            });
            card.addEventListener('mouseenter', () => card.style.transform = 'scale(1.05)');
            card.addEventListener('mouseleave', () => card.style.transform = 'scale(1)');
        }
    });
    
    console.log("HealthPIN Interactive Setup Complete!");
});
</script>'''
        
        # Add before closing body tag
        if '</body>' in template_content:
            template_content = template_content.replace('</body>', interactive_js + '\n</body>')
        else:
            template_content += interactive_js
        
        # Write back
        with open(template_file, 'w') as f:
            f.write(template_content)
        
        print("✅ Added JavaScript to template")

except Exception as e:
    print(f"❌ Error: {e}")
EOF

echo ""
echo "3. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "4. Testing the page..."
curl -s http://localhost/healthpin/ | grep -c "showPatients"

echo ""
echo "🔧 CLICKABLE BOXES FIX COMPLETE!"
echo ""
echo "Now visit: http://35.177.61.112/healthpin/"
echo ""
echo "✅ The boxes should now be clickable!"
echo "✅ Open browser developer console (F12) to see debug messages"
echo "✅ You should see cursor change to pointer when hovering"
echo "✅ Boxes should grow slightly on hover"
echo ""
echo "If still not working, press F12 and check Console tab for errors"
