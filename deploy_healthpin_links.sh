#!/bin/bash
echo "🔗 DEPLOYING HEALTHPIN LINKS AND PAGES"
cd /opt/mediamap

echo "1. Adding dedicated page routes to HealthPIN..."
python3 << 'EOF'
# Read the current routes file
with open('backend/healthpin/routes.py', 'r') as f:
    content = f.read()

# Check if routes already exist
if '/patients' in content:
    print("✅ Routes already exist")
else:
    # Add new page routes
    page_routes = '''

# Dedicated data pages
@healthpin_bp.route('/patients')
@login_required
def patients_page():
    """Clinical patients page"""
    import json, os
    try:
        data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                agent_data = json.load(f)
            clinical_entries = [entry for entry in agent_data if entry.get('category') == 'Clinical_Care']
            patients_data = []
            for i, entry in enumerate(clinical_entries):
                patients_data.append({
                    'id': i + 1,
                    'title': f"Clinical Case {i + 1}",
                    'content': entry.get('content', ''),
                    'source': entry.get('source', 'Healthcare Source'),
                    'date': entry.get('timestamp', '2025-10-06')[:10]
                })
            return render_template('healthpin/patients.html', patients=patients_data, total_count=len(clinical_entries))
    except: pass
    return render_template('healthpin/patients.html', patients=[], total_count=0)

@healthpin_bp.route('/doctors')
@login_required
def doctors_page():
    """Healthcare sources page"""
    import json, os
    try:
        data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                agent_data = json.load(f)
            sources = {}
            for entry in agent_data:
                source = entry.get('source', 'Unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(entry)
            doctors_data = []
            for i, (source, entries) in enumerate(sources.items()):
                if 'who.int' in source.lower():
                    name = "WHO Global Health Expert"
                    specialty = "Global Health Policy"
                else:
                    name = f"Healthcare Source {i + 1}"
                    specialty = "Healthcare Intelligence"
                doctors_data.append({
                    'id': i + 1,
                    'name': name,
                    'specialty': specialty,
                    'source_url': source,
                    'entries_count': len(entries)
                })
            return render_template('healthpin/doctors.html', doctors=doctors_data, total_count=len(sources))
    except: pass
    return render_template('healthpin/doctors.html', doctors=[], total_count=0)

@healthpin_bp.route('/records')
@login_required
def records_page():
    """All health records page"""
    import json, os
    try:
        data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                agent_data = json.load(f)
            records_data = []
            for i, entry in enumerate(agent_data):
                records_data.append({
                    'id': i + 1,
                    'title': f"Health Record {i + 1}",
                    'category': entry.get('category', 'Healthcare').replace('_', ' '),
                    'content': entry.get('content', ''),
                    'source': entry.get('source', 'Healthcare Source'),
                    'date': entry.get('timestamp', '2025-10-06')[:10]
                })
            return render_template('healthpin/records.html', records=records_data, total_count=len(agent_data))
    except: pass
    return render_template('healthpin/records.html', records=[], total_count=0)

@healthpin_bp.route('/matches')
@login_required
def matches_page():
    """AI matches page"""
    import json, os
    try:
        data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                agent_data = json.load(f)
            categories = {}
            for entry in agent_data:
                cat = entry.get('category', 'Unknown')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(entry)
            matches_data = []
            for i, (category, entries) in enumerate(categories.items()):
                matches_data.append({
                    'id': i + 1,
                    'category': category.replace('_', ' '),
                    'match_count': len(entries),
                    'description': f"AI categorized {len(entries)} healthcare entries"
                })
            return render_template('healthpin/matches.html', matches=matches_data, total_categories=len(categories))
    except: pass
    return render_template('healthpin/matches.html', matches=[], total_categories=0)
'''
    
    content += page_routes
    with open('backend/healthpin/routes.py', 'w') as f:
        f.write(content)
    print("✅ Added page routes")
EOF

echo ""
echo "2. Creating simple HTML templates..."
mkdir -p backend/templates/healthpin

# Simple patients template
cat > backend/templates/healthpin/patients.html << 'EOF'
{% extends "admin/base_admin.html" %}
{% block title %}Clinical Patients{% endblock %}
{% block content %}
<div class="container-fluid">
    <h1>🏥 Clinical Care Patients ({{ total_count }})</h1>
    <a href="/healthpin/" class="btn btn-secondary mb-3">← Back to Dashboard</a>
    
    {% if patients %}
    <div class="row">
        {% for patient in patients %}
        <div class="col-md-6 mb-3">
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h6>{{ patient.title }}</h6>
                </div>
                <div class="card-body">
                    <p>{{ patient.content[:200] }}...</p>
                    <small class="text-muted">Source: {{ patient.source|replace('https://www.who.int/rss-feeds/news-english.xml', 'WHO') }} | {{ patient.date }}</small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>No clinical data available.</p>
    {% endif %}
</div>
{% endblock %}
EOF

# Simple doctors template
cat > backend/templates/healthpin/doctors.html << 'EOF'
{% extends "admin/base_admin.html" %}
{% block title %}Healthcare Sources{% endblock %}
{% block content %}
<div class="container-fluid">
    <h1>👨‍⚕️ Healthcare Sources ({{ total_count }})</h1>
    <a href="/healthpin/" class="btn btn-secondary mb-3">← Back to Dashboard</a>
    
    {% if doctors %}
    <div class="row">
        {% for doctor in doctors %}
        <div class="col-md-6 mb-3">
            <div class="card">
                <div class="card-header bg-success text-white">
                    <h5>{{ doctor.name }}</h5>
                </div>
                <div class="card-body">
                    <p><strong>Specialty:</strong> {{ doctor.specialty }}</p>
                    <p><strong>Data Points:</strong> {{ doctor.entries_count }}</p>
                    <small class="text-muted">{{ doctor.source_url|replace('https://', '') }}</small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>No healthcare sources available.</p>
    {% endif %}
</div>
{% endblock %}
EOF

# Simple records template
cat > backend/templates/healthpin/records.html << 'EOF'
{% extends "admin/base_admin.html" %}
{% block title %}Health Records{% endblock %}
{% block content %}
<div class="container-fluid">
    <h1>📋 Health Records ({{ total_count }})</h1>
    <a href="/healthpin/" class="btn btn-secondary mb-3">← Back to Dashboard</a>
    
    {% if records %}
    <div class="table-responsive">
        <table class="table table-striped">
            <thead>
                <tr><th>Record</th><th>Category</th><th>Content</th><th>Date</th></tr>
            </thead>
            <tbody>
                {% for record in records %}
                <tr>
                    <td>{{ record.title }}</td>
                    <td><span class="badge bg-info">{{ record.category }}</span></td>
                    <td>{{ record.content[:100] }}...</td>
                    <td>{{ record.date }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% else %}
    <p>No health records available.</p>
    {% endif %}
</div>
{% endblock %}
EOF

# Simple matches template
cat > backend/templates/healthpin/matches.html << 'EOF'
{% extends "admin/base_admin.html" %}
{% block title %}AI Matches{% endblock %}
{% block content %}
<div class="container-fluid">
    <h1>🤖 AI Healthcare Matches ({{ total_categories }})</h1>
    <a href="/healthpin/" class="btn btn-secondary mb-3">← Back to Dashboard</a>
    
    {% if matches %}
    <div class="row">
        {% for match in matches %}
        <div class="col-md-6 mb-3">
            <div class="card">
                <div class="card-header bg-warning">
                    <h5>{{ match.category }}</h5>
                </div>
                <div class="card-body">
                    <p><strong>Matches:</strong> {{ match.match_count }}</p>
                    <p>{{ match.description }}</p>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>No AI matches available.</p>
    {% endif %}
</div>
{% endblock %}
EOF

echo "✅ Created HTML templates"

echo ""
echo "3. Adding links under boxes in dashboard template..."
python3 << 'EOF'
template_file = 'backend/templates/healthpin/dashboard.html'
try:
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Add JavaScript to inject links
    js_code = '''
<script>
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        const text = card.textContent;
        let link = '';
        
        if (text.includes('Total Patients') || text.includes('44')) {
            link = '<div class="mt-2"><a href="/healthpin/patients" class="btn btn-sm btn-outline-primary">👥 View Clinical Cases</a></div>';
        } else if (text.includes('Verified Doctors') || text.includes('2')) {
            link = '<div class="mt-2"><a href="/healthpin/doctors" class="btn btn-sm btn-outline-success">👨‍⚕️ View Sources</a></div>';
        } else if (text.includes('Health Records') || text.includes('121')) {
            link = '<div class="mt-2"><a href="/healthpin/records" class="btn btn-sm btn-outline-info">📋 View Records</a></div>';
        } else if (text.includes('AI Matches') || text.includes('4')) {
            link = '<div class="mt-2"><a href="/healthpin/matches" class="btn btn-sm btn-outline-warning">🤖 View Matches</a></div>';
        }
        
        if (link) {
            const cardBody = card.querySelector('.card-body');
            if (cardBody) {
                cardBody.insertAdjacentHTML('beforeend', link);
            }
        }
    });
});
</script>'''
    
    if '</body>' in content:
        content = content.replace('</body>', js_code + '\n</body>')
    else:
        content += js_code
    
    with open(template_file, 'w') as f:
        f.write(content)
    
    print("✅ Added links to dashboard")
except Exception as e:
    print(f"Error: {e}")
EOF

echo ""
echo "4. Testing syntax..."
python3 -m py_compile backend/healthpin/routes.py

echo ""
echo "5. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "🔗 HEALTHPIN LINKS DEPLOYMENT COMPLETE!"
echo ""
echo "✅ Added 4 new pages:"
echo "   /healthpin/patients - Clinical cases"
echo "   /healthpin/doctors - Healthcare sources"  
echo "   /healthpin/records - All health records"
echo "   /healthpin/matches - AI matches"
echo ""
echo "✅ Added links under each colorful box!"
echo ""
echo "Visit: http://35.177.61.112/healthpin/"
echo "You should now see buttons under each box!"
