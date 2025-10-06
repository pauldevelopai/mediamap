#!/bin/bash
echo "🎨 FIXING HEALTHPIN PAGE STYLING TO MATCH APP DESIGN"
cd /opt/mediamap

echo "1. Creating properly styled patients page..."
cat > backend/templates/healthpin/patients.html << 'EOF'
{% extends "admin/base_admin.html" %}

{% block title %}Clinical Cases - HealthPIN{% endblock %}

{% block extra_css %}
<style>
    .patient-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #dc3545;
        transition: all 0.3s ease;
    }
    
    .patient-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .patient-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .badge-clinical {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-active {
        background: #d4edda;
        color: #155724;
    }
    
    .patient-meta {
        font-size: 0.85em;
        color: #6c757d;
        margin-top: 10px;
    }
    
    .page-header {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Page Header -->
    <div class="page-header">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h1 class="mb-2">
                    <i class="bi bi-people-fill me-3"></i>
                    Clinical Cases
                </h1>
                <p class="mb-0 opacity-75">{{ total_count }} patient records in the system</p>
            </div>
            <div>
                <a href="/healthpin/" class="btn btn-light">
                    <i class="bi bi-arrow-left me-2"></i>Back to Dashboard
                </a>
            </div>
        </div>
    </div>

    {% if patients %}
    <!-- Statistics Row -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card text-center">
                <div class="card-body">
                    <h3 class="text-primary">{{ total_count }}</h3>
                    <p class="text-muted mb-0">Total Cases</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center">
                <div class="card-body">
                    <h3 class="text-success">{{ patients|selectattr('is_active', 'equalto', true)|list|length or total_count }}</h3>
                    <p class="text-muted mb-0">Active Cases</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center">
                <div class="card-body">
                    <h3 class="text-info">{{ patients|map(attribute='city')|unique|list|length or 'Multiple' }}</h3>
                    <p class="text-muted mb-0">Locations</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center">
                <div class="card-body">
                    <h3 class="text-warning">Recent</h3>
                    <p class="text-muted mb-0">Last Updated</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Patient Cards -->
    <div class="row">
        {% for patient in patients %}
        <div class="col-lg-6 mb-4">
            <div class="patient-card">
                <div class="d-flex justify-content-between align-items-start mb-3">
                    <h5 class="card-title mb-0">
                        <i class="bi bi-person-circle me-2 text-primary"></i>
                        {{ patient.first_name }} {{ patient.last_name }}
                    </h5>
                    <span class="patient-badge badge-active">Active</span>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <p class="mb-2">
                            <i class="bi bi-telephone me-2 text-muted"></i>
                            <strong>Phone:</strong> {{ patient.phone_number or 'Not provided' }}
                        </p>
                        <p class="mb-2">
                            <i class="bi bi-geo-alt me-2 text-muted"></i>
                            <strong>Location:</strong> {{ patient.city or 'Unknown' }}, {{ patient.province or 'Unknown' }}
                        </p>
                    </div>
                    <div class="col-md-6">
                        <p class="mb-2">
                            <i class="bi bi-calendar me-2 text-muted"></i>
                            <strong>DOB:</strong> {{ patient.date_of_birth.strftime('%Y-%m-%d') if patient.date_of_birth else 'Not provided' }}
                        </p>
                        <p class="mb-2">
                            <i class="bi bi-translate me-2 text-muted"></i>
                            <strong>Language:</strong> {{ patient.language_preference or 'English' }}
                        </p>
                    </div>
                </div>
                
                {% if patient.preferred_specialties %}
                <div class="mt-3">
                    <strong>Preferred Specialties:</strong>
                    <div class="mt-1">
                        {% for specialty in patient.preferred_specialties %}
                        <span class="badge bg-primary me-1">{{ specialty }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <div class="patient-meta">
                    <i class="bi bi-clock me-1"></i>
                    Registered: {{ patient.created_at.strftime('%Y-%m-%d %H:%M') if patient.created_at else 'Unknown' }}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <!-- Empty State -->
    <div class="row">
        <div class="col-12">
            <div class="card text-center py-5">
                <div class="card-body">
                    <i class="bi bi-people display-1 text-muted mb-4"></i>
                    <h3 class="text-muted">No Patient Records Found</h3>
                    <p class="text-muted mb-4">There are currently no patients registered in the HealthPIN system.</p>
                    <button class="btn btn-primary" onclick="showAddPatientModal()">
                        <i class="bi bi-person-plus me-2"></i>Add First Patient
                    </button>
                </div>
            </div>
        </div>
    </div>
    {% endif %}
</div>
{% endblock %}
EOF

echo "2. Creating properly styled doctors page..."
cat > backend/templates/healthpin/doctors.html << 'EOF'
{% extends "admin/base_admin.html" %}

{% block title %}South African Doctors - HealthPIN{% endblock %}

{% block extra_css %}
<style>
    .doctor-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #198754;
        transition: all 0.3s ease;
    }
    
    .doctor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .doctor-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .badge-verified {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-pending {
        background: #fff3cd;
        color: #856404;
    }
    
    .specialty-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px;
    }
    
    .page-header {
        background: linear-gradient(135deg, #198754, #157347);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .scrape-btn {
        background: linear-gradient(45deg, #198754, #20c997);
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .scrape-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(25, 135, 84, 0.3);
        color: white;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Page Header -->
    <div class="page-header">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h1 class="mb-2">
                    <i class="bi bi-stethoscope me-3"></i>
                    South African Doctors
                </h1>
                <p class="mb-0 opacity-75">{{ total_count }} verified healthcare professionals</p>
            </div>
            <div>
                <button class="btn scrape-btn me-2" onclick="triggerDoctorScraping()">
                    <i class="bi bi-arrow-clockwise me-2"></i>Scrape More Doctors
                </button>
                <a href="/healthpin/" class="btn btn-light">
                    <i class="bi bi-arrow-left me-2"></i>Back to Dashboard
                </a>
            </div>
        </div>
    </div>

    {% if doctors %}
    <!-- Statistics Row -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card text-center border-success">
                <div class="card-body">
                    <h3 class="text-success">{{ total_count }}</h3>
                    <p class="text-muted mb-0">Total Doctors</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-primary">
                <div class="card-body">
                    <h3 class="text-primary">{{ doctors|selectattr('is_verified', 'equalto', true)|list|length }}</h3>
                    <p class="text-muted mb-0">Verified</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-info">
                <div class="card-body">
                    <h3 class="text-info">{{ doctors|map(attribute='city')|unique|list|length }}</h3>
                    <p class="text-muted mb-0">Cities</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-warning">
                <div class="card-body">
                    <h3 class="text-warning">{{ doctors|map(attribute='province')|unique|list|length }}</h3>
                    <p class="text-muted mb-0">Provinces</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Doctor Cards -->
    <div class="row">
        {% for doctor in doctors %}
        <div class="col-lg-6 mb-4">
            <div class="doctor-card">
                <div class="d-flex justify-content-between align-items-start mb-3">
                    <h5 class="card-title mb-0">
                        <i class="bi bi-person-badge me-2 text-success"></i>
                        {{ doctor.full_name }}
                    </h5>
                    {% if doctor.is_verified %}
                        <span class="doctor-badge badge-verified">
                            <i class="bi bi-check-circle me-1"></i>Verified
                        </span>
                    {% else %}
                        <span class="doctor-badge badge-pending">
                            <i class="bi bi-clock me-1"></i>Pending
                        </span>
                    {% endif %}
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <p class="mb-2">
                            <i class="bi bi-building me-2 text-muted"></i>
                            <strong>Practice:</strong> {{ doctor.practice_name or 'Private Practice' }}
                        </p>
                        <p class="mb-2">
                            <i class="bi bi-geo-alt me-2 text-muted"></i>
                            <strong>Location:</strong> {{ doctor.city }}, {{ doctor.province }}
                        </p>
                        {% if doctor.phone %}
                        <p class="mb-2">
                            <i class="bi bi-telephone me-2 text-muted"></i>
                            <strong>Phone:</strong> {{ doctor.phone }}
                        </p>
                        {% endif %}
                    </div>
                    <div class="col-md-6">
                        {% if doctor.consultation_fee %}
                        <p class="mb-2">
                            <i class="bi bi-currency-dollar me-2 text-muted"></i>
                            <strong>Fee:</strong> R{{ doctor.consultation_fee }}
                        </p>
                        {% endif %}
                        {% if doctor.years_experience %}
                        <p class="mb-2">
                            <i class="bi bi-award me-2 text-muted"></i>
                            <strong>Experience:</strong> {{ doctor.years_experience }} years
                        </p>
                        {% endif %}
                        <p class="mb-2">
                            <i class="bi bi-shield-check me-2 text-muted"></i>
                            <strong>Medical Aid:</strong> {{ 'Yes' if doctor.accepts_medical_aid else 'No' }}
                        </p>
                    </div>
                </div>
                
                {% if doctor.specialties %}
                <div class="mt-3">
                    <strong class="text-muted">
                        <i class="bi bi-heart-pulse me-1"></i>Specialties:
                    </strong>
                    <div class="mt-2">
                        {% for specialty in doctor.specialties %}
                        <span class="specialty-badge">{{ specialty }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <div class="mt-3 pt-3 border-top">
                    <small class="text-muted">
                        <i class="bi bi-card-text me-1"></i>
                        License: {{ doctor.medical_license }}
                        <span class="ms-3">
                            <i class="bi bi-clock me-1"></i>
                            Added: {{ doctor.created_at[:10] if doctor.created_at else 'Unknown' }}
                        </span>
                    </small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <!-- Empty State -->
    <div class="row">
        <div class="col-12">
            <div class="card text-center py-5">
                <div class="card-body">
                    <i class="bi bi-stethoscope display-1 text-muted mb-4"></i>
                    <h3 class="text-muted">No South African Doctors Found</h3>
                    <p class="text-muted mb-4">The doctor database is empty. Use OpenStreetMap data to find real healthcare professionals in South Africa.</p>
                    <button class="btn scrape-btn" onclick="triggerDoctorScraping()">
                        <i class="bi bi-arrow-clockwise me-2"></i>Start Doctor Scraping
                    </button>
                </div>
            </div>
        </div>
    </div>
    {% endif %}
</div>

<script>
function triggerDoctorScraping() {
    const btn = event.target;
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="bi bi-arrow-clockwise me-2"></i>Scraping...';
    
    fetch('/healthpin/scrape-doctors', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({limit: 100})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            const alert = document.createElement('div');
            alert.className = 'alert alert-success alert-dismissible fade show';
            alert.innerHTML = `
                <i class="bi bi-check-circle me-2"></i>
                <strong>Success!</strong> Doctor scraping completed. Found ${data.result?.doctors_added || 'new'} doctors.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.querySelector('.container-fluid').insertBefore(alert, document.querySelector('.page-header').nextSibling);
            
            // Refresh page after 2 seconds
            setTimeout(() => location.reload(), 2000);
        } else {
            // Show error message
            const alert = document.createElement('div');
            alert.className = 'alert alert-danger alert-dismissible fade show';
            alert.innerHTML = `
                <i class="bi bi-exclamation-triangle me-2"></i>
                <strong>Error!</strong> ${data.error || 'Scraping failed'}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.querySelector('.container-fluid').insertBefore(alert, document.querySelector('.page-header').nextSibling);
            
            btn.disabled = false;
            btn.innerHTML = originalHTML;
        }
    })
    .catch(error => {
        // Show error message
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show';
        alert.innerHTML = `
            <i class="bi bi-exclamation-triangle me-2"></i>
            <strong>Error!</strong> Network error: ${error.message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.querySelector('.container-fluid').insertBefore(alert, document.querySelector('.page-header').nextSibling);
        
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    });
}
</script>
{% endblock %}
EOF

echo "3. Creating properly styled records page..."
cat > backend/templates/healthpin/records.html << 'EOF'
{% extends "admin/base_admin.html" %}

{% block title %}Health Records - HealthPIN{% endblock %}

{% block extra_css %}
<style>
    .record-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #0dcaf0;
        transition: all 0.3s ease;
    }
    
    .record-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .badge-clinical {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-research {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-policy {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-unknown {
        background: #e2e3e5;
        color: #495057;
    }
    
    .page-header {
        background: linear-gradient(135deg, #0dcaf0, #0aa2c0);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .record-content {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-size: 0.9em;
        line-height: 1.5;
    }
    
    .source-link {
        color: #0dcaf0;
        text-decoration: none;
        font-size: 0.85em;
    }
    
    .source-link:hover {
        color: #0aa2c0;
        text-decoration: underline;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Page Header -->
    <div class="page-header">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h1 class="mb-2">
                    <i class="bi bi-file-medical me-3"></i>
                    Health Records
                </h1>
                <p class="mb-0 opacity-75">{{ total_count }} records collected from various healthcare sources</p>
            </div>
            <div>
                <a href="/healthpin/" class="btn btn-light">
                    <i class="bi bi-arrow-left me-2"></i>Back to Dashboard
                </a>
            </div>
        </div>
    </div>

    {% if records %}
    <!-- Statistics Row -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card text-center border-info">
                <div class="card-body">
                    <h3 class="text-info">{{ total_count }}</h3>
                    <p class="text-muted mb-0">Total Records</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-danger">
                <div class="card-body">
                    <h3 class="text-danger">{{ records|selectattr('category', 'equalto', 'Clinical_Care')|list|length }}</h3>
                    <p class="text-muted mb-0">Clinical Care</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-success">
                <div class="card-body">
                    <h3 class="text-success">{{ records|selectattr('category', 'equalto', 'Medical_Research')|list|length }}</h3>
                    <p class="text-muted mb-0">Research</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-warning">
                <div class="card-body">
                    <h3 class="text-warning">{{ records|selectattr('category', 'equalto', 'Healthcare_Policy')|list|length }}</h3>
                    <p class="text-muted mb-0">Policy</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Record Cards -->
    <div class="row">
        {% for record in records %}
        <div class="col-lg-6 mb-4">
            <div class="record-card">
                <div class="d-flex justify-content-between align-items-start mb-3">
                    <h6 class="card-title mb-0">
                        <i class="bi bi-file-text me-2 text-info"></i>
                        Record #{{ record.id }}
                    </h6>
                    {% if record.category == 'Clinical_Care' %}
                        <span class="category-badge badge-clinical">Clinical Care</span>
                    {% elif record.category == 'Medical_Research' %}
                        <span class="category-badge badge-research">Research</span>
                    {% elif record.category == 'Healthcare_Policy' %}
                        <span class="category-badge badge-policy">Policy</span>
                    {% else %}
                        <span class="category-badge badge-unknown">{{ record.category }}</span>
                    {% endif %}
                </div>
                
                <div class="record-content">
                    {{ record.content[:300] }}{% if record.content|length > 300 %}...{% endif %}
                </div>
                
                <div class="mt-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <small class="text-muted">
                                <i class="bi bi-clock me-1"></i>
                                {{ record.created_at.strftime('%Y-%m-%d %H:%M') if record.created_at else 'Unknown date' }}
                            </small>
                        </div>
                        <div>
                            {% if 'who.int' in record.source %}
                                <span class="badge bg-primary">WHO</span>
                            {% elif 'harvard' in record.source %}
                                <span class="badge bg-success">Harvard Health</span>
                            {% elif 'ChatGPT' in record.source %}
                                <span class="badge bg-warning">AI Analysis</span>
                            {% else %}
                                <span class="badge bg-secondary">External Source</span>
                            {% endif %}
                        </div>
                    </div>
                    
                    {% if record.source and record.source.startswith('http') %}
                    <div class="mt-2">
                        <a href="{{ record.source }}" target="_blank" class="source-link">
                            <i class="bi bi-link-45deg me-1"></i>
                            View Original Source
                        </a>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <!-- Empty State -->
    <div class="row">
        <div class="col-12">
            <div class="card text-center py-5">
                <div class="card-body">
                    <i class="bi bi-file-medical display-1 text-muted mb-4"></i>
                    <h3 class="text-muted">No Health Records Found</h3>
                    <p class="text-muted mb-4">No health records have been collected yet. The AI agents will automatically gather healthcare data from various sources.</p>
                    <a href="/admin/agents" class="btn btn-primary">
                        <i class="bi bi-robot me-2"></i>Manage AI Agents
                    </a>
                </div>
            </div>
        </div>
    </div>
    {% endif %}
</div>
{% endblock %}
EOF

echo "4. Creating properly styled matches page..."
cat > backend/templates/healthpin/matches.html << 'EOF'
{% extends "admin/base_admin.html" %}

{% block title %}AI Matches - HealthPIN{% endblock %}

{% block extra_css %}
<style>
    .match-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #ffc107;
        transition: all 0.3s ease;
    }
    
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .confidence-high {
        background: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    .page-header {
        background: linear-gradient(135deg, #ffc107, #ffca2c);
        color: #212529;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .category-icon {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
        margin-right: 15px;
    }
    
    .category-clinical {
        background: #f8d7da;
        color: #721c24;
    }
    
    .category-research {
        background: #d4edda;
        color: #155724;
    }
    
    .category-policy {
        background: #fff3cd;
        color: #856404;
    }
    
    .category-unknown {
        background: #e2e3e5;
        color: #495057;
    }
    
    .progress-ring {
        width: 60px;
        height: 60px;
    }
    
    .progress-ring circle {
        fill: transparent;
        stroke: #e9ecef;
        stroke-width: 4;
        stroke-dasharray: 188.5;
        stroke-dashoffset: 188.5;
        transition: stroke-dashoffset 0.5s ease;
    }
    
    .progress-ring .progress {
        stroke: #ffc107;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Page Header -->
    <div class="page-header">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h1 class="mb-2">
                    <i class="bi bi-cpu me-3"></i>
                    AI Category Matches
                </h1>
                <p class="mb-0 opacity-75">{{ total_count }} intelligent categorizations by AI analysis</p>
            </div>
            <div>
                <a href="/healthpin/" class="btn btn-dark">
                    <i class="bi bi-arrow-left me-2"></i>Back to Dashboard
                </a>
            </div>
        </div>
    </div>

    {% if matches %}
    <!-- Statistics Row -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card text-center border-warning">
                <div class="card-body">
                    <h3 class="text-warning">{{ total_count }}</h3>
                    <p class="text-muted mb-0">Total Categories</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-success">
                <div class="card-body">
                    <h3 class="text-success">{{ matches|selectattr('confidence', '>', 0.8)|list|length }}</h3>
                    <p class="text-muted mb-0">High Confidence</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-info">
                <div class="card-body">
                    <h3 class="text-info">{{ matches|sum(attribute='count') }}</h3>
                    <p class="text-muted mb-0">Total Entries</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card text-center border-primary">
                <div class="card-body">
                    <h3 class="text-primary">{{ ((matches|sum(attribute='count') / matches|length)|round(1)) if matches else 0 }}</h3>
                    <p class="text-muted mb-0">Avg per Category</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Match Cards -->
    <div class="row">
        {% for match in matches %}
        <div class="col-lg-6 mb-4">
            <div class="match-card">
                <div class="d-flex align-items-center mb-3">
                    {% if 'Clinical' in match.category %}
                        <div class="category-icon category-clinical">
                            <i class="bi bi-heart-pulse"></i>
                        </div>
                    {% elif 'Research' in match.category %}
                        <div class="category-icon category-research">
                            <i class="bi bi-search"></i>
                        </div>
                    {% elif 'Policy' in match.category %}
                        <div class="category-icon category-policy">
                            <i class="bi bi-shield-check"></i>
                        </div>
                    {% else %}
                        <div class="category-icon category-unknown">
                            <i class="bi bi-question-circle"></i>
                        </div>
                    {% endif %}
                    
                    <div class="flex-grow-1">
                        <h5 class="mb-1">{{ match.category.replace('_', ' ').title() }}</h5>
                        <p class="text-muted mb-0">{{ match.description }}</p>
                    </div>
                    
                    <div class="text-center">
                        <div class="progress-ring">
                            <svg width="60" height="60">
                                <circle cx="30" cy="30" r="25" class="progress" 
                                        style="stroke-dashoffset: {{ 188.5 - (188.5 * match.confidence) }}"></circle>
                            </svg>
                        </div>
                        <small class="text-muted">{{ (match.confidence * 100)|round }}%</small>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-6">
                        <div class="d-flex align-items-center">
                            <i class="bi bi-collection me-2 text-muted"></i>
                            <div>
                                <strong class="text-primary">{{ match.count }}</strong>
                                <small class="text-muted d-block">Entries</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="d-flex align-items-center">
                            <i class="bi bi-speedometer2 me-2 text-muted"></i>
                            <div>
                                {% if match.confidence >= 0.8 %}
                                    <span class="confidence-badge confidence-high">High</span>
                                {% elif match.confidence >= 0.6 %}
                                    <span class="confidence-badge confidence-medium">Medium</span>
                                {% else %}
                                    <span class="confidence-badge confidence-low">Low</span>
                                {% endif %}
                                <small class="text-muted d-block">Confidence</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3 pt-3 border-top">
                    <small class="text-muted">
                        <i class="bi bi-robot me-1"></i>
                        AI Category Match #{{ match.id }}
                        <span class="ms-3">
                            <i class="bi bi-graph-up me-1"></i>
                            {{ (match.confidence * 100)|round }}% accuracy
                        </span>
                    </small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <!-- Empty State -->
    <div class="row">
        <div class="col-12">
            <div class="card text-center py-5">
                <div class="card-body">
                    <i class="bi bi-cpu display-1 text-muted mb-4"></i>
                    <h3 class="text-muted">No AI Matches Found</h3>
                    <p class="text-muted mb-4">No AI categorization matches have been created yet. The system will automatically analyze and categorize healthcare data.</p>
                    <a href="/admin/agents" class="btn btn-warning">
                        <i class="bi bi-robot me-2"></i>Check AI Agents
                    </a>
                </div>
            </div>
        </div>
    </div>
    {% endif %}
</div>
{% endblock %}
EOF

echo "5. Setting correct permissions..."
chown -R www-data:www-data backend/templates/healthpin/
chmod -R 644 backend/templates/healthpin/*.html

echo "6. Restarting service to apply new templates..."
systemctl restart mediamap

echo ""
echo "🎨 HEALTHPIN PAGE STYLING FIX COMPLETE!"
echo ""
echo "✅ All pages now match your app's professional design:"
echo "   • Bootstrap 5 styling with custom admin theme"
echo "   • Consistent card layouts with colored left borders"
echo "   • Professional color scheme (primary, success, info, warning)"
echo "   • Bootstrap Icons throughout"
echo "   • Hover effects and smooth transitions"
echo "   • Responsive grid layouts"
echo "   • Proper empty states with call-to-action buttons"
echo "   • Statistics cards at the top of each page"
echo "   • Gradient headers matching the theme"
echo ""
echo "🔗 Test the new styling:"
echo "   • 👥 Clinical Cases: Professional patient cards with badges"
echo "   • 👨‍⚕️ South African Doctors: Beautiful doctor profiles with specialties"
echo "   • 📋 Health Records: Clean record cards with source badges"
echo "   • 🤖 AI Matches: Interactive category cards with confidence rings"
echo ""
echo "All pages now integrate seamlessly with your app's design! 🎯"
