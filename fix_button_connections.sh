#!/bin/bash
echo "🔧 FIXING BUTTON CONNECTIONS TO MONITOR"
echo "======================================="

# Create the button connection fix
cat > /tmp/fix_button_connections.py << 'EOF'
import re

print("🔧 Fixing individual agent button connections...")

agents_template = '/opt/mediamap/backend/templates/admin/agents.html'

with open(agents_template, 'r') as f:
    content = f.read()

print("Original content length:", len(content))

# Find and fix the MediaMap agent buttons
mediamap_start_pattern = r'(<button[^>]*class="[^"]*btn-success[^"]*"[^>]*>\s*<i class="bi bi-play"></i> Start\s*</button>)'
mediamap_stop_pattern = r'(<button[^>]*class="[^"]*btn-danger[^"]*"[^>]*>\s*<i class="bi bi-stop"></i> Stop\s*</button>)'

# Add onclick handlers for MediaMap buttons
mediamap_start_replacement = r'<button class="btn btn-success me-2" onclick="startAgent(\'mediamap\')" title="Start MediaMap agent">\n                            <i class="bi bi-play"></i> Start\n                        </button>'

mediamap_stop_replacement = r'<button class="btn btn-danger" onclick="stopAgent(\'mediamap\')" title="Stop MediaMap agent">\n                            <i class="bi bi-stop"></i> Stop\n                        </button>'

# Apply MediaMap button fixes
content = re.sub(mediamap_start_pattern, mediamap_start_replacement, content)
content = re.sub(mediamap_stop_pattern, mediamap_stop_replacement, content)

# Find and fix the HealthPIN agent buttons  
healthpin_start_pattern = r'(<button[^>]*class="[^"]*btn-success[^"]*"[^>]*>\s*<i class="bi bi-play"></i> Start\s*</button>)'
healthpin_stop_pattern = r'(<button[^>]*class="[^"]*btn-danger[^"]*"[^>]*>\s*<i class="bi bi-stop"></i> Stop\s*</button>)'

# We need to be more specific since there are multiple Start/Stop buttons
# Let's find the HealthPIN section and replace buttons there

# Split content to find HealthPIN section
lines = content.split('\n')
healthpin_section_start = -1
healthpin_section_end = -1

for i, line in enumerate(lines):
    if 'HealthPIN Agent' in line and 'Healthcare data analysis' in line:
        healthpin_section_start = i
    if healthpin_section_start != -1 and 'Recent Agent Activity' in line:
        healthpin_section_end = i
        break

if healthpin_section_start != -1 and healthpin_section_end != -1:
    print(f"Found HealthPIN section from line {healthpin_section_start} to {healthpin_section_end}")
    
    # Process HealthPIN section
    for i in range(healthpin_section_start, healthpin_section_end):
        if '<i class="bi bi-play"></i> Start' in lines[i]:
            lines[i] = re.sub(r'<button[^>]*>', '<button class="btn btn-success me-2" onclick="startAgent(\'healthpin\')" title="Start HealthPIN agent">', lines[i])
            print(f"Fixed HealthPIN start button on line {i}")
        elif '<i class="bi bi-stop"></i> Stop' in lines[i]:
            lines[i] = re.sub(r'<button[^>]*>', '<button class="btn btn-danger" onclick="stopAgent(\'healthpin\')" title="Stop HealthPIN agent">', lines[i])
            print(f"Fixed HealthPIN stop button on line {i}")

    content = '\n'.join(lines)

# Also fix MediaMap section similarly
mediamap_section_start = -1
mediamap_section_end = -1

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'MediaMap Agent' in line and 'Media industry business' in line:
        mediamap_section_start = i
    if mediamap_section_start != -1 and ('HealthPIN Agent' in line or i > mediamap_section_start + 50):
        mediamap_section_end = i
        break

if mediamap_section_start != -1 and mediamap_section_end != -1:
    print(f"Found MediaMap section from line {mediamap_section_start} to {mediamap_section_end}")
    
    # Process MediaMap section
    for i in range(mediamap_section_start, mediamap_section_end):
        if '<i class="bi bi-play"></i> Start' in lines[i] and 'onclick=' not in lines[i]:
            lines[i] = re.sub(r'<button[^>]*>', '<button class="btn btn-success me-2" onclick="startAgent(\'mediamap\')" title="Start MediaMap agent">', lines[i])
            print(f"Fixed MediaMap start button on line {i}")
        elif '<i class="bi bi-stop"></i> Stop' in lines[i] and 'onclick=' not in lines[i]:
            lines[i] = re.sub(r'<button[^>]*>', '<button class="btn btn-danger" onclick="stopAgent(\'mediamap\')" title="Stop MediaMap agent">', lines[i])
            print(f"Fixed MediaMap stop button on line {i}")

    content = '\n'.join(lines)

# Ensure the JavaScript functions exist and are properly defined
if 'function startAgent(' not in content:
    print("Adding missing startAgent function...")
    
    start_agent_function = '''
    function startAgent(agentName) {
        console.log('Starting agent:', agentName);
        addLogEntry(`🚀 Starting ${agentName.toUpperCase()} agent...`, 'info');
        
        fetch(`/api/agents/${agentName}/start`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addLogEntry(`✅ ${agentName.toUpperCase()} agent started successfully`, 'success');
                startAgentActivityMonitoring(agentName);
                updateAgentStatus(agentName, 'running');
            } else {
                addLogEntry(`❌ Failed to start ${agentName.toUpperCase()}: ${data.error || 'Unknown error'}`, 'error');
            }
        })
        .catch(error => {
            console.error('Start agent error:', error);
            addLogEntry(`❌ Error starting ${agentName.toUpperCase()}: ${error.message}`, 'error');
        });
    }
    
    function stopAgent(agentName) {
        console.log('Stopping agent:', agentName);
        addLogEntry(`🛑 Stopping ${agentName.toUpperCase()} agent...`, 'warning');
        
        fetch(`/api/agents/${agentName}/stop`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addLogEntry(`🛑 ${agentName.toUpperCase()} agent stopped`, 'warning');
                stopAgentActivityMonitoring(agentName);
                updateAgentStatus(agentName, 'stopped');
            } else {
                addLogEntry(`❌ Failed to stop ${agentName.toUpperCase()}: ${data.error || 'Unknown error'}`, 'error');
            }
        })
        .catch(error => {
            console.error('Stop agent error:', error);
            addLogEntry(`❌ Error stopping ${agentName.toUpperCase()}: ${error.message}`, 'error');
        });
    }'''
    
    # Insert before the last </script> tag
    last_script_end = content.rfind('</script>')
    if last_script_end != -1:
        content = content[:last_script_end] + start_agent_function + '\n' + content[last_script_end:]
        print("✅ Added startAgent and stopAgent functions")

print("Final content length:", len(content))

with open(agents_template, 'w') as f:
    f.write(content)

print("✅ Fixed button connections to monitor!")
EOF

echo "📤 Copying button fix to Lightsail..."
scp -i LightsailDefaultKey-eu-west-2.pem /tmp/fix_button_connections.py ubuntu@35.177.61.112:/opt/mediamap/

echo "🔧 Running button connection fix..."
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "cd /opt/mediamap && sudo chown ubuntu:ubuntu backend/templates/admin/agents.html && python3 fix_button_connections.py"

echo "🔄 Restarting service..."
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "sudo systemctl restart mediamap"

echo "⏳ Waiting for restart..."
sleep 5

echo ""
echo "✅ BUTTON CONNECTIONS FIXED!"
echo ""
echo "🎯 What's now working:"
echo "• Individual Start buttons now have onclick='startAgent(agentName)'"
echo "• Individual Stop buttons now have onclick='stopAgent(agentName)'"
echo "• Monitor will immediately show activity when you click Start"
echo "• Console logging added for debugging"
echo ""
echo "🧪 Test it now:"
echo "1. Go to: http://35.177.61.112/admin/agents"
echo "2. Open browser console (F12) to see debug messages"
echo "3. Click 'Start' on HealthPIN agent"
echo "4. Watch monitor show: '🚀 Starting HEALTHPIN agent...'"
echo "5. See real-time activity messages appear!"
