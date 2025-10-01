#!/bin/bash

# Update Scripts with New IP Address
# ==================================
# This script updates all management scripts with a new IP address

set -e

if [ -z "$1" ]; then
    echo "❌ Usage: $0 <new_ip_address>"
    echo "💡 Example: $0 54.123.45.67"
    exit 1
fi

NEW_IP="$1"
OLD_IP="35.176.169.218"

echo "🔧 Updating Scripts with New IP"
echo "==============================="
echo "📍 Old IP: $OLD_IP"
echo "📍 New IP: $NEW_IP"
echo ""

# List of files to update
FILES=(
    "connect-lightsail.sh"
    "update-lightsail.sh"
    "push-to-github.sh"
    "lightsail-manager.sh"
    "run_quick_fix.sh"
    "deploy-fresh-instance.sh"
)

echo "🔧 Updating files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        # Create backup
        cp "$file" "$file.bak"
        
        # Update IP address
        sed -i.tmp "s/$OLD_IP/$NEW_IP/g" "$file"
        rm "$file.tmp"
        
        echo "✅ Updated: $file"
    else
        echo "⚠️  File not found: $file"
    fi
done

echo ""
echo "🎉 Scripts updated successfully!"
echo ""
echo "📋 Updated files:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    fi
done

echo ""
echo "💡 Next steps:"
echo "1. Test connection: ./connect-lightsail.sh"
echo "2. Deploy application: ./update-lightsail.sh"
echo "3. Or use manager: ./lightsail-manager.sh"
echo ""
echo "🌐 Application will be available at: http://$NEW_IP:8000"
