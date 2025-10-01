#!/bin/bash

# Lightsail Instance Recovery Script
# ==================================
# This script helps recover from instance crashes

set -e

echo "🚨 Lightsail Instance Recovery"
echo "============================="
echo ""
echo "Current situation:"
echo "❌ Instance crashed with UPSTREAM_ERROR [515]"
echo "❌ SSH and browser access unavailable"
echo "✅ All code and scripts available locally"
echo ""

# Configuration (will be updated with new IP)
OLD_IP="35.176.169.218"
NEW_IP=""

echo "🔧 Recovery Options:"
echo ""
echo "1. 🔄 Try to restart current instance"
echo "2. 🆕 Create new instance and deploy"
echo "3. 📋 Update scripts with new IP"
echo "4. 🚀 Full deployment to new instance"
echo "5. ❌ Exit"
echo ""

read -p "Choose recovery option (1-5): " choice

case $choice in
    1)
        echo "🔄 Attempting to restart current instance..."
        echo ""
        echo "📋 Manual steps required:"
        echo "1. Go to AWS Lightsail Console"
        echo "2. Select your instance"
        echo "3. Click 'Stop' → Wait 30 seconds"
        echo "4. Click 'Start' → Wait 2-3 minutes"
        echo "5. Try connecting again"
        echo ""
        echo "💡 If restart works, run: ./lightsail-manager.sh"
        ;;
    2)
        echo "🆕 Creating new instance..."
        echo ""
        echo "📋 Manual steps required:"
        echo "1. Go to AWS Lightsail Console"
        echo "2. Click 'Create instance'"
        echo "3. Choose: Ubuntu 22.04, eu-west-2a"
        echo "4. Select same bundle size"
        echo "5. Use default SSH key"
        echo "6. Note the new IP address"
        echo ""
        echo "💡 After creation, run this script again and choose option 3"
        ;;
    3)
        echo "📋 Updating scripts with new IP..."
        echo ""
        read -p "Enter the new IP address: " NEW_IP
        
        if [ -z "$NEW_IP" ]; then
            echo "❌ No IP address provided"
            exit 1
        fi
        
        echo "🔧 Updating scripts with IP: $NEW_IP"
        
        # Update all scripts with new IP
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" connect-lightsail.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" update-lightsail.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" push-to-github.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" lightsail-manager.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" run_quick_fix.sh
        
        echo "✅ Scripts updated with new IP: $NEW_IP"
        echo ""
        echo "💡 Next steps:"
        echo "1. Test connection: ./connect-lightsail.sh"
        echo "2. Deploy application: ./update-lightsail.sh"
        ;;
    4)
        echo "🚀 Full deployment to new instance..."
        echo ""
        read -p "Enter the new IP address: " NEW_IP
        
        if [ -z "$NEW_IP" ]; then
            echo "❌ No IP address provided"
            exit 1
        fi
        
        echo "🔧 Updating scripts and deploying..."
        
        # Update scripts
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" connect-lightsail.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" update-lightsail.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" push-to-github.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" lightsail-manager.sh
        sed -i.bak "s/$OLD_IP/$NEW_IP/g" run_quick_fix.sh
        
        echo "✅ Scripts updated"
        echo ""
        echo "🚀 Starting deployment..."
        
        # Wait for instance to be ready
        echo "⏳ Waiting for instance to be ready..."
        sleep 30
        
        # Deploy application
        ./update-lightsail.sh
        
        echo "🎉 Deployment completed!"
        echo ""
        echo "🌐 Application URL: http://$NEW_IP:8000"
        echo "🔑 Admin login: admin / admin123"
        ;;
    5)
        echo "👋 Exiting recovery script"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
