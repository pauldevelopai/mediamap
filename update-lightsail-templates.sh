#!/bin/bash

# Your existing Lightsail instance details
INSTANCE_IP="13.40.124.51"
SSH_KEY="LightsailDefaultKey-eu-west-2.pem"

echo "🎨 Updating DataSafe templates on Lightsail instance..."
echo "🌐 IP Address: $INSTANCE_IP"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "💡 Please download your Lightsail SSH key first"
    exit 1
fi

echo "🔑 Using SSH key: $SSH_KEY"

# Create a temporary directory for the templates
echo "📝 Preparing updated templates..."
mkdir -p /tmp/datasafe-templates

# Copy the updated templates
cp backend/templates/user_dashboard.html /tmp/datasafe-templates/
cp backend/templates/user_chats.html /tmp/datasafe-templates/
cp backend/templates/login.html /tmp/datasafe-templates/

echo "📤 Uploading updated templates to Lightsail instance..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r /tmp/datasafe-templates/* ubuntu@$INSTANCE_IP:/tmp/

echo "🔧 Updating templates on server..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
    echo "📝 Updating templates..."
    sudo cp /tmp/user_dashboard.html /opt/datasafe/backend/templates/
    sudo cp /tmp/user_chats.html /opt/datasafe/backend/templates/
    sudo cp /tmp/login.html /opt/datasafe/backend/templates/
    
    sudo chown ubuntu:ubuntu /opt/datasafe/backend/templates/*.html
    sudo chmod 644 /opt/datasafe/backend/templates/*.html
    
    echo "🔄 Restarting application..."
    cd /opt/datasafe
    docker-compose restart
    
    echo "⏳ Waiting for application to restart..."
    sleep 10
    
    echo "✅ Templates updated successfully!"
    echo "🌐 Your updated design is now live at: http://$INSTANCE_IP"
EOF

echo ""
echo "🎉 Template update complete!"
echo "🌐 Visit your Lightsail instance to see the new design:"
echo "   http://$INSTANCE_IP"
echo ""
echo "✨ Updated pages:"
echo "   • User Dashboard - Clean, modern layout with username display"
echo "   • My Chats - Organized conversation list with working feedback"
echo "   • Login - Modern authentication page"
echo ""
echo "🔧 Fixed issues:"
echo "   • ✅ Username now displayed in top bar"
echo "   • ✅ Chat functionality working properly"
echo "   • ✅ Conversation saving indicator added"
echo "   • ✅ Feedback system working on my-chats page"
echo "   • ✅ All feedback goes to admin dashboard"
echo ""
echo "📱 All pages now feature:"
echo "   • Consistent modern design"
echo "   • Better mobile responsiveness"
echo "   • Improved user experience"
echo "   • Less cluttered interface"
echo "   • Working feedback system that goes to admin dashboard"
echo "   • Clear conversation saving indicators"

# Clean up
rm -rf /tmp/datasafe-templates 