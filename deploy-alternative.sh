#!/bin/bash
set -e

echo "🚀 Alternative MediaMap Deployment"
echo "=================================="
echo ""
echo "Since you don't have Lightsail permissions, here are your options:"
echo ""

# Check what cloud services you have access to
echo "🔍 Checking available cloud services..."

# Check EC2 access
if aws ec2 describe-instances --max-items 1 &>/dev/null; then
    echo "✅ You have EC2 access"
    EC2_AVAILABLE=true
else
    echo "❌ No EC2 access"
    EC2_AVAILABLE=false
fi

# Check if you have any existing instances
if [ "$EC2_AVAILABLE" = true ]; then
    echo "🔍 Checking for existing EC2 instances..."
    INSTANCES=$(aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' --output text 2>/dev/null | grep -v "None" || echo "")
    
    if [ -n "$INSTANCES" ]; then
        echo "📋 Found existing instances:"
        echo "$INSTANCES"
        echo ""
        echo "You can deploy to an existing instance using:"
        echo "  ./deploy-to-existing-instance.sh"
    else
        echo "❌ No existing EC2 instances found"
    fi
fi

echo ""
echo "🎯 Your Deployment Options:"
echo ""

if [ "$EC2_AVAILABLE" = true ]; then
    echo "1. 🖥️  Deploy to existing EC2 instance"
    echo "   - Use: ./deploy-to-existing-instance.sh"
    echo "   - Cost: Depends on instance size"
    echo ""
fi

echo "2. 🏠 Deploy locally with Docker"
echo "   - Use: docker-compose up --build"
echo "   - Cost: Free (your computer)"
echo "   - Access: http://localhost:8000"
echo ""

echo "3. 🌐 Deploy to any VPS/Cloud provider"
echo "   - DigitalOcean, Linode, Vultr, etc."
echo "   - Use: ./deploy-to-vps.sh (I can create this)"
echo "   - Cost: $5-20/month"
echo ""

echo "4. 🔧 Manual deployment to any server"
echo "   - I can create step-by-step instructions"
echo "   - Works with any Linux server"
echo ""

echo "5. 🆓 Free tier options"
echo "   - Railway, Render, Heroku (free tiers)"
echo "   - I can create deployment configs"
echo ""

echo "Which option would you like me to help you with?"
echo ""
echo "💡 Recommendation: Start with option 2 (local Docker) to test,"
echo "   then move to option 3 (VPS) for production."
