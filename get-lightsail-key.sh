#!/bin/bash

echo "🔑 Getting Lightsail SSH Key..."
echo "================================"

# Try to download the default key pair
echo "📥 Attempting to download default key pair..."

# Try different regions
REGIONS=("us-east-1" "eu-west-2" "us-west-2" "ap-southeast-1")

for region in "${REGIONS[@]}"; do
    echo "🔍 Trying region: $region"
    if aws lightsail download-default-key-pair --region $region --output text > LightsailDefaultKey-$region.pem 2>/dev/null; then
        chmod 400 LightsailDefaultKey-$region.pem
        echo "✅ Successfully downloaded key for region: $region"
        echo "📁 Key saved as: LightsailDefaultKey-$region.pem"
        break
    else
        echo "❌ Failed to download key for region: $region"
    fi
done

echo ""
echo "🔍 Available SSH keys:"
ls -la *.pem 2>/dev/null || echo "No .pem files found"

echo ""
echo "💡 If no keys were downloaded, you can:"
echo "   1. Go to your Lightsail console"
echo "   2. Click on your 'datasafe' instance"
echo "   3. Go to the 'Connect' tab"
echo "   4. Download the SSH key manually"
echo "   5. Place it in this directory"
echo ""
echo "🚀 Then run: ./deploy-via-ssh.sh" 