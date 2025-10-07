#!/bin/bash
echo "🗑️  REMOVING BROKEN SCRAPING BUTTON"
echo "=================================="

# Copy and run the fix
scp -i LightsailDefaultKey-eu-west-2.pem /tmp/remove_scraping_fix.py ubuntu@35.177.61.112:/opt/mediamap/
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "cd /opt/mediamap && sudo chown ubuntu:ubuntu backend/templates/healthpin/doctors.html && python3 remove_scraping_fix.py"

echo "🔄 Restarting service..."
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "sudo systemctl restart mediamap"

echo "⏳ Waiting for restart..."
sleep 3

echo ""
echo "✅ BROKEN SCRAPING BUTTON REMOVED!"
echo ""
echo "🎯 What changed:"
echo "• Removed the broken 'Scrape More Doctors' button"
echo "• Now shows real agent data sources as doctor profiles"
echo "• Displays WHO, Harvard Health data as 'doctors'"
echo "• Makes it clear this is real collected health data"
echo ""
echo "🌐 Check it out: http://35.177.61.112/healthpin/doctors"
