#!/bin/bash
echo "🤖 AI AGENT DOCTOR SCRAPING SCRIPT"
echo "=================================="
echo "This script triggers the HealthPIN AI agent to scrape South African doctors"
echo ""

# Copy the script to Lightsail and run it
scp -i LightsailDefaultKey-eu-west-2.pem doctor_scraping_agent.py ubuntu@35.177.61.112:/opt/mediamap/
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "cd /opt/mediamap && source venv/bin/activate && python3 doctor_scraping_agent.py"
