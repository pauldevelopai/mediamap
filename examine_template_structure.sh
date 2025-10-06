#!/bin/bash
echo "🔍 EXAMINING ACTUAL TEMPLATE STRUCTURE"
cd /opt/mediamap

echo "1. Finding the HealthPIN dashboard template..."
find . -name "dashboard.html" -path "*/healthpin/*" 2>/dev/null

echo ""
echo "2. Checking if template exists..."
ls -la backend/templates/healthpin/dashboard.html

echo ""
echo "3. Looking at the template structure around the colorful boxes..."
echo "=== SEARCHING FOR CARD STRUCTURES ==="
grep -n -A 10 -B 5 "Total Patients" backend/templates/healthpin/dashboard.html

echo ""
echo "=== SEARCHING FOR CARD-TITLE ELEMENTS ==="
grep -n -A 5 -B 5 "card-title" backend/templates/healthpin/dashboard.html

echo ""
echo "=== SEARCHING FOR THE NUMBER 44 ==="
grep -n -A 5 -B 5 "44" backend/templates/healthpin/dashboard.html

echo ""
echo "=== SEARCHING FOR CARD-BODY ELEMENTS ==="
grep -n -A 3 -B 3 "card-body" backend/templates/healthpin/dashboard.html

echo ""
echo "4. Looking at the overall structure of colorful boxes section..."
echo "=== LINES 50-100 OF TEMPLATE ==="
sed -n '50,100p' backend/templates/healthpin/dashboard.html

echo ""
echo "=== LINES 100-150 OF TEMPLATE ==="
sed -n '100,150p' backend/templates/healthpin/dashboard.html

echo ""
echo "5. Checking what's currently in the template..."
echo "Template size:"
wc -l backend/templates/healthpin/dashboard.html

echo ""
echo "🔍 TEMPLATE EXAMINATION COMPLETE!"
echo "This will show us the exact HTML structure so we can add buttons correctly."
