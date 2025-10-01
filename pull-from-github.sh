#!/bin/bash

# Pull Changes from GitHub to Local
# =================================
# This script pulls the latest changes from GitHub to your local machine

set -e

echo "🚀 Pull Changes from GitHub to Local"
echo "==================================="
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository"
    echo "💡 Please run this script from the project root directory"
    exit 1
fi

# Check git status
echo "🔧 Checking local git status..."
git status --porcelain

echo ""
echo "Current branch: $(git branch --show-current)"
echo ""

# Ask for confirmation
read -p "🤔 Do you want to pull the latest changes from GitHub? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled"
    exit 1
fi

# Stash any local changes
echo "🔧 Stashing local changes..."
git stash push -m "Local changes before pull - $(date '+%Y-%m-%d %H:%M:%S')"

# Pull latest changes
echo "🔧 Pulling latest changes from GitHub..."
git pull origin main

# Apply stashed changes if any
echo "🔧 Checking for stashed changes..."
if git stash list | grep -q "Local changes before pull"; then
    echo "📋 Found stashed changes. You can apply them with:"
    echo "   git stash pop"
    echo ""
    read -p "🤔 Do you want to apply the stashed changes now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git stash pop
        echo "✅ Stashed changes applied"
    else
        echo "💡 You can apply them later with: git stash pop"
    fi
fi

echo ""
echo "🎉 GitHub pull completed!"
echo ""
echo "📋 Summary:"
echo "   - Latest changes pulled from GitHub"
echo "   - Local changes stashed (if any)"
echo ""
echo "💡 Next steps:"
echo "   1. Review the changes"
echo "   2. Test the application locally"
echo "   3. Run: ./update-lightsail.sh (to push to Lightsail)"
