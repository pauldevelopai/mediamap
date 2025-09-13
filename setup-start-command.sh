#!/bin/bash

# Setup script to add 'start' command to your shell

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZSH_CONFIG="$HOME/.zshrc"

echo "🚀 Setting up AIMAP start command..."

# Check if alias already exists
if grep -q "alias start=" "$ZSH_CONFIG" 2>/dev/null; then
    echo "✅ Start command already configured in $ZSH_CONFIG"
else
    # Add alias to .zshrc
    echo "" >> "$ZSH_CONFIG"
    echo "# AIMAP Start Command" >> "$ZSH_CONFIG"
    echo "alias start='$SCRIPT_DIR/start'" >> "$ZSH_CONFIG"
    echo "✅ Added start command to $ZSH_CONFIG"
fi

echo ""
echo "🎉 Setup complete! You can now use:"
echo "   start    - to start the AIMAP application"
echo ""
echo "📝 To use the command immediately, run:"
echo "   source $ZSH_CONFIG"
echo ""
echo "📍 Or restart your terminal"

















