#!/bin/bash

# Setup OpenAI API Key for DataSafe AI
echo "=== DataSafe AI - OpenAI API Key Setup ==="
echo ""

# Check if running on Lightsail/EC2
if [ -f /etc/systemd/system/datasafe.service ]; then
    echo "Detected systemd service installation"
    SERVICE_FILE="/etc/systemd/system/datasafe.service"
    ENV_FILE="/etc/default/datasafe"
    
    # Create environment file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        echo "Creating environment file: $ENV_FILE"
        sudo touch "$ENV_FILE"
        sudo chown root:root "$ENV_FILE"
        sudo chmod 644 "$ENV_FILE"
    fi
    
    echo ""
    echo "Please enter your OpenAI API key:"
    read -s OPENAI_KEY
    
    if [ -n "$OPENAI_KEY" ]; then
        # Add to environment file
        if grep -q "OPENAI_API_KEY" "$ENV_FILE"; then
            sudo sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" "$ENV_FILE"
        else
            echo "OPENAI_API_KEY=$OPENAI_KEY" | sudo tee -a "$ENV_FILE" > /dev/null
        fi
        
        # Update systemd service to use environment file
        if ! grep -q "EnvironmentFile" "$SERVICE_FILE"; then
            sudo sed -i '/\[Service\]/a EnvironmentFile=/etc/default/datasafe' "$SERVICE_FILE"
        fi
        
        # Reload systemd and restart service
        sudo systemctl daemon-reload
        sudo systemctl restart datasafe
        
        echo ""
        echo "✅ OpenAI API key configured successfully!"
        echo "Service has been restarted with the new configuration."
    else
        echo "❌ No API key provided. Setup cancelled."
    fi

# Check if running with Docker
elif [ -f docker-compose.yml ]; then
    echo "Detected Docker installation"
    
    echo ""
    echo "Please enter your OpenAI API key:"
    read -s OPENAI_KEY
    
    if [ -n "$OPENAI_KEY" ]; then
        # Create .env file if it doesn't exist
        if [ ! -f .env ]; then
            touch .env
        fi
        
        # Add or update API key in .env file
        if grep -q "OPENAI_API_KEY" .env; then
            sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
        else
            echo "OPENAI_API_KEY=$OPENAI_KEY" >> .env
        fi
        
        # Restart Docker containers
        docker-compose down
        docker-compose up -d
        
        echo ""
        echo "✅ OpenAI API key configured successfully!"
        echo "Docker containers have been restarted with the new configuration."
    else
        echo "❌ No API key provided. Setup cancelled."
    fi

# Local development
else
    echo "Detected local development environment"
    
    echo ""
    echo "Please enter your OpenAI API key:"
    read -s OPENAI_KEY
    
    if [ -n "$OPENAI_KEY" ]; then
        # Export for current session
        export OPENAI_API_KEY="$OPENAI_KEY"
        
        # Add to shell profile for persistence
        SHELL_PROFILE=""
        if [ -f ~/.bashrc ]; then
            SHELL_PROFILE=~/.bashrc
        elif [ -f ~/.zshrc ]; then
            SHELL_PROFILE=~/.zshrc
        elif [ -f ~/.profile ]; then
            SHELL_PROFILE=~/.profile
        fi
        
        if [ -n "$SHELL_PROFILE" ]; then
            # Remove existing entry if present
            sed -i '/export OPENAI_API_KEY/d' "$SHELL_PROFILE"
            # Add new entry
            echo "export OPENAI_API_KEY=\"$OPENAI_KEY\"" >> "$SHELL_PROFILE"
            echo ""
            echo "✅ OpenAI API key added to $SHELL_PROFILE"
        fi
        
        echo ""
        echo "✅ OpenAI API key configured for current session!"
        echo "Please restart your application or run: source $SHELL_PROFILE"
    else
        echo "❌ No API key provided. Setup cancelled."
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To get an OpenAI API key:"
echo "1. Go to https://platform.openai.com/api-keys"
echo "2. Sign in or create an account"
echo "3. Click 'Create new secret key'"
echo "4. Copy the key and run this script again"
echo ""
echo "Note: Keep your API key secure and never share it publicly." 