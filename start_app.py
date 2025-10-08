#!/usr/bin/env python3
"""
Start MediaMap Multi-App Platform
=================================

This script starts the Flask application on an available port.
"""

import sys
import os
import socket

# Add backend to path
sys.path.append('backend')

def find_free_port(start_port=8080):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Start the Flask application"""
    
    print("🚀 STARTING MEDIAMAP MULTI-APP PLATFORM")
    print("=======================================")
    
    # Find a free port
    port = find_free_port(8080)
    if not port:
        print("❌ Could not find a free port")
        return
    
    print(f"🌐 Starting server on port {port}...")
    print(f"🔗 Access the app at: http://localhost:{port}")
    print("📱 Multi-App Architecture Features:")
    print("   • Single login → App selector")
    print("   • 4 app options: MediaMap, MediaMap Admin, HealthPIN, HealthPIN Admin")
    print("   • Filtered admin interfaces")
    print("   • Real HealthPIN data display")
    print("")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import app
        
        # Configure the app
        app.config['DEBUG'] = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        
        # Start the server
        app.run(
            host='127.0.0.1',
            port=port,
            debug=True,
            use_reloader=False  # Disable reloader to avoid issues
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check if all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify the backend/app.py file exists")

if __name__ == "__main__":
    main()
