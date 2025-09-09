# 🚀 AIMAP Start Command

## Quick Start

Simply type `start` in your terminal to launch the AIMAP application!

```bash
start
```

## Stop the Application

Type `stop` to stop the AIMAP application:

```bash
stop
```

## What the Start Command Does

The `start` command automatically:

1. **🔍 Checks Environment**: Verifies virtual environment exists
2. **📦 Activates Environment**: Activates the Python virtual environment
3. **📋 Installs Dependencies**: Installs required packages if needed
4. **🗄️ Creates Database**: Sets up the database if it doesn't exist
5. **🔧 Handles Port Conflicts**: Stops any existing processes on port 8000
6. **🌟 Starts Application**: Launches the Flask web server

## Application Details

- **URL**: http://localhost:8000
- **Admin Login**: admin / admin123
- **Stop**: Press `Ctrl+C` in the terminal or use `stop` command

## Manual Start (Alternative)

If you prefer to start manually:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the application
python -m backend.app
```

## Troubleshooting

If the `start` command doesn't work:

1. **Restart your terminal** to reload shell configuration
2. **Run the setup again**: `./setup-start-command.sh`
3. **Manual start**: Use the manual commands above
4. **Check port conflicts**: The start script automatically handles port 8000 conflicts

## Files Created

- `start` - Main startup script
- `stop` - Stop script
- `start.sh` - Alternative startup script
- `setup-start-command.sh` - Setup script for shell integration
- `aimap-start` - Global startup script

## Available Commands

- `start` - Start the AIMAP application
- `stop` - Stop the AIMAP application

The start and stop commands are now available globally in your terminal! 🎉
