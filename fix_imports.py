import os
import re

def fix_imports_in_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix imports - replace 'from backend.' with relative imports
        # But be careful not to break legitimate imports
        patterns = [
            (r'from backend\.models import', 'from models import'),
            (r'from backend\.aimap\.models import', 'from aimap.models import'),
            (r'from backend\.healthpin\.models import', 'from healthpin.models import'),
            (r'from backend\.auth import', 'from auth import'),
            (r'from backend\.session_manager import', 'from session_manager import'),
            (r'from backend\.prompt_manager import', 'from prompt_manager import'),
            (r'from backend\.training\.', 'from training.'),
            (r'from backend\.agents\.', 'from agents.'),
            (r'from backend\.api\.', 'from api.'),
            (r'from backend\.services\.', 'from services.'),
            (r'from backend\.healthpin\.', 'from healthpin.'),
            (r'from backend\.aimap\.', 'from aimap.'),
            (r'from backend\.datasafe_integration import', 'from datasafe_integration import'),
        ]
        
        for old_pattern, new_pattern in patterns:
            content = re.sub(old_pattern, new_pattern, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
        return False

# Process all Python files in backend directory
fixed_files = []
for root, dirs, files in os.walk('backend'):
    # Skip venv and __pycache__ directories
    dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.venv']]
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if fix_imports_in_file(filepath):
                fixed_files.append(filepath)

print(f'Fixed imports in {len(fixed_files)} files:')
for f in fixed_files[:10]:  # Show first 10
    print(f'  ✅ {f}')
if len(fixed_files) > 10:
    print(f'  ... and {len(fixed_files) - 10} more files')
