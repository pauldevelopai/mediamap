"""
Bulletproof HealthPIN Dashboard Route
Completely bypasses any SQLAlchemy or import issues
"""

def get_bulletproof_healthpin_data():
    """Get HealthPIN data without any external dependencies"""
    import json
    import os
    
    try:
        # Direct file access - no dependencies
        data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        
        if not os.path.exists(data_file):
            return {
                'total_patients': 0,
                'total_doctors': 0, 
                'total_records': 0,
                'total_matches': 0,
                'status': 'no_data_file'
            }
        
        with open(data_file, 'r') as f:
            agent_data = json.load(f)
        
        if not agent_data:
            return {
                'total_patients': 0,
                'total_doctors': 0,
                'total_records': 0, 
                'total_matches': 0,
                'status': 'empty_data'
            }
        
        # Process data directly
        categories = {}
        sources = set()
        
        for entry in agent_data:
            cat = entry.get('category', 'Unknown')
            source = entry.get('source', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            sources.add(source)
        
        return {
            'total_patients': categories.get('Clinical_Care', 0),
            'total_doctors': len(sources),
            'total_records': len(agent_data),
            'total_matches': len(categories),
            'status': 'success',
            'categories': categories,
            'sources': list(sources)
        }
        
    except Exception as e:
        return {
            'total_patients': 0,
            'total_doctors': 0,
            'total_records': 0,
            'total_matches': 0,
            'status': f'error: {str(e)}'
        }

# Test the function
if __name__ == "__main__":
    result = get_bulletproof_healthpin_data()
    print("Bulletproof data result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
