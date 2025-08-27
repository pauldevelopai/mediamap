"""
AIMAP Data Transformers
Data normalization and cleaning functions
"""
import re
from typing import Dict, List, Any

def normalize_signals(raw_signals: Dict) -> Dict:
    """Normalize and clean raw signals data"""
    normalized = {}
    
    for key, value in raw_signals.items():
        if isinstance(value, str):
            # Clean string values
            normalized[key] = value.strip()
        elif isinstance(value, (int, float)):
            # Ensure numeric values are non-negative
            normalized[key] = max(0, value)
        elif isinstance(value, list):
            # Clean and deduplicate lists
            if key == 'detected_tools':
                normalized[key] = deduplicate_tools(value)
            else:
                normalized[key] = list(set(value))
        else:
            normalized[key] = value
    
    return normalized

def deduplicate_tools(tools: List[str]) -> List[str]:
    """Deduplicate and normalize AI tool names"""
    if not tools:
        return []
    
    # Normalize tool names
    normalized_tools = []
    for tool in tools:
        if not tool:
            continue
        
        # Convert to lowercase and strip
        tool = tool.lower().strip()
        
        # Normalize common variations
        tool_mapping = {
            'gpt-4': 'chatgpt',
            'gpt-3': 'chatgpt',
            'gpt': 'chatgpt',
            'openai': 'chatgpt',
            'bard': 'gemini',
            'dall-e': 'dalle',
            'dall-e 2': 'dalle',
            'dall-e 3': 'dalle',
            'stable diffusion': 'stable-diffusion',
            'eleven labs': 'elevenlabs',
            'otter ai': 'otter.ai',
            'rev ai': 'rev.ai'
        }
        
        normalized_tool = tool_mapping.get(tool, tool)
        
        # Remove duplicates case-insensitively
        if normalized_tool not in [t.lower() for t in normalized_tools]:
            normalized_tools.append(normalized_tool)
    
    return sorted(normalized_tools)

def extract_date_mentions(text: str) -> List[str]:
    """Extract date mentions from text (for timeline analysis)"""
    if not text:
        return []
    
    # Pattern for common date formats
    date_patterns = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
        r'\b\d{1,2}\/\d{1,2}\/\d{4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text.lower())
        dates.extend(matches)
    
    return list(set(dates))

def calculate_feature_confidence(signals: Dict) -> Dict[str, float]:
    """Calculate confidence scores for extracted features"""
    confidence = {}
    
    for feature, value in signals.items():
        if isinstance(value, (int, float)):
            # Higher values generally indicate higher confidence
            if value == 0:
                confidence[feature] = 0.0
            elif value >= 3:
                confidence[feature] = 1.0
            else:
                confidence[feature] = value / 3.0
        elif isinstance(value, bool):
            confidence[feature] = 1.0 if value else 0.0
        elif isinstance(value, list):
            # List length indicates confidence
            confidence[feature] = min(1.0, len(value) / 3.0)
        else:
            confidence[feature] = 0.5  # Default moderate confidence
    
    return confidence

def merge_signals(old_signals: Dict, new_signals: Dict) -> Dict:
    """Merge old and new signals, keeping the most comprehensive data"""
    if not old_signals:
        return new_signals
    if not new_signals:
        return old_signals
    
    merged = old_signals.copy()
    
    for key, new_value in new_signals.items():
        if key not in merged:
            merged[key] = new_value
        else:
            old_value = merged[key]
            
            # Merge strategy based on data type
            if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                # Take maximum for numeric values
                merged[key] = max(old_value, new_value)
            elif isinstance(new_value, list) and isinstance(old_value, list):
                # Combine and deduplicate lists
                combined = old_value + new_value
                if key == 'detected_tools':
                    merged[key] = deduplicate_tools(combined)
                else:
                    merged[key] = list(set(combined))
            elif isinstance(new_value, bool) and isinstance(old_value, bool):
                # OR logic for boolean values
                merged[key] = old_value or new_value
            else:
                # Take new value if different type or string
                merged[key] = new_value
    
    return merged
