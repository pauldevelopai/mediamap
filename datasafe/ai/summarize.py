"""
Executive-style summarization using Hugging Face transformers
"""
from transformers import pipeline
from typing import Optional
import logging

from ..config import HF_SUMMARY_MODEL

logger = logging.getLogger(__name__)

class Summarizer:
    """Executive-style text summarization"""
    
    def __init__(self):
        self.summarizer = None
        self._initialize_summarizer()
    
    def _initialize_summarizer(self):
        """Initialize the summarization pipeline"""
        try:
            self.summarizer = pipeline(
                "summarization",
                model=HF_SUMMARY_MODEL,
                tokenizer=HF_SUMMARY_MODEL
            )
            logger.info(f"Initialized summarizer with model: {HF_SUMMARY_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {e}")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> Optional[str]:
        """
        Generate executive summary of text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in tokens
            min_length: Minimum length of summary in tokens
            
        Returns:
            Summary text or None if summarization fails
        """
        if not self.summarizer:
            logger.warning("Summarizer not available, returning truncated text")
            return text[:500] + "..." if len(text) > 500 else text
        
        try:
            # Truncate input text if too long (most models have token limits)
            max_input_length = 1024  # Conservative limit for BART
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
                logger.info("Truncated input text due to length")
            
            # Generate summary
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                early_stopping=True
            )
            
            summary = result[0]['summary_text']
            
            # Clean up the summary
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'
            
            logger.info(f"Generated summary of {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple truncation
            return text[:500] + "..." if len(text) > 500 else text
    
    def executive_summary(self, text: str) -> str:
        """
        Generate executive-style summary optimized for threat intelligence
        
        Args:
            text: Input text to summarize
            
        Returns:
            Executive summary focused on key threats and impacts
        """
        # For executive summaries, we want concise but informative content
        summary = self.summarize(text, max_length=100, min_length=30)
        
        if summary:
            # Ensure executive tone
            if not summary.startswith(('The ', 'This ', 'A ', 'An ')):
                summary = f"This threat intelligence report indicates that {summary.lower()}"
            
            return summary
        
        return "Summary not available due to processing limitations."

# Global summarizer instance
_summarizer = None

def get_summarizer() -> Summarizer:
    """Get or create global summarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer

def summarize_text(text: str, executive: bool = True) -> str:
    """
    Convenience function to summarize text
    
    Args:
        text: Input text to summarize
        executive: Whether to use executive-style formatting
        
    Returns:
        Summary text
    """
    summarizer = get_summarizer()
    if executive:
        return summarizer.executive_summary(text)
    else:
        return summarizer.summarize(text) or "Summary not available."
