#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime
from training.model_manager import get_model_manager

def test_model_performance():
    """Test the trained model with real business questions"""
    
    print("🧪 MODEL PERFORMANCE VALIDATION")
    print("=" * 60)
    
    # Test questions based on your business domain
    test_questions = [
        "How can I implement AI in my media company?",
        "What's the ROI of AI automation for content creation?",
        "How do I optimize my digital transformation strategy?",
        "What AI tools should I use for audience analysis?",
        "How can I improve my content workflow with AI?",
        "What are the best practices for AI implementation?",
        "How do I measure AI success in media companies?",
        "What AI solutions work best for small media businesses?",
        "How can I automate my content calendar with AI?",
        "What's the cost-benefit analysis of AI tools?"
    ]
    
    # Initialize model manager
    try:
        manager = get_model_manager()
        print(f"✅ Model manager initialized")
        print(f"✅ Custom model loaded: {manager.is_model_loaded}")
        print(f"✅ OpenAI available: {manager.openai_client is not None}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize model manager: {e}")
        return
    
    # Test responses
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"Testing Question {i}/10: {question}")
        
        start_time = time.time()
        
        try:
            # Test with custom model first
            if manager.is_model_loaded:
                response, source = manager.generate_response(question)
                response_time = time.time() - start_time
                
                results.append({
                    'question': question,
                    'response': response,
                    'source': source,
                    'response_time': response_time,
                    'success': True
                })
                
                print(f"   ✅ Response ({source}): {response[:100]}...")
                print(f"   ⏱️  Time: {response_time:.2f}s")
                
            else:
                # Fallback to OpenAI
                response, source = manager.generate_response(question)
                response_time = time.time() - start_time
                
                results.append({
                    'question': question,
                    'response': response,
                    'source': source,
                    'response_time': response_time,
                    'success': True
                })
                
                print(f"   ✅ Response ({source}): {response[:100]}...")
                print(f"   ⏱️  Time: {response_time:.2f}s")
                
        except Exception as e:
            response_time = time.time() - start_time
            results.append({
                'question': question,
                'response': f"Error: {str(e)}",
                'source': 'error',
                'response_time': response_time,
                'success': False
            })
            print(f"   ❌ Error: {e}")
        
        print()
    
    # Calculate performance metrics
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    if successful_tests:
        avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
        custom_model_responses = [r for r in successful_tests if r['source'] == 'custom_model']
        openai_responses = [r for r in successful_tests if r['source'] == 'openai']
        
        print("📊 PERFORMANCE RESULTS:")
        print("-" * 30)
        print(f"✅ Successful tests: {len(successful_tests)}/10")
        print(f"❌ Failed tests: {len(failed_tests)}/10")
        print(f"⏱️  Average response time: {avg_response_time:.2f}s")
        print(f"🤖 Custom model responses: {len(custom_model_responses)}")
        print(f"🌐 OpenAI responses: {len(openai_responses)}")
        
        if custom_model_responses:
            custom_avg_time = sum(r['response_time'] for r in custom_model_responses) / len(custom_model_responses)
            print(f"⚡ Custom model avg time: {custom_avg_time:.2f}s")
        
        if openai_responses:
            openai_avg_time = sum(r['response_time'] for r in openai_responses) / len(openai_responses)
            print(f"🌐 OpenAI avg time: {openai_avg_time:.2f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training/model_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'avg_response_time': avg_response_time if successful_tests else 0,
            'custom_model_responses': len(custom_model_responses),
            'openai_responses': len(openai_responses),
            'results': results
        }, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    test_model_performance() 