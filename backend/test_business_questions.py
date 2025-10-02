#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime
from training.model_manager import get_model_manager

def test_business_questions():
    """Test the model with real business questions and compare with OpenAI"""
    
    print("💼 BUSINESS QUESTION TESTING")
    print("=" * 60)
    
    # Real business questions from your domain
    business_questions = [
        "How can I implement AI in my media company to improve content creation?",
        "What's the ROI of implementing AI automation for my content workflow?",
        "How do I optimize my digital transformation strategy for a media business?",
        "What AI tools should I use for audience analysis and engagement?",
        "How can I improve my content calendar and scheduling with AI?",
        "What are the best practices for AI implementation in media companies?",
        "How do I measure the success of AI tools in my content strategy?",
        "What AI solutions work best for small to medium media businesses?",
        "How can I automate my social media content creation with AI?",
        "What's the cost-benefit analysis of different AI tools for media?"
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
    
    # Test results
    test_results = []
    
    for i, question in enumerate(business_questions, 1):
        print(f"Testing Question {i}/10:")
        print(f"Q: {question}")
        print()
        
        # Test custom model
        if manager.is_model_loaded:
            print("🤖 Custom Model Response:")
            start_time = time.time()
            try:
                response, source = manager.generate_response(question)
                response_time = time.time() - start_time
                
                print(f"   {response}")
                print(f"   ⏱️  Time: {response_time:.2f}s")
                print(f"   📏 Length: {len(response)} characters")
                
                test_results.append({
                    'question': question,
                    'model': 'custom',
                    'response': response,
                    'response_time': response_time,
                    'length': len(response),
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_results.append({
                    'question': question,
                    'model': 'custom',
                    'response': f"Error: {str(e)}",
                    'response_time': 0,
                    'length': 0,
                    'success': False
                })
        
        # Test OpenAI for comparison
        if manager.openai_client:
            print("🌐 OpenAI Response:")
            start_time = time.time()
            try:
                response, source = manager.generate_response(question)
                response_time = time.time() - start_time
                
                print(f"   {response}")
                print(f"   ⏱️  Time: {response_time:.2f}s")
                print(f"   📏 Length: {len(response)} characters")
                
                test_results.append({
                    'question': question,
                    'model': 'openai',
                    'response': response,
                    'response_time': response_time,
                    'length': len(response),
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_results.append({
                    'question': question,
                    'model': 'openai',
                    'response': f"Error: {str(e)}",
                    'response_time': 0,
                    'length': 0,
                    'success': False
                })
        
        print("-" * 60)
        print()
    
    # Analyze results
    analyze_test_results(test_results)
    
    # Save results
    save_test_results(test_results)
    
    return test_results

def analyze_test_results(results):
    """Analyze and compare test results"""
    print("📊 TEST RESULTS ANALYSIS:")
    print("=" * 40)
    
    custom_results = [r for r in results if r['model'] == 'custom' and r['success']]
    openai_results = [r for r in results if r['model'] == 'openai' and r['success']]
    
    if custom_results:
        custom_avg_time = sum(r['response_time'] for r in custom_results) / len(custom_results)
        custom_avg_length = sum(r['length'] for r in custom_results) / len(custom_results)
        
        print(f"🤖 Custom Model ({len(custom_results)} responses):")
        print(f"   ⏱️  Average response time: {custom_avg_time:.2f}s")
        print(f"   📏 Average response length: {custom_avg_length:.0f} characters")
        
        # Show sample responses
        print(f"   📝 Sample responses:")
        for i, result in enumerate(custom_results[:2]):
            print(f"      Q{i+1}: {result['question'][:50]}...")
            print(f"      A{i+1}: {result['response'][:100]}...")
    
    if openai_results:
        openai_avg_time = sum(r['response_time'] for r in openai_results) / len(openai_results)
        openai_avg_length = sum(r['length'] for r in openai_results) / len(openai_results)
        
        print(f"🌐 OpenAI ({len(openai_results)} responses):")
        print(f"   ⏱️  Average response time: {openai_avg_time:.2f}s")
        print(f"   📏 Average response length: {openai_avg_length:.0f} characters")
        
        # Show sample responses
        print(f"   📝 Sample responses:")
        for i, result in enumerate(openai_results[:2]):
            print(f"      Q{i+1}: {result['question'][:50]}...")
            print(f"      A{i+1}: {result['response'][:100]}...")
    
    # Compare performance
    if custom_results and openai_results:
        print(f"\n🔄 COMPARISON:")
        time_diff = custom_avg_time - openai_avg_time
        length_diff = custom_avg_length - openai_avg_length
        
        print(f"   ⏱️  Time difference: {time_diff:+.2f}s ({'slower' if time_diff > 0 else 'faster'})")
        print(f"   📏 Length difference: {length_diff:+.0f} chars ({'longer' if length_diff > 0 else 'shorter'})")

def save_test_results(results):
    """Save test results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training/business_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results) // 2,  # Assuming 2 models per question
            'results': results
        }, f, indent=2)
    
    print(f"📁 Test results saved to: {results_file}")

if __name__ == "__main__":
    test_business_questions() 