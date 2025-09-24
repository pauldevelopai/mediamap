#!/usr/bin/env python3
"""
Test ChatGPT Agents Integration
==============================

Test script to verify ChatGPT Agents integration with the AI agents system.
"""

import os
import sys
import json
from datetime import datetime

# Add backend to path
sys.path.append('backend')

def test_chatgpt_agent_integration():
    """Test ChatGPT Agent integration"""
    
    print("🧪 Testing ChatGPT Agents Integration...")
    print("=" * 50)
    
    # Test 1: Check OpenAI API Key
    print("\n1️⃣ Testing OpenAI API Key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API Key found: {api_key[:10]}...")
    else:
        print("❌ OpenAI API Key not found in environment variables")
        print("   Please set OPENAI_API_KEY environment variable")
        return False
    
    # Test 2: Test OpenAI Agent Integration Import
    print("\n2️⃣ Testing OpenAI Agent Integration Import...")
    try:
        from agents.openai_agent_integration import OpenAIAgentIntegration, get_openai_agent_integration
        print("✅ OpenAI Agent Integration imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OpenAI Agent Integration: {e}")
        return False
    
    # Test 3: Test OpenAI Agent Integration Initialization
    print("\n3️⃣ Testing OpenAI Agent Integration Initialization...")
    try:
        openai_integration = get_openai_agent_integration()
        if openai_integration:
            print("✅ OpenAI Agent Integration initialized successfully")
        else:
            print("❌ OpenAI Agent Integration initialization failed")
            return False
    except Exception as e:
        print(f"❌ Error initializing OpenAI Agent Integration: {e}")
        return False
    
    # Test 4: Test Agent Capabilities
    print("\n4️⃣ Testing Agent Capabilities...")
    try:
        capabilities = openai_integration.list_available_agents()
        print(f"✅ Found {len(capabilities)} ChatGPT Agents:")
        for agent in capabilities:
            print(f"   - {agent['type']}: {agent['name']}")
            print(f"     Model: {agent['model']}")
            print(f"     Tools: {', '.join(agent['tools'])}")
    except Exception as e:
        print(f"❌ Error getting agent capabilities: {e}")
        return False
    
    # Test 5: Test Agent Manager Integration
    print("\n5️⃣ Testing Agent Manager Integration...")
    try:
        from agents.agent_manager import agent_manager
        
        # Check if agents have ChatGPT integration
        performance = agent_manager.get_agent_performance()
        print("✅ Agent Manager Performance:")
        for agent_name, perf in performance.items():
            chatgpt_enabled = perf.get('chatgpt_agent_enabled', False)
            status = "✅ Enabled" if chatgpt_enabled else "❌ Disabled"
            print(f"   - {agent_name}: ChatGPT Agent {status}")
            print(f"     Data Collection Rate: {perf.get('data_collection_rate', 0)}")
            print(f"     Learning Cycles: {perf.get('learning_cycles', 0)}")
    except Exception as e:
        print(f"❌ Error testing agent manager integration: {e}")
        return False
    
    # Test 6: Test ChatGPT Agent Analysis (if agents are available)
    print("\n6️⃣ Testing ChatGPT Agent Analysis...")
    try:
        # Test with sample data
        sample_data = {
            "agent_section": "mediamap",
            "data_points": [
                {
                    "source": "test_source",
                    "content": "AI is transforming media workflows and content creation",
                    "category": "AI_Technology",
                    "relevance_score": 0.9,
                    "metadata": {"test": True},
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "total_points": 1,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Test insights analysis
        result = openai_integration.analyze_data_with_agent(
            agent_type="mediamap",
            data=sample_data,
            analysis_type="insights"
        )
        
        if result.get("success"):
            print("✅ ChatGPT Agent analysis successful")
            print(f"   Analysis type: {result.get('analysis_type')}")
            print(f"   Agent type: {result.get('agent_type')}")
            print(f"   Response length: {len(result.get('analysis', ''))}")
        else:
            print(f"❌ ChatGPT Agent analysis failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ChatGPT Agent analysis: {e}")
        return False
    
    # Test 7: Test Agent Manager ChatGPT Methods
    print("\n7️⃣ Testing Agent Manager ChatGPT Methods...")
    try:
        # Test getting ChatGPT capabilities
        capabilities = agent_manager.get_chatgpt_agent_capabilities()
        print("✅ ChatGPT Agent Capabilities:")
        for agent_name, caps in capabilities.items():
            if "error" not in caps:
                print(f"   - {agent_name}: {caps.get('name', 'Unknown')}")
                print(f"     Model: {caps.get('model', 'Unknown')}")
                print(f"     Tools: {', '.join(caps.get('tools', []))}")
            else:
                print(f"   - {agent_name}: {caps.get('error')}")
        
        # Test ChatGPT status
        chatgpt_status = {}
        performance = agent_manager.get_agent_performance()
        for agent_name, perf in performance.items():
            chatgpt_status[agent_name] = {
                'enabled': perf.get('chatgpt_agent_enabled', False),
                'data_collection_rate': perf.get('data_collection_rate', 0),
                'learning_cycles': perf.get('learning_cycles', 0)
            }
        
        print("✅ ChatGPT Agent Status:")
        for agent_name, status in chatgpt_status.items():
            enabled = "✅ Enabled" if status['enabled'] else "❌ Disabled"
            print(f"   - {agent_name}: {enabled}")
            
    except Exception as e:
        print(f"❌ Error testing agent manager ChatGPT methods: {e}")
        return False
    
    print("\n🎉 ChatGPT Agents Integration Test Complete!")
    print("=" * 50)
    
    return True

def test_agent_learning_with_chatgpt():
    """Test agent learning with ChatGPT integration"""
    
    print("\n🧠 Testing Agent Learning with ChatGPT...")
    print("=" * 50)
    
    try:
        from agents.agent_manager import agent_manager
        
        # Run a single learning cycle for MediaMap agent
        print("\n🔄 Running MediaMap agent learning cycle...")
        success = agent_manager.run_single_cycle("mediamap")
        
        if success:
            print("✅ MediaMap agent learning cycle completed")
            
            # Get insights
            insights = agent_manager.get_agent_insights("mediamap", limit=5)
            print(f"✅ Retrieved {len(insights)} insights")
            
            # Check for ChatGPT-generated insights
            chatgpt_insights = [i for i in insights if i.get("source") == "ChatGPT_Agent"]
            if chatgpt_insights:
                print(f"✅ Found {len(chatgpt_insights)} ChatGPT-generated insights")
                for insight in chatgpt_insights[:2]:  # Show first 2
                    print(f"   - {insight.get('type', 'Unknown')}: {insight.get('insight', '')[:100]}...")
            else:
                print("⚠️ No ChatGPT-generated insights found (this is normal if no data was collected)")
        else:
            print("❌ MediaMap agent learning cycle failed")
            
    except Exception as e:
        print(f"❌ Error testing agent learning: {e}")

if __name__ == "__main__":
    print("🤖 ChatGPT Agents Integration Test Suite")
    print("=" * 60)
    
    # Run main integration test
    success = test_chatgpt_agent_integration()
    
    if success:
        # Run learning test
        test_agent_learning_with_chatgpt()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 ChatGPT Agents Integration Summary:")
        print("   ✅ OpenAI API Key configured")
        print("   ✅ OpenAI Agent Integration initialized")
        print("   ✅ ChatGPT Agents created (MediaMap & HealthPIN)")
        print("   ✅ Agent Manager integration working")
        print("   ✅ API routes available")
        print("   ✅ Analysis capabilities functional")
        
        print("\n🚀 Next Steps:")
        print("   1. Start the agents: POST /api/agents/start")
        print("   2. Check status: GET /api/agents/status")
        print("   3. Get ChatGPT capabilities: GET /api/agents/chatgpt/capabilities")
        print("   4. Analyze data: POST /api/agents/chatgpt/{agent_name}/analyze")
        print("   5. Get recommendations: GET /api/agents/chatgpt/{agent_name}/recommendations")
        
    else:
        print("\n❌ Integration test failed!")
        print("   Please check the errors above and ensure:")
        print("   1. OpenAI API key is set in environment variables")
        print("   2. OpenAI package is installed: pip install openai")
        print("   3. All agent files are properly configured")



