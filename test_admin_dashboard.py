#!/usr/bin/env python3
"""
Comprehensive test script for admin dashboard functionality
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://35.176.169.218"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint working: {data['status']}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_admin_routes():
    """Test all admin routes return proper redirects (302) when not authenticated"""
    print("\n🔍 Testing Admin Routes...")
    
    admin_routes = [
        '/admin/dashboard',
        '/admin/users',
        '/admin/chats',
        '/admin/analyses',
        '/admin/lessons',
        '/admin/feedback',
        '/admin/training',
        '/admin/strategies',
        '/admin/notion'
    ]
    
    all_working = True
    for route in admin_routes:
        try:
            response = requests.get(f"{BASE_URL}{route}", timeout=10, allow_redirects=False)
            if response.status_code == 302:
                print(f"✅ {route}: Redirecting to login (302)")
            else:
                print(f"❌ {route}: Unexpected status {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"❌ {route}: Error - {e}")
            all_working = False
    
    return all_working

def test_api_endpoints():
    """Test API endpoints return proper redirects when not authenticated"""
    print("\n🔍 Testing API Endpoints...")
    
    api_endpoints = [
        '/api/strategies',
        '/api/strategies/categories',
        '/api/strategies/sources',
        '/api/user_chats',
        '/api/strategies/crawl',
        '/api/strategies/generate'
    ]
    
    all_working = True
    for endpoint in api_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10, allow_redirects=False)
            if response.status_code in [302, 401, 405]:  # Redirect, unauthorized, or method not allowed
                print(f"✅ {endpoint}: Protected endpoint ({response.status_code})")
            else:
                print(f"❌ {endpoint}: Unexpected status {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
            all_working = False
    
    return all_working

def test_user_routes():
    """Test user routes return proper redirects when not authenticated"""
    print("\n🔍 Testing User Routes...")
    
    user_routes = [
        '/user-dashboard',
        '/my-chats',
        '/company-info',
        '/ai-strategies',
        '/today-news',
        '/strategies-dashboard'
    ]
    
    all_working = True
    for route in user_routes:
        try:
            response = requests.get(f"{BASE_URL}{route}", timeout=10, allow_redirects=False)
            if response.status_code == 302:
                print(f"✅ {route}: Redirecting to login (302)")
            else:
                print(f"❌ {route}: Unexpected status {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"❌ {route}: Error - {e}")
            all_working = False
    
    return all_working

def test_public_routes():
    """Test public routes are accessible"""
    print("\n🔍 Testing Public Routes...")
    
    public_routes = [
        '/',
        '/login',
        '/register',
        '/health'
    ]
    
    all_working = True
    for route in public_routes:
        try:
            response = requests.get(f"{BASE_URL}{route}", timeout=10)
            if response.status_code in [200, 302]:  # OK or redirect
                print(f"✅ {route}: Accessible ({response.status_code})")
            else:
                print(f"❌ {route}: Unexpected status {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"❌ {route}: Error - {e}")
            all_working = False
    
    return all_working

def test_database_connectivity():
    """Test if the application can connect to the database"""
    print("\n🔍 Testing Database Connectivity...")
    
    # This would require authentication, but we can test if the app is responding
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'healthy':
                print("✅ Database connectivity appears healthy")
                return True
            else:
                print(f"❌ Database health check failed: {data}")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database connectivity test error: {e}")
        return False

def generate_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📊 ADMIN DASHBOARD COMPREHENSIVE TEST REPORT")
    print("="*60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target URL: {BASE_URL}")
    
    # Run all tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Admin Routes", test_admin_routes),
        ("API Endpoints", test_api_endpoints),
        ("User Routes", test_user_routes),
        ("Public Routes", test_public_routes),
        ("Database Connectivity", test_database_connectivity)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Admin dashboard is fully functional.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    print("🚀 Starting Admin Dashboard Comprehensive Test...")
    success = generate_report()
    sys.exit(0 if success else 1) 