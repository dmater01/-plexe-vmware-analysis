#!/usr/bin/env python3
"""
Test script to verify complete Plex installation
"""
import os
import sys

def test_python_environment():
    """Test Python environment setup"""
    print("🐍 TESTING PYTHON ENVIRONMENT")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    print("✅ Python environment OK\n")

def test_dependencies():
    """Test all required dependencies"""
    print("📦 TESTING DEPENDENCIES")
    print("=" * 40)
    
    dependencies = [
        ("pandas", "pd.__version__"),
        ("numpy", "np.__version__"),
        ("sklearn", "sklearn.__version__"), 
        ("torch", "torch.__version__"),
        ("tensorflow", "tf.__version__"),
    ]
    
    for package, version_attr in dependencies:
        try:
            if package == "pandas":
                import pandas as pd
                print(f"✅ {package}: {eval(version_attr)}")
            elif package == "numpy":
                import numpy as np
                print(f"✅ {package}: {eval(version_attr)}")
            elif package == "sklearn":
                import sklearn
                print(f"✅ {package}: {eval(version_attr)}")
            elif package == "torch":
                import torch
                print(f"✅ {package}: {eval(version_attr)}")
            elif package == "tensorflow":
                import tensorflow as tf
                print(f"✅ {package}: {eval(version_attr)}")
        except ImportError as e:
            print(f"❌ {package}: Not installed - {e}")
            return False
    
    print("✅ All dependencies OK\n")
    return True

def test_plex():
    """Test Plex installation"""
    print("🤖 TESTING PLEX")
    print("=" * 40)
    
    try:
        import plex
        print("✅ Plex imported successfully")
        
        # Test basic functionality (without actually building a model)
        import pandas as pd
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        })
        
        # This should work without API key
        model_obj = plex.Model(data=test_data, intent="test intent")
        print("✅ Plex Model object creation OK")
        print("✅ Plex installation verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Plex test failed: {e}")
        return False

def test_openai_api():
    """Test OpenAI API configuration"""
    print("🔑 TESTING OPENAI API")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Set with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ API key format invalid (should start with 'sk-')")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connection
    try:
        import requests
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Simple API test (list models)
        response = requests.get(
            'https://api.openai.com/v1/models',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ OpenAI API connection successful")
            models = response.json()
            gpt_models = [m['id'] for m in models['data'] if 'gpt' in m['id']][:3]
            print(f"✅ Available models: {', '.join(gpt_models)}...")
            print("✅ OpenAI API verified\n")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_file_operations():
    """Test file operations in current directory"""
    print("📁 TESTING FILE OPERATIONS")  
    print("=" * 40)
    
    try:
        # Test write
        with open('test_file.txt', 'w') as f:
            f.write('test')
        
        # Test read
        with open('test_file.txt', 'r') as f:
            content = f.read()
        
        # Cleanup
        os.remove('test_file.txt')
        
        print("✅ File operations OK")
        print(f"✅ Working directory: {os.getcwd()}\n")
        return True
        
    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 PLEX INSTALLATION VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Python Environment", test_python_environment),
        ("Dependencies", test_dependencies), 
        ("Plex Installation", test_plex),
        ("OpenAI API", test_openai_api),
        ("File Operations", test_file_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Plex installation is ready for VMware analysis!")
        print("\nNext steps:")
        print("1. Place your Sample.csv in this directory")
        print("2. Run: python analyze_vmware.py")
    else:
        print(f"\n⚠️  {total-passed} TESTS FAILED")
        print("Please fix the issues above before proceeding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
