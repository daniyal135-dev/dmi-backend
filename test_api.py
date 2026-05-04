"""
Simple script to test the Image Analysis API
Run this while Django server is running
"""
import requests
import sys

# API endpoints
BASE_URL = "http://127.0.0.1:8000"
LOGIN_URL = f"{BASE_URL}/api/auth/login/"
IMAGE_ANALYSIS_URL = f"{BASE_URL}/api/analysis/image/"

def test_image_analysis():
    """Test the image analysis API"""
    
    # Step 1: Login and get token
    print("Step 1: Logging in...")
    username = input("Enter username (default: dani9099): ").strip() or "dani9099"
    password = input("Enter password: ").strip()
    
    login_data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(LOGIN_URL, json=login_data)
        response.raise_for_status()
        tokens = response.json()
        access_token = tokens.get('access')
        print(f"✓ Login successful!")
        print(f"  Access token: {access_token[:50]}...")
    except requests.exceptions.RequestException as e:
        print(f"✗ Login failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text}")
        return
    
    # Step 2: Test image analysis
    print("\nStep 2: Testing image analysis...")
    image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if not image_path:
        print("Skipping image analysis test.")
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            headers = {
                'Authorization': f'Bearer {access_token}'
            }
            
            print(f"Uploading image: {image_path}")
            response = requests.post(IMAGE_ANALYSIS_URL, files=files, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            print("\n✓ Image analysis successful!")
            print(f"\nResults:")
            print(f"  Verdict: {result['result']['verdict']}")
            print(f"  Confidence: {result['result']['confidence']:.2%}")
            print(f"  Explanation: {result['result']['explanation'][:100]}...")
            print(f"  Heatmap: {result['result'].get('heatmap_path', 'N/A')}")
            
    except FileNotFoundError:
        print(f"✗ Image file not found: {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Image analysis failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text}")

if __name__ == "__main__":
    print("="*60)
    print("Image Analysis API Test Script")
    print("="*60)
    print(f"Make sure Django server is running at {BASE_URL}")
    print()
    
    test_image_analysis()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

