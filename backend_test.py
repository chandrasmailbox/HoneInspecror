
import requests
import sys
import os
import time
from datetime import datetime
import base64
import tempfile
import shutil

class HomeInspectorAPITester:
    def __init__(self, base_url="https://c47a728a-04bc-4679-acda-08269eb9cf07.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_video_path = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, data=data, files=files, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return success, response.json() if response.content else {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_video(self):
        """Create a small test video file for testing"""
        try:
            # Download a small sample video for testing
            print("Downloading sample video for testing...")
            sample_url = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
            response = requests.get(sample_url, stream=True)
            
            if response.status_code == 200:
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                self.test_video_path = temp_file.name
                
                # Write the content to the file
                with open(self.test_video_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                print(f"Test video created at: {self.test_video_path}")
                return True
            else:
                print(f"Failed to download sample video: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error creating test video: {str(e)}")
            return False

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success:
            print(f"API Response: {response}")
        return success

    def test_analyze_video(self):
        """Test video analysis endpoint"""
        if not self.test_video_path or not os.path.exists(self.test_video_path):
            print("❌ No test video available")
            return False
            
        try:
            with open(self.test_video_path, 'rb') as video_file:
                files = {'file': ('test_video.mp4', video_file, 'video/mp4')}
                
                print("Uploading video for analysis (this may take some time)...")
                success, response = self.run_test(
                    "Video Analysis",
                    "POST",
                    "analyze-video",
                    200,
                    files=files
                )
                
                if success:
                    print("Video analysis completed successfully")
                    print(f"Defects found: {response.get('summary', {}).get('total_defects_found', 0)}")
                    print(f"Frames analyzed: {response.get('summary', {}).get('frames_analyzed', 0)}")
                    print(f"Severity: {response.get('summary', {}).get('severity', 'unknown')}")
                    
                    # Check if we have defects_found in the response
                    if 'defects_found' in response and len(response['defects_found']) > 0:
                        print(f"First frame defects: {response['defects_found'][0].get('defects', [])}")
                    
                return success
                
        except Exception as e:
            print(f"❌ Error testing video analysis: {str(e)}")
            return False

    def test_get_inspections(self):
        """Test getting all inspections"""
        success, response = self.run_test(
            "Get All Inspections",
            "GET",
            "inspections",
            200
        )
        if success:
            print(f"Retrieved {len(response)} inspections")
        return success

    def test_get_inspection_by_id(self, inspection_id):
        """Test getting a specific inspection by ID"""
        success, response = self.run_test(
            "Get Inspection by ID",
            "GET",
            f"inspection/{inspection_id}",
            200
        )
        if success:
            print(f"Retrieved inspection: {response.get('id')}")
        return success

    def cleanup(self):
        """Clean up any test resources"""
        if self.test_video_path and os.path.exists(self.test_video_path):
            try:
                os.unlink(self.test_video_path)
                print(f"Removed test video: {self.test_video_path}")
            except Exception as e:
                print(f"Error removing test video: {str(e)}")

def main():
    # Setup
    tester = HomeInspectorAPITester()
    
    try:
        # Test API root endpoint
        if not tester.test_root_endpoint():
            print("❌ Root API test failed, stopping tests")
            return 1
            
        # Create test video
        if not tester.create_test_video():
            print("❌ Failed to create test video, stopping tests")
            return 1
            
        # Test video analysis
        if not tester.test_analyze_video():
            print("❌ Video analysis test failed")
            # Continue with other tests
            
        # Test getting all inspections
        if not tester.test_get_inspections():
            print("❌ Get inspections test failed")
            # Continue with other tests
            
        # If we have a successful video analysis, test getting that inspection
        # This would require storing the inspection ID from the analyze_video response
        
        # Print results
        print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
        return 0 if tester.tests_passed == tester.tests_run else 1
        
    finally:
        # Clean up resources
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())
      