"""
Test script for the Network Intrusion Detection FastAPI
"""

import requests
import json
import pandas as pd

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("=== Health Check ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_list_models():
    """Test list models endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        print("\n=== Available Models ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"List models failed: {e}")
        return False

def create_sample_data():
    """Create sample network traffic data for testing"""
    # Sample data based on typical network traffic patterns
    sample_data = {
        "destination_port": 80.0,
        "flow_duration": 120000.0,
        "total_fwd_packets": 10.0,
        "total_length_fwd_packets": 1500.0,
        "fwd_packet_length_max": 200.0,
        "fwd_packet_length_min": 50.0,
        "fwd_packet_length_mean": 150.0,
        "fwd_packet_length_std": 25.0,
        "bwd_packet_length_max": 180.0,
        "bwd_packet_length_min": 40.0,
        "bwd_packet_length_mean": 140.0,
        "bwd_packet_length_std": 20.0,
        "flow_bytes_per_s": 12500.0,
        "flow_packets_per_s": 83.33,
        "flow_iat_mean": 12000.0,
        "flow_iat_std": 1000.0,
        "flow_iat_max": 15000.0,
        "flow_iat_min": 8000.0,
        "fwd_iat_total": 108000.0,
        "fwd_iat_mean": 12000.0,
        "fwd_iat_std": 1000.0,
        "fwd_iat_max": 15000.0,
        "fwd_iat_min": 8000.0,
        "bwd_iat_total": 96000.0,
        "bwd_iat_mean": 12000.0,
        "bwd_iat_std": 800.0,
        "bwd_iat_max": 14000.0,
        "bwd_iat_min": 9000.0,
        "fwd_header_length": 320.0,
        "bwd_header_length": 280.0,
        "fwd_packets_per_s": 41.67,
        "bwd_packets_per_s": 41.67,
        "min_packet_length": 40.0,
        "max_packet_length": 200.0,
        "packet_length_mean": 145.0,
        "packet_length_std": 22.5,
        "packet_length_variance": 506.25,
        "fin_flag_count": 2.0,
        "psh_flag_count": 4.0,
        "ack_flag_count": 18.0,
        "average_packet_size": 145.0,
        "subflow_fwd_bytes": 1500.0,
        "init_win_bytes_forward": 8192.0,
        "init_win_bytes_backward": 8192.0,
        "act_data_pkt_fwd": 8.0,
        "min_seg_size_forward": 20.0,
        "active_mean": 1000.0,
        "active_max": 2000.0,
        "active_min": 500.0,
        "idle_mean": 500.0,
        "idle_max": 1000.0,
        "idle_min": 100.0
    }
    return sample_data

def create_ddos_sample():
    """Create sample data that might indicate DDoS attack"""
    ddos_data = {
        "destination_port": 80.0,
        "flow_duration": 5000.0,  # Very short duration
        "total_fwd_packets": 1000.0,  # High packet count
        "total_length_fwd_packets": 50000.0,  # High total bytes
        "fwd_packet_length_max": 64.0,  # Small packets (typical for DDoS)
        "fwd_packet_length_min": 64.0,
        "fwd_packet_length_mean": 64.0,
        "fwd_packet_length_std": 0.0,
        "bwd_packet_length_max": 0.0,  # No response packets
        "bwd_packet_length_min": 0.0,
        "bwd_packet_length_mean": 0.0,
        "bwd_packet_length_std": 0.0,
        "flow_bytes_per_s": 10000000.0,  # Very high bytes per second
        "flow_packets_per_s": 200000.0,  # Very high packets per second
        "flow_iat_mean": 5.0,  # Very small inter-arrival time
        "flow_iat_std": 1.0,
        "flow_iat_max": 10.0,
        "flow_iat_min": 1.0,
        "fwd_iat_total": 5000.0,
        "fwd_iat_mean": 5.0,
        "fwd_iat_std": 1.0,
        "fwd_iat_max": 10.0,
        "fwd_iat_min": 1.0,
        "bwd_iat_total": 0.0,
        "bwd_iat_mean": 0.0,
        "bwd_iat_std": 0.0,
        "bwd_iat_max": 0.0,
        "bwd_iat_min": 0.0,
        "fwd_header_length": 20000.0,  # High header length
        "bwd_header_length": 0.0,
        "fwd_packets_per_s": 200000.0,
        "bwd_packets_per_s": 0.0,
        "min_packet_length": 64.0,
        "max_packet_length": 64.0,
        "packet_length_mean": 64.0,
        "packet_length_std": 0.0,
        "packet_length_variance": 0.0,
        "fin_flag_count": 0.0,
        "psh_flag_count": 0.0,
        "ack_flag_count": 0.0,
        "average_packet_size": 64.0,
        "subflow_fwd_bytes": 50000.0,
        "init_win_bytes_forward": 1024.0,
        "init_win_bytes_backward": 0.0,
        "act_data_pkt_fwd": 1000.0,
        "min_seg_size_forward": 20.0,
        "active_mean": 5000.0,
        "active_max": 5000.0,
        "active_min": 5000.0,
        "idle_mean": 0.0,
        "idle_max": 0.0,
        "idle_min": 0.0
    }
    return ddos_data

def test_single_prediction(model_name="Random Forest"):
    """Test single prediction with one model"""
    sample_data = create_sample_data()
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/{model_name}",
            json=sample_data
        )
        print(f"\n=== Single Prediction ({model_name}) ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("All Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Single prediction failed: {e}")
        return False

def test_all_models_prediction():
    """Test prediction with all models"""
    ddos_data = create_ddos_sample()
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/all",
            json=ddos_data
        )
        print("\n=== All Models Prediction (DDoS Sample) ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            for result in results:
                print(f"\n{result['model_name']}:")
                print(f"  Predicted Class: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"All models prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    batch_data = [
        create_sample_data(),
        create_ddos_sample(),
        create_sample_data()
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch/Random Forest",
            json=batch_data
        )
        print("\n=== Batch Prediction ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Number of predictions: {len(result['predictions'])}")
            print("Summary:")
            for class_name, count in result['summary'].items():
                print(f"  {class_name}: {count}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("Starting API Tests...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_api_health),
        ("List Models", test_list_models),
        ("Single Prediction", test_single_prediction),
        ("All Models Prediction", test_all_models_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nTest '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

if __name__ == "__main__":
    run_all_tests()
