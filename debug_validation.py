import sys
sys.path.insert(0, 'src')

from teer_cdss.geometric_features import validate_geometric_features

# Create a mock features dict to test
features = {
    'signed_distances': [[1, 2, 3], [4, 5, 6]],
    'unsigned_distances': [[1, 2, 3], [4, 5, 6]], 
    'laplace_beltrami_spectrum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

print("Testing valid features:")
result = validate_geometric_features(features)
print(f"Result: {result}")

# Test missing key
invalid_features = features.copy()
del invalid_features['signed_distances']
print("\nTesting invalid features (missing signed_distances):")
result2 = validate_geometric_features(invalid_features)
print(f"Result: {result2}")