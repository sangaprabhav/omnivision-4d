"""
Phase 2 Validation Script - Tests validation and consistency features
"""

import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("Phase 2: Validation & Consistency - Tests")
print("=" * 60)

# Test 1: Multi-Modal Consistency Checker
print("\n[Test 1/6] Multi-Modal Consistency Checker")
try:
    from src.core.multimodal_consistency import MultiModalConsistencyChecker

    checker = MultiModalConsistencyChecker()

    # Create test mask
    mask = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask.astype(np.uint8), (50, 50), 20, 1, -1)
    mask = mask.astype(bool)

    # Create test depth map
    depth = np.ones((100, 100)) * 2.5  # 2.5 meters uniform depth

    # Create test Cosmos bbox
    cosmos_bbox = {"x": 30, "y": 30, "w": 40, "h": 40}

    # Check consistency
    metrics = checker.check_consistency(mask, depth, cosmos_bbox, frame_shape=(100, 100))

    assert 0.0 <= metrics.overall_consistency <= 1.0, "Consistency should be in [0, 1]"
    assert isinstance(metrics.is_consistent, bool), "Should return bool"

    print(f"  ✓ Spatial IoU: {metrics.spatial_iou:.3f}")
    print(f"  ✓ Centroid distance: {metrics.centroid_distance:.3f}")
    print(f"  ✓ Depth consistency: {metrics.depth_consistency:.3f}")
    print(f"  ✓ Overall consistency: {metrics.overall_consistency:.3f}")
    print(f"  ✓ Is consistent: {metrics.is_consistent}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Cross-Modal Confidence Computation
print("\n[Test 2/6] Cross-Modal Confidence")
try:
    from src.core.multimodal_consistency import compute_cross_modal_confidence

    # High confidence from all models + high consistency
    conf_high = compute_cross_modal_confidence(0.9, 0.85, 0.8, 0.9)

    # Low confidence from all models + low consistency
    conf_low = compute_cross_modal_confidence(0.3, 0.4, 0.35, 0.2)

    assert conf_high > conf_low, "High inputs should give higher confidence"
    assert 0.0 <= conf_high <= 1.0, "Confidence should be in [0, 1]"
    assert 0.0 <= conf_low <= 1.0, "Confidence should be in [0, 1]"

    print(f"  ✓ High confidence case: {conf_high:.3f}")
    print(f"  ✓ Low confidence case: {conf_low:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 3: Physics Validator
print("\n[Test 3/6] Physics Validator")
try:
    from src.core.physics_validator import PhysicsValidator, PHYSICS_CONSTRAINTS_LIBRARY

    validator = PhysicsValidator()

    # Create test trajectory
    trajectory = [
        {"x": 0.0, "y": 0.0, "z": 1.0, "timestamp": 0.0, "vx": 0, "vy": 0, "vz": 0},
        {"x": 1.0, "y": 0.0, "z": 1.0, "timestamp": 0.1, "vx": 10, "vy": 0, "vz": 0},
        {"x": 2.0, "y": 0.0, "z": 1.0, "timestamp": 0.2, "vx": 10, "vy": 0, "vz": 0}
    ]

    # Validate for "human" class
    validated, report = validator.validate_trajectory(
        trajectory,
        object_class="human",
        units="meters"
    )

    assert isinstance(validated, list), "Should return list"
    assert isinstance(report, dict), "Should return dict"
    assert "status" in report, "Report should have status"

    print(f"  ✓ Validation status: {report['status']}")
    print(f"  ✓ Total violations: {report.get('total_violations', 0)}")
    print(f"  ✓ Frames corrected: {report.get('frames_corrected', 0)}")
    print(f"  ✓ Constraints for 'human': max_speed={PHYSICS_CONSTRAINTS_LIBRARY['human'].max_speed} m/s")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Motion Type Classification
print("\n[Test 4/6] Motion Type Classification")
try:
    from src.core.physics_validator import classify_motion_type

    # Stationary trajectory
    traj_stationary = [
        {"x": 10.0, "y": 10.0, "z": 1.0},
        {"x": 10.0, "y": 10.0, "z": 1.0},
        {"x": 10.0, "y": 10.0, "z": 1.0}
    ]

    # Linear trajectory
    traj_linear = [
        {"x": 0.0, "y": 0.0, "z": 1.0},
        {"x": 10.0, "y": 0.0, "z": 1.0},
        {"x": 20.0, "y": 0.0, "z": 1.0}
    ]

    # Erratic trajectory
    traj_erratic = [
        {"x": 0.0, "y": 0.0, "z": 1.0},
        {"x": 5.0, "y": 10.0, "z": 1.0},
        {"x": 2.0, "y": 3.0, "z": 1.0},
        {"x": 15.0, "y": 1.0, "z": 1.0}
    ]

    motion_stationary = classify_motion_type(traj_stationary)
    motion_linear = classify_motion_type(traj_linear)
    motion_erratic = classify_motion_type(traj_erratic)

    print(f"  ✓ Stationary trajectory: {motion_stationary}")
    print(f"  ✓ Linear trajectory: {motion_linear}")
    print(f"  ✓ Erratic trajectory: {motion_erratic}")

    assert motion_stationary == "stationary", "Should classify as stationary"
    assert motion_linear == "linear", "Should classify as linear"

    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 5: Semantic-Geometric Matcher
print("\n[Test 5/6] Semantic-Geometric Matcher")
try:
    from src.core.semantic_matcher import SemanticGeometricMatcher

    matcher = SemanticGeometricMatcher()

    # Create test SAM mask
    mask = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask.astype(np.uint8), (50, 50), 20, 1, -1)
    mask = mask.astype(bool)

    # Create test Cosmos objects
    cosmos_objects = [
        {
            "label": "robot",
            "bbox": {"x": 30, "y": 30, "w": 40, "h": 40},
            "confidence": 0.9
        }
    ]

    # Match
    matches = matcher.match_objects_to_labels(
        {1: mask},
        cosmos_objects,
        frame_shape=(100, 100)
    )

    assert len(matches) > 0, "Should produce at least one match"
    assert hasattr(matches[0], 'semantic_label'), "Match should have label"
    assert 0.0 <= matches[0].confidence <= 1.0, "Confidence should be in [0, 1]"

    print(f"  ✓ Matched label: {matches[0].semantic_label}")
    print(f"  ✓ Match confidence: {matches[0].confidence:.3f}")
    print(f"  ✓ Spatial IoU: {matches[0].spatial_iou:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Hierarchical Label Matching
print("\n[Test 6/6] Hierarchical Label Matching")
try:
    from src.core.semantic_matcher import SemanticGeometricMatcher

    matcher = SemanticGeometricMatcher(enable_hierarchical=True)

    # Test parent label finding
    parent = matcher._find_parent_label("robot_gripper")

    assert parent == "robot", f"Should find parent 'robot', got '{parent}'"

    print(f"  ✓ robot_gripper → {parent}")
    print(f"  ✓ Hierarchy enabled: {matcher.enable_hierarchical}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\n" + "=" * 60)
print("✅ All Phase 2 validation tests completed!")
print("=" * 60)
print("\nPhase 2 Features:")
print("  ✓ Multi-modal consistency checking")
print("  ✓ Cross-modal confidence computation")
print("  ✓ Physics validation with class-specific constraints")
print("  ✓ Motion type classification")
print("  ✓ Semantic-geometric matching")
print("  ✓ Hierarchical label support")
print("\nNext: Run with annotator and create documentation")
print("=" * 60)
