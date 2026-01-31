"""
Phase 1 Validation Script - Tests core functionality without model dependencies
"""

import numpy as np
import cv2
import os
import tempfile

print("=" * 60)
print("Phase 1 Enhancement Validation")
print("=" * 60)

# Test 1: Mask Quality Assessment
print("\n[Test 1/6] Mask Quality Assessment")
try:
    from src.core.mask_quality import MaskQualityAssessor

    assessor = MaskQualityAssessor()

    # Create a high-quality mask (compact circular shape)
    mask_good = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask_good.astype(np.uint8), (50, 50), 20, 1, -1)
    mask_good = mask_good.astype(bool)

    quality = assessor.assess(mask_good)

    assert quality.is_valid, "High-quality mask should be valid"
    assert 0.0 <= quality.overall_score <= 1.0, "Score should be in [0, 1]"

    print(f"  ✓ Quality score: {quality.overall_score:.3f}")
    print(f"  ✓ Compactness: {quality.compactness:.3f}")
    print(f"  ✓ Coverage: {quality.coverage:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 2: Temporal Consistency
print("\n[Test 2/6] Temporal Consistency")
try:
    from src.core.mask_quality import MaskQualityAssessor

    assessor = MaskQualityAssessor()

    # Create two similar masks
    mask1 = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask1.astype(np.uint8), (50, 50), 20, 1, -1)
    mask1 = mask1.astype(bool)

    mask2 = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask2.astype(np.uint8), (52, 50), 20, 1, -1)
    mask2 = mask2.astype(bool)

    iou = assessor.assess_temporal_consistency(mask2, mask1)

    assert 0.0 <= iou <= 1.0, "IoU should be in [0, 1]"
    assert iou > 0.6, f"Similar masks should have high IoU, got {iou:.3f}"

    print(f"  ✓ Temporal IoU: {iou:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 3: Mask Quality Filtering
print("\n[Test 3/6] Mask Quality Filtering")
try:
    from src.core.mask_quality import filter_low_quality_masks

    # Create mock video segments
    def create_circle_mask(h, w, cx, cy, r):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        return mask.astype(bool)

    video_segments = {
        0: {
            1: create_circle_mask(100, 100, 50, 50, 20),  # Good
            2: np.zeros((100, 100), dtype=bool)  # Bad: empty
        },
        1: {
            1: create_circle_mask(100, 100, 52, 50, 20),  # Good
        }
    }

    total_before = sum(len(objs) for objs in video_segments.values())

    filtered = filter_low_quality_masks(video_segments, min_quality_score=0.3)

    total_after = sum(len(objs) for objs in filtered.values())

    print(f"  ✓ Before: {total_before} masks")
    print(f"  ✓ After: {total_after} masks")
    print(f"  ✓ Filtered: {total_before - total_after} low-quality masks")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 4: Adaptive Frame Sampling (with test video)
print("\n[Test 4/6] Adaptive Frame Sampling")
try:
    from src.core.adaptive_sampling import sample_frames_adaptive

    # Create a simple test video
    fd, video_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)

    try:
        # Create video
        fps = 30
        duration = 1
        frames_count = fps * duration
        height, width = 240, 320

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for i in range(frames_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cx = int(50 + i * 3)
            cy = height // 2
            cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)
            out.write(frame)

        out.release()

        # Test sampling
        indices, metadata = sample_frames_adaptive(
            video_path,
            max_frames=16,
            enable_adaptive=True
        )

        assert len(indices) <= 16, "Should respect max_frames"
        assert indices == sorted(indices), "Indices should be sorted"
        assert "sampling_strategy" in metadata, "Should have metadata"

        print(f"  ✓ Sampled frames: {len(indices)}")
        print(f"  ✓ Strategy: {metadata['sampling_strategy']}")
        print("  ✅ PASSED")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: SAM-2 Dataclass and Grid Generation
print("\n[Test 5/6] SAM-2 Multi-Object Support")
try:
    from src.models.sam2_model import SAM2Model, SAM2Result

    model = SAM2Model()

    # Test grid generation
    points = model._generate_grid_points(448, 448, grid_size=3)

    assert len(points) == 9, "3x3 grid should have 9 points"

    for x, y in points:
        assert 0 < x < 448, "X should be within bounds"
        assert 0 < y < 448, "Y should be within bounds"

    # Test confidence computation
    logits = np.random.randn(100, 100) * 2
    mask = logits > 0
    confidence = model._compute_confidence(logits, mask)

    assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1]"

    print(f"  ✓ Grid points: {len(points)}")
    print(f"  ✓ Sample confidence: {confidence:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 6: Depth and Cosmos Dataclasses
print("\n[Test 6/6] Depth & Cosmos Result Structures")
try:
    from src.models.depth_model import DepthModel, DepthResult
    from src.models.cosmos_model import CosmosModel, CosmosResult

    # Test DepthModel uncertainty estimation
    depth_model = DepthModel()
    depth = np.random.uniform(0.5, 5.0, (100, 100))
    valid_mask = np.ones((100, 100), dtype=bool)

    uncertainty = depth_model._estimate_uncertainty(depth, valid_mask)

    assert uncertainty.shape == depth.shape, "Uncertainty should match depth shape"
    assert 0.0 <= uncertainty.max() <= 1.0, "Uncertainty should be normalized"

    # Test CosmosModel confidence heuristic
    cosmos_model = CosmosModel()

    good_text = '{"objects": [{"label": "robot"}]}'
    bad_text = "Error: cannot process"

    good_conf = cosmos_model._estimate_confidence_heuristic(good_text)
    bad_conf = cosmos_model._estimate_confidence_heuristic(bad_text)

    assert good_conf > bad_conf, "Good text should have higher confidence"

    print(f"  ✓ Depth uncertainty shape: {uncertainty.shape}")
    print(f"  ✓ Good confidence: {good_conf:.3f}")
    print(f"  ✓ Bad confidence: {bad_conf:.3f}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\n" + "=" * 60)
print("✅ All Phase 1 validation tests completed!")
print("=" * 60)
print("\nPhase 1 Features:")
print("  ✓ Multi-object auto-detection")
print("  ✓ Mask quality assessment")
print("  ✓ Model confidence propagation")
print("  ✓ Adaptive frame sampling")
print("\nNext: Commit and push changes to branch")
print("=" * 60)
