"""
Test suite for Phase 1 precision enhancements

Tests:
1. Multi-object auto-detection
2. Mask quality assessment
3. Model confidence propagation
4. Adaptive frame sampling
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

# Test mask quality assessment
def test_mask_quality_assessment():
    """Test mask quality metrics computation"""
    from src.core.mask_quality import MaskQualityAssessor

    assessor = MaskQualityAssessor()

    # Create a high-quality mask (compact circular shape)
    mask_good = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask_good.astype(np.uint8), (50, 50), 20, 1, -1)
    mask_good = mask_good.astype(bool)

    quality_good = assessor.assess(mask_good)

    assert quality_good.is_valid, "High-quality mask should be marked as valid"
    assert quality_good.compactness > 0.5, "Circle should have high compactness"
    assert quality_good.coverage > 0.0, "Mask should have non-zero coverage"

    print(f"✓ Good mask quality: {quality_good.overall_score:.3f}")

    # Create a low-quality mask (fragmented)
    mask_bad = np.zeros((100, 100), dtype=bool)
    mask_bad[10:15, 10:15] = True  # Small fragment 1
    mask_bad[80:85, 80:85] = True  # Small fragment 2

    quality_bad = assessor.assess(mask_bad)

    assert quality_bad.overall_score < quality_good.overall_score, "Fragmented mask should have lower quality"

    print(f"✓ Bad mask quality: {quality_bad.overall_score:.3f}")


def test_temporal_consistency():
    """Test temporal consistency between consecutive masks"""
    from src.core.mask_quality import MaskQualityAssessor

    assessor = MaskQualityAssessor()

    # Create two similar masks (high IoU)
    mask1 = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask1.astype(np.uint8), (50, 50), 20, 1, -1)
    mask1 = mask1.astype(bool)

    mask2 = np.zeros((100, 100), dtype=bool)
    cv2.circle(mask2.astype(np.uint8), (52, 50), 20, 1, -1)  # Slightly moved
    mask2 = mask2.astype(bool)

    iou = assessor.assess_temporal_consistency(mask2, mask1)

    assert iou > 0.7, f"Similar masks should have high IoU, got {iou:.3f}"
    print(f"✓ Temporal consistency IoU: {iou:.3f}")


def test_mask_quality_filtering():
    """Test filtering of low-quality masks"""
    from src.core.mask_quality import filter_low_quality_masks

    # Create mock video segments
    video_segments = {
        0: {
            1: create_circle_mask(100, 100, 50, 50, 20),  # Good mask
            2: create_tiny_mask(100, 100),  # Bad: too small
        },
        1: {
            1: create_circle_mask(100, 100, 52, 50, 20),  # Good mask (moved)
            2: create_tiny_mask(100, 100),  # Bad: too small
        }
    }

    filtered = filter_low_quality_masks(
        video_segments,
        min_quality_score=0.4,
        min_temporal_iou=0.5
    )

    assert 1 in filtered[0], "High-quality object should be retained"
    assert 2 not in filtered[0] or len(filtered[0]) == 1, "Low-quality object should be filtered"

    print(f"✓ Filtered {sum(len(objs) for objs in video_segments.values()) - sum(len(objs) for objs in filtered.values())} low-quality masks")


def test_adaptive_frame_sampling():
    """Test adaptive frame sampling with motion detection"""
    from src.core.adaptive_sampling import AdaptiveFrameSampler

    # Create a test video with motion
    video_path = create_test_video_with_motion()

    try:
        sampler = AdaptiveFrameSampler()

        # Test adaptive sampling
        indices, metadata = sampler.sample_frames(
            video_path,
            max_frames=32,
            uniform_fallback=True
        )

        assert len(indices) <= 32, "Should not exceed max_frames"
        assert len(indices) >= 8, "Should sample at least min frames"
        assert indices == sorted(indices), "Frame indices should be sorted"
        assert metadata["sampling_strategy"] in ["adaptive_motion_based", "uniform_fallback"], \
            "Should have valid sampling strategy"

        print(f"✓ Adaptive sampling: {len(indices)} frames selected ({metadata['sampling_strategy']})")

    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)


def test_sam2_confidence_computation():
    """Test SAM-2 confidence score computation"""
    from src.models.sam2_model import SAM2Model

    model = SAM2Model()

    # Create mock logits and mask
    logits = np.random.randn(100, 100) * 2  # Higher magnitude = more confident
    mask = logits > 0

    confidence = model._compute_confidence(logits, mask)

    assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1]"
    assert confidence > 0.5, "High-magnitude logits should give high confidence"

    print(f"✓ SAM-2 confidence: {confidence:.3f}")


def test_depth_uncertainty_estimation():
    """Test depth uncertainty estimation"""
    from src.models.depth_model import DepthModel

    model = DepthModel()

    # Create mock depth map
    depth = np.random.uniform(0.5, 5.0, (100, 100))
    valid_mask = np.ones((100, 100), dtype=bool)

    # Add some edges (high gradient = high uncertainty)
    depth[45:55, :] += 2.0  # Step edge

    uncertainty = model._estimate_uncertainty(depth, valid_mask)

    assert uncertainty.shape == depth.shape, "Uncertainty should match depth shape"
    assert 0.0 <= uncertainty.max() <= 1.0, "Uncertainty should be normalized to [0, 1]"

    # Edge region should have higher uncertainty
    edge_uncertainty = uncertainty[48:52, 50]
    flat_uncertainty = uncertainty[20:30, 50]

    assert edge_uncertainty.mean() > flat_uncertainty.mean(), \
        "Edge regions should have higher uncertainty"

    print(f"✓ Depth uncertainty: edge={edge_uncertainty.mean():.3f}, flat={flat_uncertainty.mean():.3f}")


def test_cosmos_confidence_heuristic():
    """Test Cosmos confidence estimation from text"""
    from src.models.cosmos_model import CosmosModel

    model = CosmosModel()

    # Good output (contains JSON and relevant terms)
    good_output = """
    {
        "objects": [
            {"label": "robot", "position": [100, 200], "movement": "forward"}
        ],
        "trajectories": []
    }
    """

    good_confidence = model._estimate_confidence_heuristic(good_output)

    # Bad output (no JSON, has error)
    bad_output = "I'm sorry, I cannot detect any objects in this video."

    bad_confidence = model._estimate_confidence_heuristic(bad_output)

    assert good_confidence > bad_confidence, \
        f"Good output should have higher confidence: {good_confidence:.3f} vs {bad_confidence:.3f}"

    print(f"✓ Cosmos confidence: good={good_confidence:.3f}, bad={bad_confidence:.3f}")


def test_multi_object_grid_generation():
    """Test grid-based multi-object initialization points"""
    from src.models.sam2_model import SAM2Model

    model = SAM2Model()

    # Generate 3x3 grid
    points = model._generate_grid_points(448, 448, grid_size=3)

    assert len(points) == 9, "3x3 grid should have 9 points"

    # Check that points are within frame (with margins)
    for x, y in points:
        assert 0 < x < 448, f"X coordinate {x} should be within frame"
        assert 0 < y < 448, f"Y coordinate {y} should be within frame"

    print(f"✓ Grid generation: {len(points)} points")
    print(f"  Sample points: {points[:3]}")


def test_nms_duplicate_removal():
    """Test Non-Maximum Suppression for removing duplicate detections"""
    from src.models.sam2_model import SAM2Model

    model = SAM2Model()

    # Create overlapping masks
    mask1 = create_circle_mask(100, 100, 50, 50, 20)
    mask2 = create_circle_mask(100, 100, 52, 50, 20)  # Highly overlapping
    mask3 = create_circle_mask(100, 100, 80, 80, 15)  # Different location

    video_segments = {
        0: {1: mask1, 2: mask2, 3: mask3}
    }

    confidence_scores = {
        0: {1: 0.9, 2: 0.7, 3: 0.8}  # mask1 has highest confidence
    }

    iou_predictions = {
        0: {1: 0.9, 2: 0.7, 3: 0.8}
    }

    filtered_segments, filtered_conf, filtered_iou = model._apply_nms(
        video_segments,
        confidence_scores,
        iou_predictions,
        iou_threshold=0.7
    )

    # Should keep mask1 (highest confidence) and mask3 (different location)
    # Should remove mask2 (overlaps with mask1 and has lower confidence)
    assert len(filtered_segments[0]) <= 2, "Should remove overlapping duplicates"
    assert 1 in filtered_segments[0], "Highest confidence mask should be kept"

    print(f"✓ NMS: {len(video_segments[0])} → {len(filtered_segments[0])} objects")


# Helper functions
def create_circle_mask(height, width, cx, cy, radius):
    """Create a circular mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask.astype(bool)


def create_tiny_mask(height, width):
    """Create a very small mask (should be filtered)"""
    mask = np.zeros((height, width), dtype=bool)
    mask[50:53, 50:53] = True  # Only 9 pixels
    return mask


def create_test_video_with_motion():
    """Create a test video with moving object"""
    import tempfile

    # Create temporary video file
    fd, video_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)

    # Video parameters
    fps = 30
    duration = 2  # seconds
    frames_count = fps * duration
    height, width = 240, 320

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Generate frames with moving object
    for i in range(frames_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Moving circle
        cx = int(50 + i * 3)  # Move right
        cy = height // 2

        cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)

        out.write(frame)

    out.release()

    return video_path


if __name__ == "__main__":
    """Run tests manually"""
    print("=" * 60)
    print("Phase 1 Enhancement Tests")
    print("=" * 60)

    try:
        print("\n[1/9] Testing mask quality assessment...")
        test_mask_quality_assessment()

        print("\n[2/9] Testing temporal consistency...")
        test_temporal_consistency()

        print("\n[3/9] Testing mask quality filtering...")
        test_mask_quality_filtering()

        print("\n[4/9] Testing adaptive frame sampling...")
        test_adaptive_frame_sampling()

        print("\n[5/9] Testing SAM-2 confidence computation...")
        test_sam2_confidence_computation()

        print("\n[6/9] Testing depth uncertainty estimation...")
        test_depth_uncertainty_estimation()

        print("\n[7/9] Testing Cosmos confidence heuristic...")
        test_cosmos_confidence_heuristic()

        print("\n[8/9] Testing multi-object grid generation...")
        test_multi_object_grid_generation()

        print("\n[9/9] Testing NMS duplicate removal...")
        test_nms_duplicate_removal()

        print("\n" + "=" * 60)
        print("✅ All Phase 1 tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
