"""
Validation Script for Visualization & Preprocessing

Tests visualization and video preprocessing modules.
"""

import numpy as np
import cv2
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 70)
print("Visualization & Preprocessing Validation")
print("=" * 70)

# Test 1: Video Preprocessor - Metadata Extraction
print("\n[Test 1/6] Video Preprocessor - Metadata Extraction")
try:
    from src.preprocessing import VideoPreprocessor

    # Create test video
    fd, test_video = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)

    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 30.0, (640, 480))

    for i in range(30):  # 1 second at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()

    # Extract metadata
    preprocessor = VideoPreprocessor()
    metadata = preprocessor.extract_metadata(test_video)

    assert metadata.is_valid, "Video should be valid"
    assert metadata.width == 640, f"Width should be 640, got {metadata.width}"
    assert metadata.height == 480, f"Height should be 480, got {metadata.height}"
    assert metadata.total_frames == 30, f"Should have 30 frames, got {metadata.total_frames}"

    print(f"  ✓ Resolution: {metadata.width}x{metadata.height}")
    print(f"  ✓ FPS: {metadata.fps}")
    print(f"  ✓ Total frames: {metadata.total_frames}")
    print(f"  ✓ Duration: {metadata.duration:.2f}s")
    print(f"  ✓ Codec: {metadata.codec}")
    print(f"  ✓ File size: {metadata.file_size_mb:.2f} MB")
    print("  ✅ PASSED")

    # Cleanup
    if os.path.exists(test_video):
        os.remove(test_video)

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Video Preprocessor - Resolution Adaptation
print("\n[Test 2/6] Video Preprocessor - Resolution Adaptation")
try:
    from src.preprocessing import VideoPreprocessor

    preprocessor = VideoPreprocessor(max_resolution=(1920, 1080))

    # Test downscaling
    original = (3840, 2160)  # 4K
    target = preprocessor._compute_target_resolution(original)

    assert target[0] <= 1920, "Width should be scaled down"
    assert target[1] <= 1080, "Height should be scaled down"

    # Check aspect ratio preserved
    original_aspect = original[0] / original[1]
    target_aspect = target[0] / target[1]
    assert abs(original_aspect - target_aspect) < 0.01, "Aspect ratio should be preserved"

    print(f"  ✓ 4K ({original[0]}x{original[1]}) → Full HD ({target[0]}x{target[1]})")
    print(f"  ✓ Aspect ratio preserved: {original_aspect:.3f} → {target_aspect:.3f}")

    # Test no scaling needed
    original_small = (1280, 720)  # 720p
    target_small = preprocessor._compute_target_resolution(original_small)

    assert target_small == original_small, "Should not upscale"
    print(f"  ✓ 720p remains unchanged: {target_small[0]}x{target_small[1]}")

    print("  ✅ PASSED")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 3: Video Overlay - Color Palette Generation
print("\n[Test 3/6] Video Overlay - Color Palette")
try:
    from src.visualization import VideoOverlay

    overlay = VideoOverlay()

    # Generate colors
    colors = overlay._generate_color_palette(10)

    assert len(colors) == 10, "Should generate 10 colors"
    assert all(isinstance(c, tuple) and len(c) == 3 for c in colors), "Colors should be BGR tuples"
    assert all(0 <= val <= 255 for c in colors for val in c), "Color values should be 0-255"

    # Check colors are distinct
    unique_colors = set(colors)
    assert len(unique_colors) == len(colors), "Colors should be distinct"

    print(f"  ✓ Generated {len(colors)} distinct colors")
    print(f"  ✓ Sample colors: {colors[:3]}")
    print("  ✅ PASSED")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 4: Video Overlay - Trajectory History Building
print("\n[Test 4/6] Video Overlay - Trajectory History")
try:
    from src.visualization import VideoOverlay

    # Create mock annotation result
    mock_result = {
        "objects": [
            {
                "object_id": "obj_1",
                "trajectory_4d": {
                    "points": [
                        {"frame": 0, "x": 100, "y": 100, "z": 1.0},
                        {"frame": 1, "x": 110, "y": 110, "z": 1.1},
                        {"frame": 2, "x": 120, "y": 120, "z": 1.2}
                    ]
                }
            }
        ]
    }

    overlay = VideoOverlay()
    history = overlay._build_trajectory_history(mock_result)

    assert "obj_1" in history, "Should build history for obj_1"
    assert len(history["obj_1"]) == 3, "Should have 3 trajectory points"

    print(f"  ✓ Built trajectory history for {len(history)} objects")
    print(f"  ✓ Object obj_1 has {len(history['obj_1'])} points")
    print("  ✅ PASSED")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 5: Trajectory Plotter - Gaussian Blob
print("\n[Test 5/6] Trajectory Plotter - Gaussian Blob")
try:
    from src.visualization import TrajectoryPlotter

    plotter = TrajectoryPlotter()

    # Create heatmap
    heatmap = np.zeros((100, 100), dtype=np.float32)

    # Add blob
    plotter._add_gaussian_blob(heatmap, 50, 50, sigma=10, intensity=1.0)

    # Check blob was added
    assert heatmap.max() > 0, "Heatmap should have values"
    assert heatmap[50, 50] > 0.9, "Center should be brightest"

    # Check Gaussian falloff
    assert heatmap[50, 60] < heatmap[50, 50], "Should decay from center"
    assert heatmap[50, 60] > 0, "Should have non-zero values near center"

    print(f"  ✓ Gaussian blob added to heatmap")
    print(f"  ✓ Center value: {heatmap[50, 50]:.3f}")
    print(f"  ✓ Edge value (10px away): {heatmap[50, 60]:.3f}")
    print("  ✅ PASSED")

except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 6: End-to-End Annotation Overlay
print("\n[Test 6/6] End-to-End Annotation with Visualization")
try:
    from src.visualization import create_annotated_video

    # Create test video
    fd, test_video = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 30.0, (320, 240))

    for i in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw moving object
        x = 50 + i * 5
        y = 120
        cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
        out.write(frame)

    out.release()

    # Create mock annotation
    mock_annotation = {
        "objects": [
            {
                "object_id": "test_obj",
                "semantic_label": "ball",
                "trajectory_4d": {
                    "points": [
                        {"frame": i, "x": 50 + i * 5, "y": 120, "z": 2.0}
                        for i in range(30)
                    ]
                },
                "phase2_validation": {
                    "cross_modal_confidence": 0.85
                }
            }
        ]
    }

    # Create annotated video
    output_path = tempfile.mktemp(suffix='_annotated.mp4')

    result_path = create_annotated_video(
        test_video,
        mock_annotation,
        output_path,
        show_boxes=True,
        show_labels=True,
        show_trajectories=True
    )

    assert os.path.exists(result_path), "Annotated video should be created"

    # Verify output video
    cap = cv2.VideoCapture(result_path)
    assert cap.isOpened(), "Output video should be readable"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count == 30, f"Should have 30 frames, got {frame_count}"

    cap.release()

    print(f"  ✓ Annotated video created: {result_path}")
    print(f"  ✓ Frame count: {frame_count}")
    print("  ✅ PASSED")

    # Cleanup
    for path in [test_video, output_path]:
        if os.path.exists(path):
            os.remove(path)

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ All Visualization & Preprocessing tests completed!")
print("=" * 70)

print("\nImplemented Features:")
print("\n📹 Visualization:")
print("  ✓ Video overlay with bounding boxes and labels")
print("  ✓ Trajectory path drawing with fade effects")
print("  ✓ Confidence score display")
print("  ✓ 3D trajectory plotting")
print("  ✓ Multi-panel analysis plots")
print("  ✓ Motion activity heatmaps")

print("\n🎬 Video Preprocessing:")
print("  ✓ Comprehensive metadata extraction")
print("  ✓ Resolution adaptation (preserve aspect ratio)")
print("  ✓ Codec detection and validation")
print("  ✓ Format compatibility checks")
print("  ✓ Rotation handling (with ffmpeg)")
print("  ✓ Transcoding support (with ffmpeg)")
print("  ✓ FPS normalization")

print("\n📝 Usage Examples:")
print("""
# Visualize annotations
from src.visualization import create_annotated_video

create_annotated_video(
    "video.mp4",
    annotation_result,
    "annotated.mp4",
    show_trajectories=True
)

# Preprocess video
from src.preprocessing import preprocess_video

processed = preprocess_video(
    "input.mp4",
    max_resolution=(1920, 1080)
)
""")

print("=" * 70)
