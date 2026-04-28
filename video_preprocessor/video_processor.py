import cv2
import os

def process_video(input_path, output_path, roi):
    """
    Process a video file by extracting the ROI from each frame and saving a compressed video.

    :param input_path: Path to the input video file
    :param output_path: Path to save the processed video
    :param roi: Tuple (x, y, w, h) defining the region of interest
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define ROI
    x, y, w, h = roi
    if x + w > width or y + h > height or x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("Invalid ROI coordinates")

    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI
        roi_frame = frame[y:y+h, x:x+w]

        # Write to output
        out.write(roi_frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Processed {frame_count} frames. Saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    process_video("input.mp4", "output.mp4", (100, 100, 200, 200))