# Attendance ViT Pipeline

End-to-end face-recognition attendance pipeline built in a single notebook: `att-vit-pe.ipynb`.

The system enrolls identities from one image per person, expands identity coverage with generative/traditional augmentation, and marks attendance from classroom video using detection, tracking, and embedding matching.

## What This Project Does

- Detects faces with SCRFD (InsightFace).
- Aligns face crops using 5-point landmarks.
- Extracts embeddings with AdaFace ViT (with ArcFace fallback).
- Tracks people across frames with ByteTrack.
- Matches embeddings with FAISS (inner-product search).
- Applies dynamic quality-aware thresholds and short-window voting to reduce flicker.
- Exports annotated video and attendance logs.

## Notebook Scope

All pipeline logic lives in:

- `att-vit-pe.ipynb`

The notebook includes:

- Environment setup and dependency installation.
- Config for paths, thresholds, and runtime flags.
- Enrollment from identity images.
- Optional InstantID-based generative augmentation (with traditional fallback).
- Video inference and attendance generation.
- Benchmark/diagnostic metrics (Rank-1, ROC/AUC, EER, silhouette, threshold sweep).

## Architecture (High Level)

1. Enrollment
   - Read one image per person from enrollment directory.
   - Generate additional appearance variations.
   - Build an embedding index (FAISS).

2. Inference
   - Read classroom video.
   - Detect and align faces per frame.
   - Track face identities across frames.
   - Recognize with embedding similarity + dynamic threshold.
   - Apply temporal voting for stable identity assignment.

3. Export
   - Save annotated output video.
   - Save attendance summary JSON.
   - Save sample diagnostic frames.

## Requirements

Core stack used by the notebook:

- Python 3.9+
- PyTorch
- OpenCV
- InsightFace
- ONNX Runtime (GPU recommended)
- FAISS (CPU)
- Supervision + LAP (tracking)
- Transformers / timm / safetensors (ViT)
- Diffusers + Accelerate (for InstantID augmentation)
- gdown, prettytable, easydict

For exact install behavior (including Kaggle-specific compatibility handling), run the setup cells in the notebook in order.

## Quick Start

1. Open `att-vit-pe.ipynb`.
2. Update configured paths for your environment (enrollment directory, video directory, output directory).
3. Run cells from top to bottom.
4. Enroll identities from the enrollment folder.
5. Run inference on a video.
6. Review outputs in the configured output directory.

## Expected Inputs

- Enrollment images: one clear face image per person (filename stem used as identity label).
- Video: classroom video file (or Drive link if using notebook helper cell).

## Expected Outputs

Typical outputs include:

- `annotated_output.mp4`
- `attendance.json`
- `sample_frame_*.jpg`
- Optional augmentation previews under a generated visualization folder

## Configuration Notes

Key behavior is controlled by notebook config values, including:

- Similarity thresholds (base + quality adjustment).
- Voting window and minimum vote count.
- Augmentation mode (generative enabled/disabled).
- Model fallback behavior (ViT to ArcFace).

If you run locally, replace Kaggle paths with local paths before execution.

## Troubleshooting

- InsightFace install conflicts in managed environments:
  - Use the notebook’s setup strategy (includes no-dependency install workaround where needed).
- Out-of-memory during generative augmentation:
  - Disable generative mode or rely on traditional augmentation fallback.
- Recognition instability in difficult frames:
  - Increase threshold strictness and/or voting requirements.

## License

No license file is currently included. Add a project license if you plan to distribute this repository.
