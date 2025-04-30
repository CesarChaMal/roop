#!/bin/bash

run_face_swap() {
  local description="$1"
  shift
  local log_file="logs/$(date +%F_%T)_${description// /_}.log"
  echo "[🔁] Running: $description..."
  python3 run.py "$@" | tee "$log_file"
}

mkdir -p logs

while true; do
  echo ""
  echo "🟢 Choose a face swap mode to run:"
  echo "1) Face Swap - Image (HQ with Enhancer)"
  echo "2) Face Swap - Image (Fast, no Enhancer)"
  echo "3) Face Swap - Video (HQ)"
  echo "4) Face Swap - Multi-face Video (custom reference)"
  echo "5) Face Swap - Compressed Video (HEVC, lower quality)"
  echo "6) Face Swap - NVENC Video (fast encoding)"
  echo "7) Face Swap - Debug (keep temp frames, no audio)"
  echo "8) Face Swap - Minimal Example (manual testing)"
  echo "9) Face Swap - Image (2 source faces)"
  echo "10) Face Swap - Video (2 source faces)"
  echo "11) Face Swap - Video (2 source faces onto 3 target faces, ref by position 0)"
  echo "12) Face Swap - Video (2 source faces onto 3 target faces, ref by position 1)"
  echo "13) Face Swap - Video (3 source faces onto 3 target faces, ref by position 0)"
  echo "14) Face Swap - Video (3 source faces onto 3 target faces, ref by position 1)"
  echo "0) ❌ Exit"

  read -p "👉 Enter number [0-14]: " choice

  case $choice in
    1) run_face_swap "Image (HQ with Enhancer)" \
        --target content/target_image.png \
        --source content/source_image.png \
        --output content/output_image.png \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer ;;
    2) run_face_swap "Image (Fast, no Enhancer)" \
        --target content/target_image.png \
        --source content/source_image.png \
        --output content/output_image_fast.png \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper ;;
    3) run_face_swap "Video (HQ)" \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise ;;
    4) run_face_swap "Multi-face Video (custom reference)" \
        --target content/target_multiface_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_multiface.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --reference-face-position 0 \
        --reference-frame-number 10 ;;
    5) run_face_swap "Compressed Video (HEVC)" \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_compressed.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --output-video-encoder libx265 \
        --output-video-quality 40 \
        --keep-fps \
        --framewise ;;
    6) run_face_swap "NVENC Video" \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_nvenc.mp4 \
        --execution-provider cuda \
        --frame-processor face_swapper face_enhancer \
        --output-video-encoder h264_nvenc \
        --output-video-quality 30 \
        --keep-fps \
        --framewise ;;
    7) run_face_swap "Debug Video (keep frames, skip audio)" \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_debug.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --keep-fps \
        --framewise \
        --keep-frames \
        --skip-audio ;;
    8) run_face_swap "Minimal Example (manual testing)" \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_test.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper \
        --keep-fps \
        --framewise ;;
    9) run_face_swap "Image (2 source faces)" \
        --target content/target_multiface_image.png \
        --source "content/source_image1.png;content/source_image1.png" \
        --output content/output_image_multiface.png \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --many-faces \
        --multi-source ;;
    10) run_face_swap "Video (2 source faces)" \
        --target content/target_multiface_video.mp4 \
        --source "content/source_image1.png;content/source_image1.png" \
        --output content/output_video_multifaces.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source ;;
    11) run_face_swap "Video (2→3 targets, ref=0)" \
        --target content/target_3faces_video.mp4 \
        --source "content/source_image1.png;content/source_image2.png" \
        --output content/output_video_multifaces_2sources_3targets_byPosition0.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 0 \
        --reference-frame-number 0 ;;
    12) run_face_swap "Video (2→3 targets, ref=1)" \
        --target content/target_3faces_video.mp4 \
        --source "content/source_image1.png;content/source_image2.png" \
        --output content/output_video_multifaces_2sources_3targets_byPosition1.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 1 \
        --reference-frame-number 0 ;;
    13) run_face_swap "Video (3→3 targets, ref=0)" \
        --target content/target_3faces_video.mp4 \
        --source "content/source_image1.png;content/source_image2.png;content/source_image3.png" \
        --output content/output_video_multifaces_3sources_3targets_byPosition0.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 0 \
        --reference-frame-number 0 ;;
    14) run_face_swap "Video (3→3 targets, ref=1)" \
        --target content/target_3faces_video.mp4 \
        --source "content/source_image1.png;content/source_image2.png;content/source_image3.png" \
        --output content/output_video_multifaces_3sources_3targets_byPosition1.mp4 \
        --execution-provider "$EXECUTION_PROVIDER" \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 1 \
        --reference-frame-number 0 ;;
    0) echo "[👋] Exiting. Have a nice day!"; break ;;
    *) echo "[❌] Invalid option. Please enter a number between 0 and 14." ;;
  esac

  echo ""
  read -p "🔁 Press [Enter] to continue or [Ctrl+C] to exit..."
  clear
done
