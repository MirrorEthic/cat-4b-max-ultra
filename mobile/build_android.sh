#!/usr/bin/env bash
# Stage 1 of the mobile port (docs/MOBILE.md): cross-compile llama.cpp for the Galaxy Z Fold 6
# (Snapdragon 8 Gen 3, arm64-v8a), CPU backend only. Uses the same pinned llama.cpp checkout the
# release GGUFs were made with (b11139). Flags follow its docs/android.md; GGML_CPU_ARM_ARCH is
# set for the 8 Gen 3's cores (Cortex-X4/A720/A520: dotprod, i8mm, fp16 — no SVE), since the Fold 6
# is the only target. Stage 2 (Hexagon NPU) uses scripts/snapdragon/build.py in the same checkout
# instead (Docker + Qualcomm toolchain images); this script is CPU-only on purpose.
#
#   bash mobile/build_android.sh
set -euo pipefail
NDK=${ANDROID_NDK:-$HOME/android-sdk/ndk/27.2.12479018}
L=/media/garret-sutherland/dev-data/cat4b/tools/llama.cpp-b11139
B=$L/build-android-arm64
[ -d "$NDK" ] || { echo "NDK not found at $NDK"; exit 1; }

cmake -S "$L" -B "$B" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.7-a+dotprod+i8mm+fp16 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$B" --config Release -j"$(nproc)" --target llama-cli llama-server llama-bench
echo "binaries: $B/bin/{llama-cli,llama-server,llama-bench}"

# the app bundles llama-server as a "native library" so Android extracts it executable
REPO=$(cd "$(dirname "$0")/.." && pwd)
J="$REPO/mobile/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$J" && cp "$B/bin/llama-server" "$J/libllamaserver.so"
echo "app engine: $J/libllamaserver.so (build the APK with: cd mobile/android && ./gradlew assembleDebug)"
