#!/usr/bin/env bash
set -e

# Edit these two lines:
REGISTRY_USER="your-github-username"
IMAGE_NAME="track1-agent"

IMAGE="ghcr.io/${REGISTRY_USER}/${IMAGE_NAME}:latest"

# Must build for linux/amd64 -- the judging VM will fail to pull an
# Apple-Silicon-only image (PULL_ERROR).
docker buildx build --platform linux/amd64 --tag "$IMAGE" --push .

echo ""
echo "Pushed: $IMAGE"
echo "IMPORTANT: on GitHub -> Packages -> ${IMAGE_NAME} -> Package settings,"
echo "make sure visibility is set to PUBLIC, or the judging pull will fail."
