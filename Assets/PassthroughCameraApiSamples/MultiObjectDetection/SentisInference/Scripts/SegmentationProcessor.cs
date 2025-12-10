// Segmentation mask processor for YOLOv11-seg
// Combines mask coefficients with prototype masks to generate final segmentation
// Supports solid fill and outline-only rendering modes

using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    public enum SegmentationRenderMode
    {
        SolidFill,
        OutlineOnly
    }

    public static class SegmentationProcessor
    {
        public static float[,] GenerateMask(float[] maskCoeffs, Tensor<float> prototypeMasks)
        {
            int maskSize = 160;
            float[,] resultMask = new float[maskSize, maskSize];

            for (int y = 0; y < maskSize; y++)
            {
                for (int x = 0; x < maskSize; x++)
                {
                    float pixelValue = 0f;

                    for (int i = 0; i < 32; i++)
                    {
                        float prototypeValue = prototypeMasks[0, i, y, x];
                        pixelValue += maskCoeffs[i] * prototypeValue;
                    }

                    resultMask[y, x] = Sigmoid(pixelValue);
                }
            }

            return resultMask;
        }

        /// <summary>
        /// Creates texture with solid fill (original behavior)
        /// </summary>
        public static Texture2D MaskToTexture(float[,] fullMask, Color maskColor, float boxCenterX, float boxCenterY, float boxWidth, float boxHeight, float imageWidth, float imageHeight)
        {
            return MaskToTextureWithMode(fullMask, maskColor, boxCenterX, boxCenterY, boxWidth, boxHeight, imageWidth, imageHeight, SegmentationRenderMode.SolidFill, 0);
        }

        /// <summary>
        /// Creates texture with specified render mode and outline width
        /// </summary>
        /// <param name="outlineWidthPixels">Width of outline in mask-space pixels (only used when mode is OutlineOnly)</param>
        public static Texture2D MaskToTextureWithMode(
            float[,] fullMask,
            Color maskColor,
            float boxCenterX,
            float boxCenterY,
            float boxWidth,
            float boxHeight,
            float imageWidth,
            float imageHeight,
            SegmentationRenderMode mode,
            int outlineWidthPixels)
        {
            int maskSize = 160;
            float scaleFactor = maskSize / imageWidth;

            float maskCenterX = boxCenterX * scaleFactor;
            float maskCenterY = boxCenterY * scaleFactor;
            float maskBoxWidth = boxWidth * scaleFactor;
            float maskBoxHeight = boxHeight * scaleFactor;

            int maskLeft = Mathf.Max(0, Mathf.FloorToInt(maskCenterX - maskBoxWidth / 2));
            int maskTop = Mathf.Max(0, Mathf.FloorToInt(maskCenterY - maskBoxHeight / 2));
            int maskRight = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterX + maskBoxWidth / 2));
            int maskBottom = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterY + maskBoxHeight / 2));

            int croppedWidth = maskRight - maskLeft;
            int croppedHeight = maskBottom - maskTop;

            if (croppedWidth <= 0 || croppedHeight <= 0) return null;

            // Pre-compute edge detection for outline mode
            bool[,] isEdge = null;
            if (mode == SegmentationRenderMode.OutlineOnly && outlineWidthPixels > 0)
            {
                isEdge = ComputeEdgeMask(fullMask, maskSize, outlineWidthPixels);
            }

            Texture2D texture = new Texture2D(croppedWidth, croppedHeight, TextureFormat.RGBA32, false);

            for (int y = 0; y < croppedHeight; y++)
            {
                for (int x = 0; x < croppedWidth; x++)
                {
                    int maskX = maskLeft + x;
                    int maskY = maskTop + y;

                    if (maskX >= maskSize || maskY >= maskSize)
                    {
                        texture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        continue;
                    }

                    float maskValue = fullMask[maskY, maskX];

                    if (maskValue > 0.5f)
                    {
                        bool shouldRender = true;

                        if (mode == SegmentationRenderMode.OutlineOnly && isEdge != null)
                        {
                            // Only render if this pixel is on the edge
                            shouldRender = isEdge[maskY, maskX];
                        }

                        if (shouldRender)
                        {
                            float alpha = maskValue * 0.8f;
                            Color pixelColor = new Color(maskColor.r, maskColor.g, maskColor.b, alpha);
                            texture.SetPixel(x, croppedHeight - 1 - y, pixelColor);
                        }
                        else
                        {
                            texture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        }
                    }
                    else
                    {
                        texture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                    }
                }
            }

            texture.Apply();
            texture.filterMode = FilterMode.Bilinear;
            return texture;
        }

        /// <summary>
        /// Computes edge mask using morphological erosion approach
        /// A pixel is an edge if it's inside the mask but within outlineWidth pixels of the boundary
        /// </summary>
        private static bool[,] ComputeEdgeMask(float[,] mask, int maskSize, int outlineWidth)
        {
            bool[,] isEdge = new bool[maskSize, maskSize];
            float threshold = 0.5f;

            for (int y = 0; y < maskSize; y++)
            {
                for (int x = 0; x < maskSize; x++)
                {
                    // Only process pixels inside the mask
                    if (mask[y, x] <= threshold)
                        continue;

                    // Check if any pixel within outlineWidth distance is outside the mask
                    bool nearBoundary = false;

                    for (int dy = -outlineWidth; dy <= outlineWidth && !nearBoundary; dy++)
                    {
                        for (int dx = -outlineWidth; dx <= outlineWidth && !nearBoundary; dx++)
                        {
                            // Skip if outside the circular kernel
                            if (dx * dx + dy * dy > outlineWidth * outlineWidth)
                                continue;

                            int nx = x + dx;
                            int ny = y + dy;

                            // Boundary of image counts as edge
                            if (nx < 0 || nx >= maskSize || ny < 0 || ny >= maskSize)
                            {
                                nearBoundary = true;
                                continue;
                            }

                            // Check if neighbor is outside mask
                            if (mask[ny, nx] <= threshold)
                            {
                                nearBoundary = true;
                            }
                        }
                    }

                    isEdge[y, x] = nearBoundary;
                }
            }

            return isEdge;
        }

        /// <summary>
        /// Generates both solid and outline textures in one pass (more efficient for distance-based switching)
        /// </summary>
        public static void GenerateBothTextures(
            float[,] fullMask,
            Color maskColor,
            float boxCenterX,
            float boxCenterY,
            float boxWidth,
            float boxHeight,
            float imageWidth,
            float imageHeight,
            int outlineWidthPixels,
            out Texture2D solidTexture,
            out Texture2D outlineTexture)
        {
            int maskSize = 160;
            float scaleFactor = maskSize / imageWidth;

            float maskCenterX = boxCenterX * scaleFactor;
            float maskCenterY = boxCenterY * scaleFactor;
            float maskBoxWidth = boxWidth * scaleFactor;
            float maskBoxHeight = boxHeight * scaleFactor;

            int maskLeft = Mathf.Max(0, Mathf.FloorToInt(maskCenterX - maskBoxWidth / 2));
            int maskTop = Mathf.Max(0, Mathf.FloorToInt(maskCenterY - maskBoxHeight / 2));
            int maskRight = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterX + maskBoxWidth / 2));
            int maskBottom = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterY + maskBoxHeight / 2));

            int croppedWidth = maskRight - maskLeft;
            int croppedHeight = maskBottom - maskTop;

            if (croppedWidth <= 0 || croppedHeight <= 0)
            {
                solidTexture = null;
                outlineTexture = null;
                return;
            }

            // Pre-compute edge detection
            bool[,] isEdge = ComputeEdgeMask(fullMask, maskSize, outlineWidthPixels);

            solidTexture = new Texture2D(croppedWidth, croppedHeight, TextureFormat.RGBA32, false);
            outlineTexture = new Texture2D(croppedWidth, croppedHeight, TextureFormat.RGBA32, false);

            for (int y = 0; y < croppedHeight; y++)
            {
                for (int x = 0; x < croppedWidth; x++)
                {
                    int maskX = maskLeft + x;
                    int maskY = maskTop + y;

                    if (maskX >= maskSize || maskY >= maskSize)
                    {
                        solidTexture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        outlineTexture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        continue;
                    }

                    float maskValue = fullMask[maskY, maskX];

                    if (maskValue > 0.5f)
                    {
                        float alpha = maskValue * 0.8f;
                        Color pixelColor = new Color(maskColor.r, maskColor.g, maskColor.b, alpha);

                        // Solid texture - always fill
                        solidTexture.SetPixel(x, croppedHeight - 1 - y, pixelColor);

                        // Outline texture - only fill if on edge
                        if (isEdge[maskY, maskX])
                        {
                            outlineTexture.SetPixel(x, croppedHeight - 1 - y, pixelColor);
                        }
                        else
                        {
                            outlineTexture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        }
                    }
                    else
                    {
                        solidTexture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                        outlineTexture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                    }
                }
            }

            solidTexture.Apply();
            solidTexture.filterMode = FilterMode.Bilinear;
            outlineTexture.Apply();
            outlineTexture.filterMode = FilterMode.Bilinear;
        }

        private static float Sigmoid(float x)
        {
            return 1f / (1f + Mathf.Exp(-x));
        }
    }
}