// Segmentation mask processor for YOLOv11-seg
// Combines mask coefficients with prototype masks to generate final segmentation

using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    public static class SegmentationProcessor
    {
        // Generate segmentation mask for a single detection
        public static float[,] GenerateMask(float[] maskCoeffs, Tensor<float> prototypeMasks)
        {
            int maskSize = 160;
            float[,] resultMask = new float[maskSize, maskSize];

            for (int y = 0; y < maskSize; y++)
            {
                for (int x = 0; x < maskSize; x++)
                {
                    float pixelValue = 0f;

                    // Combine all 32 prototype masks using the coefficients
                    for (int i = 0; i < 32; i++)
                    {
                        float prototypeValue = prototypeMasks[0, i, y, x];
                        pixelValue += maskCoeffs[i] * prototypeValue;
                    }

                    // Apply sigmoid activation
                    resultMask[y, x] = Sigmoid(pixelValue);
                }
            }

            return resultMask;
        }

        // ADDED: Crop and scale mask to bounding box region
        public static Texture2D MaskToTexture(float[,] fullMask, Color maskColor, float boxCenterX, float boxCenterY, float boxWidth, float boxHeight, float imageWidth, float imageHeight)
        {
            int maskSize = 160;

            // Calculate scale factor: prototype mask (160) to input image (640)
            float scaleFactor = maskSize / imageWidth;

            // Convert bounding box from image coordinates to mask coordinates
            float maskCenterX = boxCenterX * scaleFactor;
            float maskCenterY = boxCenterY * scaleFactor;
            float maskBoxWidth = boxWidth * scaleFactor;
            float maskBoxHeight = boxHeight * scaleFactor;

            // Calculate bounding box bounds in mask space
            int maskLeft = Mathf.Max(0, Mathf.FloorToInt(maskCenterX - maskBoxWidth / 2));
            int maskTop = Mathf.Max(0, Mathf.FloorToInt(maskCenterY - maskBoxHeight / 2));
            int maskRight = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterX + maskBoxWidth / 2));
            int maskBottom = Mathf.Min(maskSize, Mathf.CeilToInt(maskCenterY + maskBoxHeight / 2));

            int croppedWidth = maskRight - maskLeft;
            int croppedHeight = maskBottom - maskTop;

            // Create texture from cropped region
            Texture2D texture = new Texture2D(croppedWidth, croppedHeight, TextureFormat.RGBA32, false);

            for (int y = 0; y < croppedHeight; y++)
            {
                for (int x = 0; x < croppedWidth; x++)
                {
                    int maskX = maskLeft + x;
                    int maskY = maskTop + y;

                    float maskValue = fullMask[maskY, maskX];

                    // Apply threshold - only show pixels above 0.5
                    if (maskValue > 0.5f)
                    {
                        Color pixelColor = new Color(maskColor.r, maskColor.g, maskColor.b, maskValue * 0.6f);
                        texture.SetPixel(x, croppedHeight - 1 - y, pixelColor);
                    }
                    else
                    {
                        texture.SetPixel(x, croppedHeight - 1 - y, Color.clear);
                    }
                }
            }

            texture.Apply();
            return texture;
        }

        // Sigmoid activation function
        private static float Sigmoid(float x)
        {
            return 1f / (1f + Mathf.Exp(-x));
        }
    }
}