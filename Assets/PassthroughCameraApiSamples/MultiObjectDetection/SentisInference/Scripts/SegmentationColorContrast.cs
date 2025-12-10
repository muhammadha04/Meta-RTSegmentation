// Segmentation Color Contrast Utility
// Uses Delta E (CIE76) to select maximum contrast segmentation colors based on background

using UnityEngine;
using System.Collections.Generic;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    public static class SegmentationColorContrast
    {
        // Predefined palette of segmentation colors (saturated, good for overlays)
        private static readonly Color[] s_candidateColors = new Color[]
        {
            new Color(1.0f, 0.0f, 0.0f, 0.7f),      // Red
            new Color(0.0f, 1.0f, 0.0f, 0.7f),      // Green
            new Color(0.0f, 0.0f, 1.0f, 0.7f),      // Blue
            new Color(1.0f, 1.0f, 0.0f, 0.7f),      // Yellow
            new Color(1.0f, 0.0f, 1.0f, 0.7f),      // Magenta
            new Color(0.0f, 1.0f, 1.0f, 0.7f),      // Cyan
            new Color(1.0f, 0.5f, 0.0f, 0.7f),      // Orange
            new Color(0.5f, 0.0f, 1.0f, 0.7f),      // Purple
            new Color(0.0f, 1.0f, 0.5f, 0.7f),      // Spring Green
            new Color(1.0f, 0.0f, 0.5f, 0.7f),      // Rose
            new Color(1.0f, 1.0f, 1.0f, 0.7f),      // White
            new Color(0.0f, 0.0f, 0.0f, 0.7f),      // Black
        };

        /// <summary>
        /// Get the best contrasting color for a segmentation overlay based on background
        /// </summary>
        /// <param name="backgroundColor">Average background color behind the object</param>
        /// <param name="alpha">Alpha value for the returned color</param>
        /// <returns>Color with highest Delta E contrast against background</returns>
        public static Color GetBestContrastColor(Color backgroundColor, float alpha = 0.7f)
        {
            // Convert background to LAB
            Vector3 bgLab = RGBtoLAB(backgroundColor);

            Color bestColor = s_candidateColors[0];
            float maxDeltaE = 0f;

            foreach (var candidate in s_candidateColors)
            {
                Vector3 candidateLab = RGBtoLAB(candidate);
                float deltaE = CalculateDeltaE(bgLab, candidateLab);

                if (deltaE > maxDeltaE)
                {
                    maxDeltaE = deltaE;
                    bestColor = candidate;
                }
            }

            Debug.Log($"DEBUG-COLOR: [CONTRAST] Background RGB=({backgroundColor.r:F2},{backgroundColor.g:F2},{backgroundColor.b:F2}), " +
                      $"BgLAB=({bgLab.x:F1},{bgLab.y:F1},{bgLab.z:F1}), " +
                      $"BestColor RGB=({bestColor.r:F2},{bestColor.g:F2},{bestColor.b:F2}), DeltaE={maxDeltaE:F1}");

            return new Color(bestColor.r, bestColor.g, bestColor.b, alpha);
        }

        /// <summary>
        /// Get best contrast color with custom candidate palette
        /// </summary>
        public static Color GetBestContrastColor(Color backgroundColor, Color[] customPalette, float alpha = 0.7f)
        {
            Vector3 bgLab = RGBtoLAB(backgroundColor);

            Color bestColor = customPalette[0];
            float maxDeltaE = 0f;

            foreach (var candidate in customPalette)
            {
                Vector3 candidateLab = RGBtoLAB(candidate);
                float deltaE = CalculateDeltaE(bgLab, candidateLab);

                if (deltaE > maxDeltaE)
                {
                    maxDeltaE = deltaE;
                    bestColor = candidate;
                }
            }

            return new Color(bestColor.r, bestColor.g, bestColor.b, alpha);
        }

        /// <summary>
        /// Sample average background color from a texture region
        /// </summary>
        /// <param name="texture">Source texture (camera feed)</param>
        /// <param name="normalizedRect">Normalized rect (0-1) defining the sample region</param>
        /// <param name="sampleCount">Number of samples to average (more = accurate but slower)</param>
        public static Color SampleBackgroundColor(Texture2D texture, Rect normalizedRect, int sampleCount = 16)
        {
            if (texture == null)
            {
                Debug.LogWarning("DEBUG-COLOR: [SAMPLE] Texture is null, returning gray");
                return Color.gray;
            }

            float r = 0f, g = 0f, b = 0f;
            int validSamples = 0;

            int texWidth = texture.width;
            int texHeight = texture.height;

            // Convert normalized rect to pixel coordinates
            int left = Mathf.FloorToInt(normalizedRect.x * texWidth);
            int bottom = Mathf.FloorToInt(normalizedRect.y * texHeight);
            int width = Mathf.CeilToInt(normalizedRect.width * texWidth);
            int height = Mathf.CeilToInt(normalizedRect.height * texHeight);

            // Clamp to texture bounds
            left = Mathf.Clamp(left, 0, texWidth - 1);
            bottom = Mathf.Clamp(bottom, 0, texHeight - 1);
            width = Mathf.Clamp(width, 1, texWidth - left);
            height = Mathf.Clamp(height, 1, texHeight - bottom);

            // Sample grid
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(sampleCount));
            float stepX = width / (float)gridSize;
            float stepY = height / (float)gridSize;

            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    int x = left + Mathf.FloorToInt(i * stepX + stepX / 2);
                    int y = bottom + Mathf.FloorToInt(j * stepY + stepY / 2);

                    x = Mathf.Clamp(x, 0, texWidth - 1);
                    y = Mathf.Clamp(y, 0, texHeight - 1);

                    Color pixel = texture.GetPixel(x, y);
                    r += pixel.r;
                    g += pixel.g;
                    b += pixel.b;
                    validSamples++;
                }
            }

            if (validSamples == 0)
            {
                Debug.LogWarning("DEBUG-COLOR: [SAMPLE] No valid samples, returning gray");
                return Color.gray;
            }

            Color avgColor = new Color(r / validSamples, g / validSamples, b / validSamples);
            Debug.Log($"DEBUG-COLOR: [SAMPLE] Sampled {validSamples} pixels, AvgColor=({avgColor.r:F2},{avgColor.g:F2},{avgColor.b:F2})");

            return avgColor;
        }

        /// <summary>
        /// Sample background color from RenderTexture (for live camera feed)
        /// </summary>
        public static Color SampleBackgroundColorFromRenderTexture(RenderTexture rt, Rect normalizedRect, int sampleCount = 16)
        {
            if (rt == null)
            {
                Debug.LogWarning("DEBUG-COLOR: [SAMPLE-RT] RenderTexture is null");
                return Color.gray;
            }

            // Create temporary Texture2D to read pixels
            RenderTexture currentRT = RenderTexture.active;
            RenderTexture.active = rt;

            Texture2D tempTex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tempTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tempTex.Apply();

            RenderTexture.active = currentRT;

            Color result = SampleBackgroundColor(tempTex, normalizedRect, sampleCount);

            // Clean up
            Object.Destroy(tempTex);

            return result;
        }

        /// <summary>
        /// Sample background from any Texture type
        /// </summary>
        public static Color SampleBackgroundFromTexture(Texture texture, Rect normalizedRect, int sampleCount = 16)
        {
            if (texture == null)
            {
                Debug.LogWarning("DEBUG-COLOR: [SAMPLE-ANY] Texture is null");
                return Color.gray;
            }

            if (texture is Texture2D tex2D)
            {
                // Check if texture is readable
                try
                {
                    tex2D.GetPixel(0, 0);
                    return SampleBackgroundColor(tex2D, normalizedRect, sampleCount);
                }
                catch
                {
                    Debug.LogWarning("DEBUG-COLOR: [SAMPLE-ANY] Texture2D not readable, using fallback");
                    return Color.gray;
                }
            }
            else if (texture is RenderTexture rt)
            {
                return SampleBackgroundColorFromRenderTexture(rt, normalizedRect, sampleCount);
            }
            else
            {
                // For other texture types, try to blit to RenderTexture
                RenderTexture tempRT = RenderTexture.GetTemporary(texture.width, texture.height);
                Graphics.Blit(texture, tempRT);
                Color result = SampleBackgroundColorFromRenderTexture(tempRT, normalizedRect, sampleCount);
                RenderTexture.ReleaseTemporary(tempRT);
                return result;
            }
        }

        #region Color Space Conversion

        /// <summary>
        /// Convert RGB to CIE LAB color space
        /// </summary>
        public static Vector3 RGBtoLAB(Color rgb)
        {
            // RGB to XYZ
            Vector3 xyz = RGBtoXYZ(rgb);

            // XYZ to LAB
            return XYZtoLAB(xyz);
        }

        /// <summary>
        /// Convert RGB to XYZ (sRGB with D65 illuminant)
        /// </summary>
        private static Vector3 RGBtoXYZ(Color rgb)
        {
            // Linearize sRGB
            float r = LinearizeChannel(rgb.r);
            float g = LinearizeChannel(rgb.g);
            float b = LinearizeChannel(rgb.b);

            // RGB to XYZ matrix (sRGB, D65)
            float x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
            float y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
            float z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

            return new Vector3(x * 100f, y * 100f, z * 100f);
        }

        /// <summary>
        /// Linearize sRGB channel
        /// </summary>
        private static float LinearizeChannel(float c)
        {
            if (c > 0.04045f)
                return Mathf.Pow((c + 0.055f) / 1.055f, 2.4f);
            else
                return c / 12.92f;
        }

        /// <summary>
        /// Convert XYZ to LAB (D65 reference white)
        /// </summary>
        private static Vector3 XYZtoLAB(Vector3 xyz)
        {
            // D65 reference white
            const float refX = 95.047f;
            const float refY = 100.000f;
            const float refZ = 108.883f;

            float x = LabFunc(xyz.x / refX);
            float y = LabFunc(xyz.y / refY);
            float z = LabFunc(xyz.z / refZ);

            float L = 116f * y - 16f;
            float a = 500f * (x - y);
            float b = 200f * (y - z);

            return new Vector3(L, a, b);
        }

        /// <summary>
        /// LAB conversion function
        /// </summary>
        private static float LabFunc(float t)
        {
            const float delta = 6f / 29f;
            const float deltaCubed = delta * delta * delta;

            if (t > deltaCubed)
                return Mathf.Pow(t, 1f / 3f);
            else
                return t / (3f * delta * delta) + 4f / 29f;
        }

        #endregion

        #region Delta E Calculation

        /// <summary>
        /// Calculate Delta E (CIE76) between two LAB colors
        /// Simple Euclidean distance in LAB space
        /// </summary>
        public static float CalculateDeltaE(Vector3 lab1, Vector3 lab2)
        {
            float dL = lab1.x - lab2.x;
            float da = lab1.y - lab2.y;
            float db = lab1.z - lab2.z;

            return Mathf.Sqrt(dL * dL + da * da + db * db);
        }

        /// <summary>
        /// Calculate Delta E directly from two RGB colors
        /// </summary>
        public static float CalculateDeltaE(Color rgb1, Color rgb2)
        {
            Vector3 lab1 = RGBtoLAB(rgb1);
            Vector3 lab2 = RGBtoLAB(rgb2);
            return CalculateDeltaE(lab1, lab2);
        }

        #endregion

        #region Utility

        /// <summary>
        /// Get Delta E interpretation
        /// </summary>
        public static string GetDeltaEInterpretation(float deltaE)
        {
            if (deltaE < 1f) return "Not perceptible";
            if (deltaE < 2f) return "Perceptible through close observation";
            if (deltaE < 3.5f) return "Perceptible at a glance";
            if (deltaE < 5f) return "Obvious difference";
            return "Very obvious difference";
        }

        /// <summary>
        /// Debug: Log all candidate colors with their Delta E against a background
        /// </summary>
        public static void DebugLogAllCandidates(Color backgroundColor)
        {
            Vector3 bgLab = RGBtoLAB(backgroundColor);
            Debug.Log($"DEBUG-COLOR: [ALL-CANDIDATES] Background: RGB({backgroundColor.r:F2},{backgroundColor.g:F2},{backgroundColor.b:F2}) LAB({bgLab.x:F1},{bgLab.y:F1},{bgLab.z:F1})");

            for (int i = 0; i < s_candidateColors.Length; i++)
            {
                var c = s_candidateColors[i];
                Vector3 cLab = RGBtoLAB(c);
                float deltaE = CalculateDeltaE(bgLab, cLab);
                Debug.Log($"DEBUG-COLOR: [CANDIDATE {i}] RGB({c.r:F2},{c.g:F2},{c.b:F2}) LAB({cLab.x:F1},{cLab.y:F1},{cLab.z:F1}) DeltaE={deltaE:F1} ({GetDeltaEInterpretation(deltaE)})");
            }
        }

        #endregion
    }
}
