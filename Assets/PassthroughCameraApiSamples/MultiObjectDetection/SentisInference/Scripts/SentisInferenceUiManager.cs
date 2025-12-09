// Copyright (c) Meta Platforms, Inc. and affiliates.
// Updated with spatial segmentation support - stores world-space dimensions

using System.Collections.Generic;
using Meta.XR;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceUiManager : MonoBehaviour
    {
        [Header("Placement configuration")]
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;

        [Header("UI display references")]
        [SerializeField] private SentisObjectDetectedUiManager m_detectionCanvas;
        [SerializeField] private RawImage m_displayImage;
        [SerializeField] private Sprite m_boxTexture;
        [SerializeField] private Color m_boxColor;
        [SerializeField] private Font m_font;
        [SerializeField] private Color m_fontColor;
        [SerializeField] private int m_fontSize = 80;
        [Space(10)]
        public UnityEvent<int> OnObjectsDetected;

        public List<BoundingBox> BoxDrawn = new();

        private string[] m_labels;
        private List<GameObject> m_boxPool = new();
        private Transform m_displayLocation;

        // Bounding box data with spatial segmentation support
        public struct BoundingBox
        {
            // Canvas-space coordinates
            public float CenterX;
            public float CenterY;
            public float Width;
            public float Height;

            // Metadata
            public string Label;
            public string ClassName;
            public float Confidence;  // Detection confidence score

            // World-space data
            public Vector3? WorldPos;

            // Segmentation data
            public float[] MaskCoefficients;
            public Texture2D SegmentationMask;

            // Normalized coordinates (0-1 range) for world-space estimation
            public float NormalizedWidth;
            public float NormalizedHeight;

            // Estimated real-world dimensions in meters
            public float EstimatedWorldWidth;
            public float EstimatedWorldHeight;

            // Coverage metric (how much of the view the object covers)
            public float Coverage => NormalizedWidth * NormalizedHeight;
        }

        #region Unity Functions
        private void Start()
        {
            m_displayLocation = m_displayImage.transform;
        }
        #endregion

        #region Detection Functions
        public void OnObjectDetectionError()
        {
            ClearAnnotations();
            OnObjectsDetected?.Invoke(0);
        }
        #endregion

        #region BoundingBoxes functions
        public void SetLabels(TextAsset labelsAsset)
        {
            m_labels = labelsAsset.text.Split('\n');
        }

        public void SetDetectionCapture(Texture image)
        {
            m_displayImage.texture = image;
            m_detectionCanvas.CapturePosition();
            m_detectionCanvas.UpdatePosition();
        }

        public void DrawUIBoxes(Tensor<float> output, Tensor<float> prototypeMasks, float imageWidth, float imageHeight, Pose cameraPose)
        {
            ClearAnnotations();

            Debug.Log($"DEBUG: output.shape = {output.shape}");
            Debug.Log($"DEBUG: prototypeMasks.shape = {prototypeMasks.shape}");

            float displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;

            var boxesFound = output.shape[2];
            Debug.Log($"DEBUG: Total anchor boxes = {boxesFound}");

            List<DetectionCandidate> candidates = new List<DetectionCandidate>();

            for (var n = 0; n < boxesFound; n++)
            {
                int classId;
                float confidence;
                GetClassIdAndConfidence(output, n, out classId, out confidence);

                if (confidence < 0.75f)
                    continue;

                if (classId != 64 && classId != 66 && classId != 62)
                    continue;

                candidates.Add(new DetectionCandidate
                {
                    Index = n,
                    ClassId = classId,
                    Confidence = confidence,
                    X = output[0, 0, n],
                    Y = output[0, 1, n],
                    W = output[0, 2, n],
                    H = output[0, 3, n]
                });
            }

            List<DetectionCandidate> filteredCandidates = ApplyNMS(candidates, 0.5f);

            Debug.Log($"DEBUG: Before NMS: {candidates.Count}, After NMS: {filteredCandidates.Count}");

            int validDetections = 0;
            foreach (var candidate in filteredCandidates)
            {
                // Normalized coordinates (0-1 range)
                var normalizedCenterX = candidate.X / imageWidth;
                var normalizedCenterY = candidate.Y / imageHeight;
                var normalizedWidth = candidate.W / imageWidth;
                var normalizedHeight = candidate.H / imageHeight;

                // Canvas-space coordinates
                var centerX = displayWidth * (normalizedCenterX - 0.5f);
                var centerY = displayHeight * (normalizedCenterY - 0.5f);

                var classname = m_labels[candidate.ClassId].Replace(" ", "_");

                // Raycast center to get world position
                var centerRay = m_cameraAccess.ViewportPointToRay(
                    new Vector2(normalizedCenterX, 1.0f - normalizedCenterY),
                    cameraPose
                );
                var worldPos = m_environmentRaycast.Raycast(centerRay);

                // Estimate world dimensions using corner raycasts
                float estimatedWorldWidth = 0f;
                float estimatedWorldHeight = 0f;

                if (worldPos.HasValue)
                {
                    // Calculate corner positions
                    float left = normalizedCenterX - normalizedWidth / 2f;
                    float right = normalizedCenterX + normalizedWidth / 2f;
                    float top = normalizedCenterY - normalizedHeight / 2f;
                    float bottom = normalizedCenterY + normalizedHeight / 2f;

                    // Clamp to valid viewport range
                    left = Mathf.Clamp01(left);
                    right = Mathf.Clamp01(right);
                    top = Mathf.Clamp01(top);
                    bottom = Mathf.Clamp01(bottom);

                    // Raycast corners (Y is flipped for viewport coordinates)
                    var leftRay = m_cameraAccess.ViewportPointToRay(new Vector2(left, 1f - normalizedCenterY), cameraPose);
                    var rightRay = m_cameraAccess.ViewportPointToRay(new Vector2(right, 1f - normalizedCenterY), cameraPose);
                    var topRay = m_cameraAccess.ViewportPointToRay(new Vector2(normalizedCenterX, 1f - top), cameraPose);
                    var bottomRay = m_cameraAccess.ViewportPointToRay(new Vector2(normalizedCenterX, 1f - bottom), cameraPose);

                    var worldLeft = m_environmentRaycast.Raycast(leftRay);
                    var worldRight = m_environmentRaycast.Raycast(rightRay);
                    var worldTop = m_environmentRaycast.Raycast(topRay);
                    var worldBottom = m_environmentRaycast.Raycast(bottomRay);

                    // Calculate world dimensions from successful raycasts
                    if (worldLeft.HasValue && worldRight.HasValue)
                    {
                        estimatedWorldWidth = Vector3.Distance(worldLeft.Value, worldRight.Value);
                    }

                    if (worldTop.HasValue && worldBottom.HasValue)
                    {
                        estimatedWorldHeight = Vector3.Distance(worldTop.Value, worldBottom.Value);
                    }

                    // Fallback: estimate from depth and normalized dimensions
                    if (estimatedWorldWidth < 0.01f || estimatedWorldHeight < 0.01f)
                    {
                        float depth = Vector3.Distance(cameraPose.position, worldPos.Value);
                        // Approximate based on typical Quest 3 camera FOV (~90 degrees horizontal)
                        float fovScale = 2f * depth * Mathf.Tan(45f * Mathf.Deg2Rad);

                        if (estimatedWorldWidth < 0.01f)
                        {
                            estimatedWorldWidth = normalizedWidth * fovScale;
                        }
                        if (estimatedWorldHeight < 0.01f)
                        {
                            estimatedWorldHeight = normalizedHeight * fovScale;
                        }
                    }

                    Debug.Log($"DEBUG: {classname} world size estimate: {estimatedWorldWidth:F3}m x {estimatedWorldHeight:F3}m");
                }

                // Extract mask coefficients
                float[] maskCoeffs = new float[32];
                for (int i = 0; i < 32; i++)
                {
                    maskCoeffs[i] = output[0, 84 + i, candidate.Index];
                }

                // Generate segmentation mask
                float[,] rawMask = SegmentationProcessor.GenerateMask(maskCoeffs, prototypeMasks);

                Color maskColor = candidate.ClassId == 0 ? new Color(0, 1, 0, 0.5f) : new Color(0, 0, 1, 0.5f);

                Texture2D maskTexture = SegmentationProcessor.MaskToTexture(
                    rawMask,
                    maskColor,
                    candidate.X,
                    candidate.Y,
                    candidate.W,
                    candidate.H,
                    imageWidth,
                    imageHeight
                );

                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = classname,
                    Confidence = candidate.Confidence,
                    Width = candidate.W * (displayWidth / imageWidth),
                    Height = candidate.H * (displayHeight / imageHeight),
                    Label = $"Class: {classname} Conf: {candidate.Confidence:F2}",
                    WorldPos = worldPos,
                    MaskCoefficients = maskCoeffs,
                    SegmentationMask = maskTexture,
                    NormalizedWidth = normalizedWidth,
                    NormalizedHeight = normalizedHeight,
                    EstimatedWorldWidth = estimatedWorldWidth,
                    EstimatedWorldHeight = estimatedWorldHeight
                };

                BoxDrawn.Add(box);
                DrawBox(box, validDetections);
                validDetections++;

                if (validDetections >= 50)
                    break;
            }

            Debug.Log($"DEBUG: Displayed {validDetections} valid detections");
            OnObjectsDetected?.Invoke(validDetections);
        }

        private struct DetectionCandidate
        {
            public int Index;
            public int ClassId;
            public float Confidence;
            public float X, Y, W, H;
        }

        private List<DetectionCandidate> ApplyNMS(List<DetectionCandidate> candidates, float iouThreshold)
        {
            candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

            List<DetectionCandidate> result = new List<DetectionCandidate>();
            bool[] suppressed = new bool[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                if (suppressed[i])
                    continue;

                result.Add(candidates[i]);

                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (suppressed[j])
                        continue;

                    if (candidates[i].ClassId != candidates[j].ClassId)
                        continue;

                    float iou = CalculateIOU(candidates[i], candidates[j]);
                    if (iou > iouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }

            return result;
        }

        private float CalculateIOU(DetectionCandidate a, DetectionCandidate b)
        {
            float x1 = Mathf.Max(a.X - a.W / 2, b.X - b.W / 2);
            float y1 = Mathf.Max(a.Y - a.H / 2, b.Y - b.H / 2);
            float x2 = Mathf.Min(a.X + a.W / 2, b.X + b.W / 2);
            float y2 = Mathf.Min(a.Y + a.H / 2, b.Y + b.H / 2);

            float intersectionWidth = Mathf.Max(0, x2 - x1);
            float intersectionHeight = Mathf.Max(0, y2 - y1);
            float intersectionArea = intersectionWidth * intersectionHeight;

            float areaA = a.W * a.H;
            float areaB = b.W * b.H;
            float unionArea = areaA + areaB - intersectionArea;

            return intersectionArea / (unionArea + 1e-6f);
        }

        private void GetClassIdAndConfidence(Tensor<float> output, int detectionIndex, out int classId, out float confidence)
        {
            classId = 0;
            confidence = output[0, 4, detectionIndex];

            for (int i = 1; i < 80; i++)
            {
                float score = output[0, 4 + i, detectionIndex];
                if (score > confidence)
                {
                    confidence = score;
                    classId = i;
                }
            }
        }

        private void ClearAnnotations()
        {
            foreach (var box in m_boxPool)
            {
                box?.SetActive(false);
            }
            BoxDrawn.Clear();
        }

        private void DrawBox(BoundingBox box, int id)
        {
            GameObject panel;
            if (id < m_boxPool.Count)
            {
                panel = m_boxPool[id];
                if (panel == null)
                {
                    panel = CreateNewBox(m_boxColor);
                }
                else
                {
                    panel.SetActive(true);
                }
            }
            else
            {
                panel = CreateNewBox(m_boxColor);
            }

            // Use LOCAL transforms only - let canvas handle world positioning
            panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, 0.0f);
            panel.transform.localRotation = Quaternion.identity;

            var rt = panel.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(box.Width, box.Height);

            var label = panel.GetComponentInChildren<Text>();
            label.text = box.Label;
            label.fontSize = 12;

            // Segmentation mask on canvas
            var maskImage = panel.transform.Find("SegmentationMask")?.GetComponent<RawImage>();
            if (maskImage != null && box.SegmentationMask != null)
            {
                maskImage.texture = box.SegmentationMask;
                maskImage.transform.localRotation = Quaternion.identity;
                maskImage.gameObject.SetActive(true);
            }
        }

        private GameObject CreateNewBox(Color color)
        {
            var panel = new GameObject("ObjectBox");
            _ = panel.AddComponent<CanvasRenderer>();
            var img = panel.AddComponent<Image>();
            img.color = color;
            img.sprite = m_boxTexture;
            img.type = Image.Type.Sliced;
            img.fillCenter = false;
            panel.transform.SetParent(m_displayLocation, false);

            // Create segmentation mask overlay
            var maskObj = new GameObject("SegmentationMask");
            _ = maskObj.AddComponent<CanvasRenderer>();
            maskObj.transform.SetParent(panel.transform, false);
            var maskImage = maskObj.AddComponent<RawImage>();

            var maskRt = maskObj.GetComponent<RectTransform>();
            maskRt.anchorMin = new Vector2(0, 0);
            maskRt.anchorMax = new Vector2(1, 1);
            maskRt.offsetMin = Vector2.zero;
            maskRt.offsetMax = Vector2.zero;
            maskObj.SetActive(false);

            var text = new GameObject("ObjectLabel");
            _ = text.AddComponent<CanvasRenderer>();
            text.transform.SetParent(panel.transform, false);
            var txt = text.AddComponent<Text>();
            txt.font = m_font;
            txt.color = m_fontColor;
            txt.fontSize = m_fontSize;
            txt.horizontalOverflow = HorizontalWrapMode.Overflow;

            var rt2 = text.GetComponent<RectTransform>();
            rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
            rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
            rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
            rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
            rt2.anchorMin = new Vector2(0, 0);
            rt2.anchorMax = new Vector2(1, 1);

            m_boxPool.Add(panel);
            return panel;
        }
        #endregion
    }
}