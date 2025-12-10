// Copyright (c) Meta Platforms, Inc. and affiliates.
// Updated with spatial segmentation support - stores world-space dimensions
// Added distance-based outline rendering for live segmentations

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

        [Header("Anchored Segmentation Filter")]
        [Tooltip("Reference to DetectionManager to check for anchored segmentations")]
        [SerializeField] private DetectionManager m_detectionManager;
        [Tooltip("Hide live segmentation if anchored version exists within this distance")]
        [SerializeField] private float m_anchorFilterDistance = 0.5f;

        [Header("UI display references")]
        [SerializeField] private SentisObjectDetectedUiManager m_detectionCanvas;
        [SerializeField] private RawImage m_displayImage;
        [SerializeField] private Sprite m_boxTexture;
        [SerializeField] private Color m_boxColor;
        [SerializeField] private Font m_font;
        [SerializeField] private Color m_fontColor;
        [SerializeField] private int m_fontSize = 80;

        [Header("Distance-Based Live Segmentation")]
        [Tooltip("Enable distance-based outline switching for live segmentations")]
        [SerializeField] private bool m_enableLiveDistanceRendering = true;
        [Tooltip("Distance threshold (meters) to switch to outline mode")]
        [SerializeField] private float m_liveOutlineDistanceThreshold = 0.5f;
        [Tooltip("Outline border width in mask pixels")]
        [Range(1, 15)]
        [SerializeField] private int m_liveOutlineWidthPixels = 3;

        [Header("Adaptive Color Contrast")]
        [Tooltip("Enable Delta E based color selection for maximum contrast")]
        [SerializeField] private bool m_enableAdaptiveColor = true;
        [Tooltip("Number of background samples for color averaging")]
        [SerializeField] private int m_colorSampleCount = 16;
        [Tooltip("Alpha for segmentation overlay")]
        [Range(0.3f, 1.0f)]
        [SerializeField] private float m_segmentationAlpha = 0.7f;

        [Space(10)]
        public UnityEvent<int> OnObjectsDetected;

        public List<BoundingBox> BoxDrawn = new();
        private bool m_isLiveUiVisible = true;
        private string[] m_labels;
        private List<GameObject> m_boxPool = new();
        private Transform m_displayLocation;
        private Pose m_lastCaptureCameraPose;
        private OVRCameraRig m_cameraRig;
        public void ToggleLiveUiVisibility()
        {
            m_isLiveUiVisible = !m_isLiveUiVisible;

            // Immediate cleanup if turning off
            if (!m_isLiveUiVisible)
            {
                ClearAnnotations();
            }
        }
        // Bounding box data with spatial segmentation support
        public struct BoundingBox
        {
            // Canvas-space coordinates
            public float CenterX;
            public float CenterY;
            public float Width;
            public float Height;
            public Vector3? SurfaceNormal { get; set; }

            // Metadata
            public string Label;
            public string ClassName;
            public float Confidence;

            // World-space data
            public Vector3? WorldPos;
            public Pose CaptureCameraPose;

            // Segmentation data
            public float[] MaskCoefficients;
            public Texture2D SegmentationMask;        // Solid fill texture
            public Texture2D SegmentationMaskOutline; // Outline-only texture

            // Normalized coordinates (0-1 range) for world-space estimation
            public float NormalizedWidth;
            public float NormalizedHeight;

            // Estimated real-world dimensions in meters
            public float EstimatedWorldWidth;
            public float EstimatedWorldHeight;

            // Coverage metric (how much of the view the object covers)
            public float Coverage => NormalizedWidth * NormalizedHeight;
        }

        // Track which texture mode each box is currently showing
        private Dictionary<int, SegmentationRenderMode> m_currentBoxModes = new();

        #region Unity Functions
        private void Start()
        {
            m_displayLocation = m_displayImage.transform;
            Debug.Log($"DEBUG-LIVE-SEG: [INIT] SentisInferenceUiManager started");
            Debug.Log($"DEBUG-LIVE-SEG: [INIT] LiveDistanceRendering={m_enableLiveDistanceRendering}, Threshold={m_liveOutlineDistanceThreshold}m, OutlinePixels={m_liveOutlineWidthPixels}");
        }

        private void Update()
        {
            // Update live segmentation textures based on distance
            if (m_enableLiveDistanceRendering && BoxDrawn.Count > 0)
            {
                UpdateLiveSegmentationsByDistance();
            }
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
            // 3. ADD THIS CHECK AT THE VERY TOP OF THE FUNCTION
            if (!m_isLiveUiVisible)
            {
                ClearAnnotations();
                OnObjectsDetected?.Invoke(0);
                return; // Skip all calculation and drawing
            }
            ClearAnnotations();
            m_lastCaptureCameraPose = cameraPose;

            Debug.Log($"DEBUG-DETECTION: output.shape = {output.shape}");
            Debug.Log($"DEBUG-DETECTION: prototypeMasks.shape = {prototypeMasks.shape}");

            float displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;

            var boxesFound = output.shape[2];
            Debug.Log($"DEBUG-DETECTION: Total anchor boxes = {boxesFound}");

            List<DetectionCandidate> candidates = new List<DetectionCandidate>();

            for (var n = 0; n < boxesFound; n++)
            {
                int classId;
                float confidence;
                GetClassIdAndConfidence(output, n, out classId, out confidence);

                if (confidence < 0.75f)
                    continue;

                if (classId != 66)
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

            Debug.Log($"DEBUG-DETECTION: Before NMS: {candidates.Count}, After NMS: {filteredCandidates.Count}");

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

                    Debug.Log($"DEBUG-DETECTION: {classname} world size estimate: {estimatedWorldWidth:F3}m x {estimatedWorldHeight:F3}m");
                }

                // Extract mask coefficients
                float[] maskCoeffs = new float[32];
                for (int i = 0; i < 32; i++)
                {
                    maskCoeffs[i] = output[0, 84 + i, candidate.Index];
                }

                // Generate segmentation mask
                float[,] rawMask = SegmentationProcessor.GenerateMask(maskCoeffs, prototypeMasks);

                // Determine mask color using Delta E contrast if enabled
                Color maskColor;
                if (m_enableAdaptiveColor)
                {
                    // Sample background color from camera texture at object location
                    Rect sampleRect = new Rect(
                        normalizedCenterX - normalizedWidth / 2f,
                        normalizedCenterY - normalizedHeight / 2f,
                        normalizedWidth,
                        normalizedHeight
                    );

                    Texture cameraTexture = m_displayImage.texture;
                    Color bgColor = SegmentationColorContrast.SampleBackgroundFromTexture(cameraTexture, sampleRect, m_colorSampleCount);
                    maskColor = SegmentationColorContrast.GetBestContrastColor(bgColor, m_segmentationAlpha);

                    Debug.Log($"DEBUG-COLOR: [LIVE] '{classname}' - Background sampled, contrast color selected");
                }
                else
                {
                    // Fallback to class-based color
                    maskColor = candidate.ClassId == 0 ? new Color(0, 1, 0, m_segmentationAlpha) : new Color(0, 0, 1, m_segmentationAlpha);
                }

                // Generate both solid and outline textures for live distance-based rendering
                Texture2D solidTexture;
                Texture2D outlineTexture;

                if (m_enableLiveDistanceRendering)
                {
                    Debug.Log($"DEBUG-LIVE-SEG: [TEXTURE-GEN] Generating both textures for '{classname}'");

                    SegmentationProcessor.GenerateBothTextures(
                        rawMask,
                        maskColor,
                        candidate.X,
                        candidate.Y,
                        candidate.W,
                        candidate.H,
                        imageWidth,
                        imageHeight,
                        m_liveOutlineWidthPixels,
                        out solidTexture,
                        out outlineTexture
                    );

                    Debug.Log($"DEBUG-LIVE-SEG: [TEXTURE-GEN] Solid: {solidTexture?.width}x{solidTexture?.height}, Outline: {outlineTexture?.width}x{outlineTexture?.height}");
                }
                else
                {
                    // Original behavior - only solid texture
                    solidTexture = SegmentationProcessor.MaskToTexture(
                        rawMask,
                        maskColor,
                        candidate.X,
                        candidate.Y,
                        candidate.W,
                        candidate.H,
                        imageWidth,
                        imageHeight
                    );
                    outlineTexture = null;
                }

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
                    CaptureCameraPose = m_lastCaptureCameraPose,
                    MaskCoefficients = maskCoeffs,
                    SegmentationMask = solidTexture,
                    SegmentationMaskOutline = outlineTexture,
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

            Debug.Log($"DEBUG-DETECTION: Displayed {validDetections} valid detections");
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
            m_currentBoxModes.Clear();
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

            // Segmentation mask on canvas - determine which texture based on distance
            var maskImage = panel.transform.Find("SegmentationMask")?.GetComponent<RawImage>();
            if (maskImage != null && box.SegmentationMask != null)
            {
                // Check if there's already an anchored segmentation for this object
                bool hasAnchoredVersion = HasAnchoredSegmentationNearby(box);

                if (hasAnchoredVersion)
                {
                    // Hide live segmentation - anchored version exists
                    maskImage.gameObject.SetActive(false);
                    Debug.Log($"DEBUG-LIVE-SEG: [ANCHOR-FILTER] '{box.ClassName}' id={id} - Hiding live seg, anchored version exists");
                    return;
                }

                // Determine initial texture based on distance
                Texture2D initialTexture = box.SegmentationMask; // Default to solid
                SegmentationRenderMode initialMode = SegmentationRenderMode.SolidFill;

                if (m_enableLiveDistanceRendering && box.WorldPos.HasValue && box.SegmentationMaskOutline != null)
                {
                    float distance = GetDistanceToObject(box);

                    if (distance <= m_liveOutlineDistanceThreshold)
                    {
                        initialTexture = box.SegmentationMaskOutline;
                        initialMode = SegmentationRenderMode.OutlineOnly;
                    }

                    Debug.Log($"DEBUG-LIVE-SEG: [DRAW-BOX] '{box.ClassName}' id={id} - Distance={distance:F3}m, Threshold={m_liveOutlineDistanceThreshold}m, Mode={initialMode}");
                }

                maskImage.texture = initialTexture;
                maskImage.transform.localRotation = Quaternion.identity;
                maskImage.gameObject.SetActive(true);

                // Track current mode
                m_currentBoxModes[id] = initialMode;
            }
        }

        /// <summary>
        /// Check if there's an anchored segmentation nearby for the same class
        /// </summary>
        private bool HasAnchoredSegmentationNearby(BoundingBox box)
        {
            if (m_detectionManager == null || !box.WorldPos.HasValue)
                return false;

            return m_detectionManager.HasAnchoredSegmentationAt(box.WorldPos.Value, box.ClassName, m_anchorFilterDistance);
        }

        /// <summary>
        /// Update live segmentation textures based on current distance to user
        /// Called every frame when distance-based rendering is enabled
        /// </summary>
        private void UpdateLiveSegmentationsByDistance()
        {
            if (m_cameraRig == null)
            {
                m_cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (m_cameraRig == null)
                {
                    if (Time.frameCount % 60 == 0)
                    {
                        Debug.LogWarning("DEBUG-LIVE-SEG: [UPDATE] OVRCameraRig not found!");
                    }
                    return;
                }
            }

            for (int i = 0; i < BoxDrawn.Count && i < m_boxPool.Count; i++)
            {
                var box = BoxDrawn[i];
                var panel = m_boxPool[i];

                if (panel == null || !panel.activeInHierarchy)
                    continue;

                if (!box.WorldPos.HasValue || box.SegmentationMaskOutline == null)
                    continue;

                var maskImage = panel.transform.Find("SegmentationMask")?.GetComponent<RawImage>();
                if (maskImage == null || !maskImage.gameObject.activeInHierarchy)
                    continue;

                // Calculate current distance
                float distance = GetDistanceToObject(box);

                // Determine target mode
                SegmentationRenderMode targetMode = distance <= m_liveOutlineDistanceThreshold
                    ? SegmentationRenderMode.OutlineOnly
                    : SegmentationRenderMode.SolidFill;

                // Get current mode
                SegmentationRenderMode currentMode = SegmentationRenderMode.SolidFill;
                if (m_currentBoxModes.ContainsKey(i))
                {
                    currentMode = m_currentBoxModes[i];
                }

                // Log distance periodically
                if (Time.frameCount % 30 == 0)
                {
                    Debug.Log($"DEBUG-LIVE-SEG: [DISTANCE] '{box.ClassName}' id={i} - Distance={distance:F3}m, Threshold={m_liveOutlineDistanceThreshold}m, CurrentMode={currentMode}");
                }

                // Switch texture if mode changed
                if (targetMode != currentMode)
                {
                    Texture2D targetTexture = targetMode == SegmentationRenderMode.SolidFill
                        ? box.SegmentationMask
                        : box.SegmentationMaskOutline;

                    if (targetTexture != null)
                    {
                        maskImage.texture = targetTexture;
                        m_currentBoxModes[i] = targetMode;

                        Debug.Log($"DEBUG-LIVE-SEG: [MODE-CHANGE] '{box.ClassName}' id={i} - Switching {currentMode} -> {targetMode} at distance {distance:F3}m");
                    }
                }
            }
        }

        /// <summary>
        /// Calculate distance from user to detected object
        /// </summary>
        private float GetDistanceToObject(BoundingBox box)
        {
            if (!box.WorldPos.HasValue)
                return float.MaxValue;

            if (m_cameraRig == null)
            {
                m_cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (m_cameraRig == null)
                    return float.MaxValue;
            }

            Vector3 userPos = m_cameraRig.centerEyeAnchor.position;
            return Vector3.Distance(userPos, box.WorldPos.Value);
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

        #region Public Settings Access
        /// <summary>
        /// Update live outline distance threshold at runtime
        /// </summary>
        public void SetLiveOutlineDistanceThreshold(float distance)
        {
            m_liveOutlineDistanceThreshold = Mathf.Max(0.1f, distance);
            Debug.Log($"DEBUG-LIVE-SEG: [SETTINGS] Live outline threshold updated to {m_liveOutlineDistanceThreshold}m");
        }

        /// <summary>
        /// Enable/disable live distance-based rendering
        /// </summary>
        public void SetLiveDistanceRenderingEnabled(bool enabled)
        {
            m_enableLiveDistanceRendering = enabled;
            Debug.Log($"DEBUG-LIVE-SEG: [SETTINGS] Live distance rendering {(enabled ? "ENABLED" : "DISABLED")}");
        }
        #endregion
    }
}