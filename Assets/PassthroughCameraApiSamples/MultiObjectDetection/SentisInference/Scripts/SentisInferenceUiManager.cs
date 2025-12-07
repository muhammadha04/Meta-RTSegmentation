// Copyright (c) Meta Platforms, Inc. and affiliates.

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
        [Header("Placement configureation")]
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

        //bounding box data
        public struct BoundingBox
        {
            public float CenterX;
            public float CenterY;
            public float Width;
            public float Height;
            public string Label;
            public Vector3? WorldPos;
            public string ClassName;
            // ADDED: Store mask coefficients for segmentation processing
            public float[] MaskCoefficients;
            public Texture2D SegmentationMask;  // ADDED: Generated mask texture
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
        }

        // CHANGED: Signature now accepts Tensor<float> prototypeMasks instead of Tensor<int> labelIDs
        // For YOLOv11n-seg: output shape is (1, 116, 8400) where 116 = 4 box + 80 classes + 32 mask coeffs
        public void DrawUIBoxes(Tensor<float> output, Tensor<float> prototypeMasks, float imageWidth, float imageHeight, Pose cameraPose)
        {
            m_detectionCanvas.UpdatePosition();
            ClearAnnotations();

            Debug.Log($"DEBUG: output.shape = {output.shape}");
            Debug.Log($"DEBUG: prototypeMasks.shape = {prototypeMasks.shape}");

            float displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;

            var boxesFound = output.shape[2];  // 8400 anchor boxes
            Debug.Log($"DEBUG: Total anchor boxes = {boxesFound}");

            // ADDED: Collect all valid detections first
            List<DetectionCandidate> candidates = new List<DetectionCandidate>();

            // First pass: collect all detections above confidence threshold
            for (var n = 0; n < boxesFound; n++)
            {
                int classId;
                float confidence;
                GetClassIdAndConfidence(output, n, out classId, out confidence);

                // Filter by confidence threshold (hardcoded 0.75)
                if (confidence < 0.75f)
                    continue;

                // Filter by class - only person (0) and tv (62)
                if (classId != 43 && classId != 62 && classId!= 67)
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

            // ADDED: Apply NMS with hardcoded IOU threshold 0.5
            List<DetectionCandidate> filteredCandidates = ApplyNMS(candidates, 0.5f);

            Debug.Log($"DEBUG: Before NMS: {candidates.Count}, After NMS: {filteredCandidates.Count}");

            // Second pass: draw the filtered detections
            int validDetections = 0;
            foreach (var candidate in filteredCandidates)
            {
                var normalizedCenterX = candidate.X / imageWidth;
                var normalizedCenterY = candidate.Y / imageHeight;
                var centerX = displayWidth * (normalizedCenterX - 0.5f);
                var centerY = displayHeight * (normalizedCenterY - 0.5f);

                var classname = m_labels[candidate.ClassId].Replace(" ", "_");

                var ray = m_cameraAccess.ViewportPointToRay(new Vector2(normalizedCenterX, 1.0f - normalizedCenterY), cameraPose);
                var worldPos = m_environmentRaycast.Raycast(ray);

                // Extract mask coefficients
                float[] maskCoeffs = new float[32];
                for (int i = 0; i < 32; i++)
                {
                    maskCoeffs[i] = output[0, 84 + i, candidate.Index];
                }

                // ADDED: Generate full segmentation mask (160x160)
                float[,] rawMask = SegmentationProcessor.GenerateMask(maskCoeffs, prototypeMasks);

                // ADDED: Create texture with color (green for person, blue for TV)
                Color maskColor = candidate.ClassId == 0 ? new Color(0, 1, 0, 0.5f) : new Color(0, 0, 1, 0.5f);  // Green for person, blue for TV

                // FIXED: Crop mask to bounding box region and create texture
                Texture2D maskTexture = SegmentationProcessor.MaskToTexture(
                    rawMask,
                    maskColor,
                    candidate.X,      // Box center X in image coordinates
                    candidate.Y,      // Box center Y in image coordinates
                    candidate.W,      // Box width in image coordinates
                    candidate.H,      // Box height in image coordinates
                    imageWidth,       // Full image width (640)
                    imageHeight       // Full image height (640)
                );

                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = classname,
                    Width = candidate.W * (displayWidth / imageWidth),
                    Height = candidate.H * (displayHeight / imageHeight),
                    Label = $"Class: {classname} Conf: {candidate.Confidence:F2} Center: {normalizedCenterX:F2},{normalizedCenterY:F2}",
                    WorldPos = worldPos,
                    MaskCoefficients = maskCoeffs,
                    SegmentationMask = maskTexture  // ADDED: Store generated mask
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

        // ADDED: Helper struct for NMS
        private struct DetectionCandidate
        {
            public int Index;
            public int ClassId;
            public float Confidence;
            public float X, Y, W, H;
        }

        // ADDED: Non-Maximum Suppression
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

        // ADDED: Calculate IOU
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

        // ADDED: Helper method to extract class ID and confidence from output tensor
        // Finds the index and value of maximum score among the 80 class channels (4-83)
        // Tensor shape is (1, 116, 8400) so indexing is [0, channel, detection]
        private void GetClassIdAndConfidence(Tensor<float> output, int detectionIndex, out int classId, out float confidence)
        {
            classId = 0;
            confidence = output[0, 4, detectionIndex];  // First class score

            // Find max score among channels 4-83 (80 classes)
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

            panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, box.WorldPos.HasValue ? box.WorldPos.Value.z : 0.0f);

            // Calculate direction but lock Y rotation only (prevents tilting)
            Vector3 directionToCamera = panel.transform.position - m_detectionCanvas.GetCapturedCameraPosition();
            directionToCamera.y = 0;  // Lock to horizontal plane
            Quaternion panelRotation;
            if (directionToCamera != Vector3.zero)
                panelRotation = Quaternion.LookRotation(directionToCamera);
            else
                panelRotation = Quaternion.identity;

            panel.transform.rotation = panelRotation;

            var rt = panel.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(box.Width, box.Height);

            var label = panel.GetComponentInChildren<Text>();
            label.text = box.Label;
            label.fontSize = 12;

            // FIXED: Update segmentation mask overlay and counter-rotate it to stay flat
            var maskImage = panel.transform.Find("SegmentationMask")?.GetComponent<RawImage>();
            if (maskImage != null && box.SegmentationMask != null)
            {
                maskImage.texture = box.SegmentationMask;
                // CRITICAL FIX: Counter-rotate the mask to keep it aligned with the UI plane
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

            // ADDED: Create segmentation mask overlay
            var maskObj = new GameObject("SegmentationMask");
            _ = maskObj.AddComponent<CanvasRenderer>();
            maskObj.transform.SetParent(panel.transform, false);
            var maskImage = maskObj.AddComponent<RawImage>();

            var maskRt = maskObj.GetComponent<RectTransform>();
            maskRt.anchorMin = new Vector2(0, 0);
            maskRt.anchorMax = new Vector2(1, 1);
            maskRt.offsetMin = Vector2.zero;
            maskRt.offsetMax = Vector2.zero;
            maskObj.SetActive(false);  // Hidden until mask is set

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