// Copyright (c) Meta Platforms, Inc. and affiliates.
// Updated with distance-based segmentation rendering support

using System;
using System.Collections;
using System.Collections.Generic;
using Meta.XR;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UIElements;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionManager : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;

        [Header("Controls configuration")]
        [SerializeField] private OVRInput.RawButton m_actionButton = OVRInput.RawButton.A;
        [SerializeField] private OVRInput.RawButton m_resetButton = OVRInput.RawButton.B;
        [SerializeField] private OVRInput.RawButton m_toggleUiButton = OVRInput.RawButton.RIndexTrigger;
        [SerializeField] private OVRInput.RawButton m_toggleAutoUpdateButton = OVRInput.RawButton.RHandTrigger;
        [Header("Ui references")]
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;

        [Header("Placement configuration")]
        [SerializeField] private GameObject m_spwanMarker;
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private float m_spawnDistance = 0.25f;
        [SerializeField] private AudioSource m_placeSound;

        [Header("Sentis inference ref")]
        [SerializeField] private SentisInferenceRunManager m_runInference;
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("Spatial Segmentation")]
        [Tooltip("Enable spatial segmentation markers")]
        [SerializeField] private bool m_enableSpatialSegmentation = true;
        [Tooltip("Offset from surface to prevent z-fighting (meters)")]
        [SerializeField] private float m_segmentationSurfaceOffset = 0.01f;
        [Tooltip("Default size if world estimation fails (meters)")]
        [SerializeField] private float m_fallbackSegmentationSize = 0.3f;

        [Header("Distance-Based Segmentation")]
        [Tooltip("Enable distance-based outline switching")]
        [SerializeField] private bool m_enableDistanceBasedRendering = true;
        [Tooltip("Distance threshold (meters) to switch to outline mode - closer than this shows outline")]
        [SerializeField] private float m_outlineDistanceThreshold = 1.0f;
        [Tooltip("Smooth transition zone width (meters)")]
        [SerializeField] private float m_transitionBlendWidth = 0.2f;
        [Tooltip("Outline border width in mask pixels (1-10 recommended)")]
        [Range(1, 15)]
        [SerializeField] private int m_outlineWidthPixels = 3;

        [Header("Adaptive Color Contrast")]
        [Tooltip("Enable Delta E based color selection for anchored segmentations")]
        [SerializeField] private bool m_enableAdaptiveColorAnchored = true;
        [Tooltip("Alpha for anchored segmentation overlay")]
        [Range(0.3f, 1.0f)]
        [SerializeField] private float m_anchoredSegmentationAlpha = 0.7f;

        [Header("Auto-Update Segmentation")]
        [Tooltip("Automatically update segmentations for objects in view")]
        [SerializeField] private bool m_autoUpdateSegmentations = true;
        [Tooltip("How often to check for updates (seconds)")]
        [SerializeField] private float m_autoUpdateInterval = 0.5f;
        [Tooltip("Minimum quality improvement needed to trigger update (0-1). Higher = fewer updates")]
        [SerializeField] private float m_qualityImprovementThreshold = 0.05f;

        [Space(10)]
        public UnityEvent<int> OnObjectsIdentified;

        private bool m_isPaused = true;
        private List<GameObject> m_spwanedEntities = new();
        private List<GameObject> m_spawnedSegmentations = new();
        private bool m_isStarted = false;
        private bool m_isSentisReady = false;
        private float m_delayPauseBackTime = 0;

        // Auto-update timer
        private float m_autoUpdateTimer = 0f;

        // Cached shader for Quest compatibility
        private Shader m_segmentationShader;

        #region Unity Functions
        private void Awake()
        {
            OVRManager.display.RecenteredPose += CleanMarkersCallBack;

            // Cache shader at startup - use Mobile/Particles/Alpha Blended which exists on Quest
            m_segmentationShader = Shader.Find("Mobile/Particles/Alpha Blended");
            if (m_segmentationShader == null)
            {
                m_segmentationShader = Shader.Find("Particles/Standard Unlit");
            }
            if (m_segmentationShader == null)
            {
                m_segmentationShader = Shader.Find("UI/Default");
            }

            Debug.Log($"DEBUG-INIT: Segmentation shader: {(m_segmentationShader != null ? m_segmentationShader.name : "NULL")}");
        }

        private void OnDestroy() => OVRManager.display.RecenteredPose -= CleanMarkersCallBack;

        private IEnumerator Start()
        {
            var sentisInference = FindAnyObjectByType<SentisInferenceRunManager>();
            while (!sentisInference.IsModelLoaded)
            {
                yield return null;
            }
            m_isSentisReady = true;
        }

        private void Update()
        {
            if (!m_isStarted)
            {
                if (m_cameraAccess.IsPlaying && m_isSentisReady)
                {
                    m_isStarted = true;
                }
            }
            else
            {
                // Press A button to spawn 3d markers and localize segmentations
                if (OVRInput.GetUp(m_actionButton) && m_delayPauseBackTime <= 0)
                {
                    SpwanCurrentDetectedObjects();
                }

                // Press B button to reset/clear all markers and segmentations
                if (OVRInput.GetUp(m_resetButton) && m_delayPauseBackTime <= 0)
                {
                    ResetAllMarkers();
                }

                
                // Check for Trigger press to toggle live UI
                if (OVRInput.GetDown(m_toggleUiButton))
                {
                    m_uiInference.ToggleLiveUiVisibility();
                }

                if (OVRInput.GetDown(m_toggleAutoUpdateButton))
                {
                    m_autoUpdateSegmentations = !m_autoUpdateSegmentations;
                    Debug.Log($"DEBUG-INPUT: Auto-Update Segmentations is now {(m_autoUpdateSegmentations ? "ON" : "OFF")}");
                }

                // Auto-update segmentations for objects currently in view
                if (m_autoUpdateSegmentations && m_enableSpatialSegmentation && !m_isPaused)
                {
                    m_autoUpdateTimer += Time.deltaTime;
                    if (m_autoUpdateTimer >= m_autoUpdateInterval)
                    {
                        m_autoUpdateTimer = 0f;
                        AutoUpdateSegmentationsInView();
                    }
                }

                // Cooldown for buttons after return from the pause menu
                m_delayPauseBackTime -= Time.deltaTime;
                if (m_delayPauseBackTime <= 0)
                {
                    m_delayPauseBackTime = 0;
                }
            }

            if (m_isPaused || !m_cameraAccess.IsPlaying)
            {
                if (m_isPaused)
                {
                    m_delayPauseBackTime = 0.1f;
                }
                return;
            }

            if (!m_runInference.IsRunning())
            {
                m_runInference.RunInference(m_cameraAccess);
            }
        }
        #endregion

        #region Marker Functions
        private void CleanMarkersCallBack()
        {
            foreach (var e in m_spwanedEntities)
            {
                Destroy(e, 0.1f);
            }
            m_spwanedEntities.Clear();

            foreach (var seg in m_spawnedSegmentations)
            {
                Destroy(seg, 0.1f);
            }
            m_spawnedSegmentations.Clear();

            OnObjectsIdentified?.Invoke(-1);
        }

        private void SpwanCurrentDetectedObjects()
        {
            var count = 0;
            foreach (var box in m_uiInference.BoxDrawn)
            {
                bool isNewMarker = PlaceMarkerUsingEnvironmentRaycast(box.WorldPos, box.ClassName);

                if (isNewMarker)
                {
                    count++;
                }

                // Spawn segmentation for new detections (even if marker already exists at that spot)
                if (m_enableSpatialSegmentation && box.SegmentationMask != null && box.WorldPos.HasValue)
                {
                    SpawnSpatialSegmentation(box);
                }
            }

            if (count > 0)
            {
                m_placeSound.Play();
            }
            OnObjectsIdentified?.Invoke(count);
        }

        private bool PlaceMarkerUsingEnvironmentRaycast(Vector3? position, string className)
        {
            if (!position.HasValue)
            {
                return false;
            }

            var existMarker = false;
            foreach (var e in m_spwanedEntities)
            {
                var markerClass = e.GetComponent<DetectionSpawnMarkerAnim>();
                if (markerClass)
                {
                    var dist = Vector3.Distance(e.transform.position, position.Value);
                    if (dist < m_spawnDistance && markerClass.GetYoloClassName() == className)
                    {
                        existMarker = true;
                        break;
                    }
                }
            }

            if (!existMarker)
            {
                var eMarker = Instantiate(m_spwanMarker);
                m_spwanedEntities.Add(eMarker);
                eMarker.transform.SetPositionAndRotation(position.Value, Quaternion.identity);
                eMarker.GetComponent<DetectionSpawnMarkerAnim>().SetYoloClassName(className);
            }

            return !existMarker;
        }

        private void SpawnSpatialSegmentation(SentisInferenceUiManager.BoundingBox box)
        {
            if (!box.WorldPos.HasValue || box.SegmentationMask == null)
                return;

            // Check for existing segmentation at this position - if found, REMOVE it so we can replace
            for (int i = m_spawnedSegmentations.Count - 1; i >= 0; i--)
            {
                var seg = m_spawnedSegmentations[i];
                if (seg != null)
                {
                    var billboard = seg.GetComponent<SpatialSegmentationBillboard>();
                    var dist = Vector3.Distance(seg.transform.position, box.WorldPos.Value);

                    // Same check as markers: distance AND class name must match
                    if (dist < m_spawnDistance && billboard != null && billboard.ClassName == box.ClassName)
                    {
                        // Remove old segmentation so we can replace it with updated one
                        Destroy(seg);
                        m_spawnedSegmentations.RemoveAt(i);
                        Debug.Log($"DEBUG-SPAWN: Replacing existing segmentation for '{box.ClassName}'");
                        break;
                    }
                }
            }

            // Get texture dimensions for aspect ratio
            Texture2D maskTex = box.SegmentationMask;
            float textureAspect = (float)maskTex.width / (float)maskTex.height;

            // Calculate world dimensions
            float worldWidth, worldHeight;

            if (box.EstimatedWorldWidth > 0.01f && box.EstimatedWorldHeight > 0.01f)
            {
                worldWidth = box.EstimatedWorldWidth;
                worldHeight = box.EstimatedWorldHeight;
            }
            else
            {
                OVRCameraRig cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (cameraRig != null)
                {
                    float distance = Vector3.Distance(box.CaptureCameraPose.position, box.WorldPos.Value);
                    float fovScale = distance * 1.2f;
                    worldWidth = box.NormalizedWidth * fovScale;
                    worldHeight = box.NormalizedHeight * fovScale;
                }
                else
                {
                    worldWidth = m_fallbackSegmentationSize * textureAspect;
                    worldHeight = m_fallbackSegmentationSize;
                }
            }

            worldWidth = Mathf.Max(worldWidth, 0.05f);
            worldHeight = Mathf.Max(worldHeight, 0.05f);

            GameObject segObj = CreateSegmentationMesh(maskTex, worldWidth, worldHeight, box.ClassName, box);

            if (segObj != null)
            {
                Vector3 placementPos;
                Quaternion placementRot;

                if (GetBestSurfaceNormal(box, out placementPos, out placementRot))
                {
                    segObj.transform.position = placementPos;
                    segObj.transform.rotation = placementRot;
                }
                else
                {
                    segObj.transform.position = box.WorldPos.Value;
                }

                segObj.transform.parent = null;

                var billboard = segObj.GetComponent<SpatialSegmentationBillboard>();
                if (billboard != null)
                {
                    billboard.Confidence = box.Confidence;
                    billboard.Coverage = box.Coverage;
                }

                m_spawnedSegmentations.Add(segObj);

                Debug.Log($"DEBUG-SPAWN: Spawned segmentation '{box.ClassName}' at {segObj.transform.position}, size: {worldWidth:F3}x{worldHeight:F3}m, distance-mode: {m_enableDistanceBasedRendering}");
            }
        }

        private bool GetBestSurfaceNormal(SentisInferenceUiManager.BoundingBox box, out Vector3 finalPosition, out Quaternion finalRotation)
        {
            finalPosition = Vector3.zero;
            finalRotation = Quaternion.identity;

            if (!box.WorldPos.HasValue) return false;

            finalPosition = box.WorldPos.Value;
            OVRCameraRig cam = FindFirstObjectByType<OVRCameraRig>();
            if (cam == null) return false;

            Vector3 toCamera = (cam.centerEyeAnchor.position - finalPosition).normalized;
            Vector3 assumedSurfaceNormal;

            if (Mathf.Abs(Vector3.Dot(toCamera, Vector3.up)) > 0.7f)
            {
                assumedSurfaceNormal = Vector3.up;
                finalRotation = Quaternion.FromToRotation(Vector3.forward, assumedSurfaceNormal);
            }
            else
            {
                Vector3 horizontalToCamera = new Vector3(toCamera.x, 0, toCamera.z).normalized;
                assumedSurfaceNormal = -horizontalToCamera;
                finalRotation = Quaternion.LookRotation(assumedSurfaceNormal, Vector3.up);
            }

            finalPosition += assumedSurfaceNormal * m_segmentationSurfaceOffset;

            return true;
        }

        private void AutoUpdateSegmentationsInView()
        {
            if (m_uiInference.BoxDrawn.Count == 0)
                return;

            foreach (var box in m_uiInference.BoxDrawn)
            {
                if (!box.WorldPos.HasValue || box.SegmentationMask == null)
                    continue;

                float newQualityScore = box.Confidence * 0.7f + box.Coverage * 0.3f;

                foreach (var seg in m_spawnedSegmentations)
                {
                    if (seg == null)
                        continue;

                    var billboard = seg.GetComponent<SpatialSegmentationBillboard>();
                    if (billboard == null || billboard.ClassName != box.ClassName)
                        continue;

                    var dist = Vector3.Distance(seg.transform.position, box.WorldPos.Value);
                    if (dist > m_spawnDistance * 2f)
                        continue;

                    float existingQualityScore = billboard.QualityScore;
                    float improvement = newQualityScore - existingQualityScore;

                    if (improvement > m_qualityImprovementThreshold)
                    {
                        Debug.Log($"DEBUG-AUTO-UPDATE: Auto-updating '{box.ClassName}': quality {existingQualityScore:F3} -> {newQualityScore:F3}");
                        SpawnSpatialSegmentationWithQuality(box, newQualityScore);
                    }

                    break;
                }
            }
        }

        private void SpawnSpatialSegmentationWithQuality(SentisInferenceUiManager.BoundingBox box, float qualityScore)
        {
            if (!box.WorldPos.HasValue || box.SegmentationMask == null)
                return;

            for (int i = m_spawnedSegmentations.Count - 1; i >= 0; i--)
            {
                var seg = m_spawnedSegmentations[i];
                if (seg != null)
                {
                    var billboard = seg.GetComponent<SpatialSegmentationBillboard>();
                    var dist = Vector3.Distance(seg.transform.position, box.WorldPos.Value);

                    if (dist < m_spawnDistance && billboard != null && billboard.ClassName == box.ClassName)
                    {
                        Destroy(seg);
                        m_spawnedSegmentations.RemoveAt(i);
                        break;
                    }
                }
            }

            Texture2D maskTex = box.SegmentationMask;
            float textureAspect = (float)maskTex.width / (float)maskTex.height;

            float worldWidth, worldHeight;

            if (box.EstimatedWorldWidth > 0.01f && box.EstimatedWorldHeight > 0.01f)
            {
                worldWidth = box.EstimatedWorldWidth;
                worldHeight = box.EstimatedWorldHeight;
            }
            else
            {
                OVRCameraRig cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (cameraRig != null)
                {
                    float distance = Vector3.Distance(box.CaptureCameraPose.position, box.WorldPos.Value);
                    float fovScale = distance * 1.2f;
                    worldWidth = box.NormalizedWidth * fovScale;
                    worldHeight = box.NormalizedHeight * fovScale;
                }
                else
                {
                    worldWidth = m_fallbackSegmentationSize * textureAspect;
                    worldHeight = m_fallbackSegmentationSize;
                }
            }

            worldWidth = Mathf.Max(worldWidth, 0.05f);
            worldHeight = Mathf.Max(worldHeight, 0.05f);

            GameObject segObj = CreateSegmentationMesh(maskTex, worldWidth, worldHeight, box.ClassName, box);

            if (segObj != null)
            {
                Vector3 placementPos;
                Quaternion placementRot;

                if (GetBestSurfaceNormal(box, out placementPos, out placementRot))
                {
                    segObj.transform.position = placementPos;
                    segObj.transform.rotation = placementRot;
                }
                else
                {
                    segObj.transform.position = box.WorldPos.Value;
                }

                segObj.transform.parent = null;

                var billboard = segObj.GetComponent<SpatialSegmentationBillboard>();
                if (billboard != null)
                {
                    billboard.Confidence = box.Confidence;
                    billboard.Coverage = box.Coverage;
                }

                m_spawnedSegmentations.Add(segObj);
            }
        }

        private GameObject CreateSegmentationMesh(Texture2D texture, float worldWidth, float worldHeight, string className, SentisInferenceUiManager.BoundingBox box)
        {
            GameObject obj = new GameObject($"SpatialSegmentation_{className}");
            obj.transform.parent = null;

            MeshFilter meshFilter = obj.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = obj.AddComponent<MeshRenderer>();

            Mesh mesh = new Mesh();

            float halfW = worldWidth / 2f;
            float halfH = worldHeight / 2f;

            Vector3[] vertices = new Vector3[]
            {
                new Vector3(-halfW, -halfH, 0),
                new Vector3( halfW, -halfH, 0),
                new Vector3(-halfW,  halfH, 0),
                new Vector3( halfW,  halfH, 0)
            };

            Vector2[] uvs = new Vector2[]
            {
                new Vector2(0, 0),
                new Vector2(1, 0),
                new Vector2(0, 1),
                new Vector2(1, 1)
            };

            int[] triangles = new int[]
            {
                0, 2, 1,
                2, 3, 1
            };

            mesh.vertices = vertices;
            mesh.uv = uvs;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            meshFilter.mesh = mesh;

            Material mat;
            if (m_segmentationShader != null)
            {
                mat = new Material(m_segmentationShader);
            }
            else
            {
                mat = new Material(Shader.Find("Sprites/Default"));
            }

            mat.mainTexture = texture;
            mat.color = Color.white;
            mat.renderQueue = 3000;

            if (mat.HasProperty("_Color"))
            {
                mat.SetColor("_Color", Color.white);
            }

            meshRenderer.material = mat;
            meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            meshRenderer.receiveShadows = false;

            // Add billboard behavior
            SpatialSegmentationBillboard billboard = obj.AddComponent<SpatialSegmentationBillboard>();
            billboard.ClassName = className;
            obj.transform.parent = null;

            Debug.Log($"DEBUG-MESH: [CREATE] Created segmentation mesh '{obj.name}' - Size: {worldWidth:F3}x{worldHeight:F3}m");

            // Add distance-based rendering if enabled
            if (m_enableDistanceBasedRendering)
            {
                Debug.Log($"DEBUG-OUTLINE: [SETUP] Distance-based rendering ENABLED for '{obj.name}'");
                Debug.Log($"DEBUG-OUTLINE: [SETUP] Settings - Threshold={m_outlineDistanceThreshold}m, BlendWidth={m_transitionBlendWidth}m, OutlinePixels={m_outlineWidthPixels}");
                Debug.Log($"DEBUG-OUTLINE: [SETUP] MaskCoefficients present: {box.MaskCoefficients != null} (length={box.MaskCoefficients?.Length ?? 0})");

                DistanceBasedSegmentation distanceRenderer = obj.AddComponent<DistanceBasedSegmentation>();
                Debug.Log($"DEBUG-OUTLINE: [SETUP] Added DistanceBasedSegmentation component");

                // Determine color for outline texture
                Color outlineColor;
                if (m_enableAdaptiveColorAnchored && texture != null)
                {
                    // Sample average color from the existing segmentation texture to determine background
                    // The texture already has the object's appearance baked in, so we sample it
                    Color avgBgColor = SampleAverageTextureColor(texture);
                    outlineColor = SegmentationColorContrast.GetBestContrastColor(avgBgColor, m_anchoredSegmentationAlpha);
                    Debug.Log($"DEBUG-COLOR: [ANCHORED] '{className}' - Adaptive color selected for outline");
                }
                else
                {
                    outlineColor = GetMaskColorForClass(className);
                }

                // Generate outline texture from existing solid texture
                Debug.Log($"DEBUG-OUTLINE: [TEXTURE-GEN] Generating outline from solid texture ({texture.width}x{texture.height})");
                Texture2D outlineTexture = GenerateOutlineFromTextureWithColor(texture, m_outlineWidthPixels, outlineColor);
                Debug.Log($"DEBUG-OUTLINE: [TEXTURE-GEN] Outline texture generated: {outlineTexture != null} ({outlineTexture?.width}x{outlineTexture?.height})");

                distanceRenderer.Initialize(texture, outlineTexture);
                distanceRenderer.SetDistanceParameters(m_outlineDistanceThreshold, m_transitionBlendWidth);
                Debug.Log($"DEBUG-OUTLINE: [SETUP] Initialization complete for '{obj.name}'");
            }
            else
            {
                Debug.Log($"DEBUG-OUTLINE: [SETUP] Distance-based rendering DISABLED for '{obj.name}'");
            }

            return obj;
        }

        /// <summary>
        /// Sample average color from a texture (for determining background color)
        /// </summary>
        private Color SampleAverageTextureColor(Texture2D texture)
        {
            if (texture == null) return Color.gray;

            Color[] pixels;
            try
            {
                pixels = texture.GetPixels();
            }
            catch
            {
                Debug.LogWarning("DEBUG-COLOR: [SAMPLE-TEX] Texture not readable");
                return Color.gray;
            }

            float r = 0, g = 0, b = 0;
            int count = 0;

            // Sample every Nth pixel for performance
            int step = Mathf.Max(1, pixels.Length / 64);
            for (int i = 0; i < pixels.Length; i += step)
            {
                // Only count non-transparent pixels
                if (pixels[i].a > 0.1f)
                {
                    r += pixels[i].r;
                    g += pixels[i].g;
                    b += pixels[i].b;
                    count++;
                }
            }

            if (count == 0) return Color.gray;
            return new Color(r / count, g / count, b / count);
        }

        /// <summary>
        /// Generate outline texture from existing solid texture using edge detection
        /// </summary>
        private Texture2D GenerateOutlineFromTexture(Texture2D source, int outlineWidth)
        {
            return GenerateOutlineFromTextureWithColor(source, outlineWidth, Color.white);
        }

        /// <summary>
        /// Generate outline texture with custom color
        /// </summary>
        private Texture2D GenerateOutlineFromTextureWithColor(Texture2D source, int outlineWidth, Color outlineColor)
        {
            Debug.Log($"DEBUG-OUTLINE: [EDGE-DETECT] Starting edge detection - Source: {source.width}x{source.height}, OutlineWidth: {outlineWidth}px, Color: ({outlineColor.r:F2},{outlineColor.g:F2},{outlineColor.b:F2})");

            int width = source.width;
            int height = source.height;

            Color[] sourcePixels = source.GetPixels();
            Color[] outlinePixels = new Color[width * height];

            // Initialize all to transparent
            for (int i = 0; i < outlinePixels.Length; i++)
            {
                outlinePixels[i] = Color.clear;
            }

            int opaquePixelCount = 0;
            int edgePixelCount = 0;

            // Edge detection: pixel is edge if it's opaque and has transparent neighbor within outlineWidth
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    Color pixel = sourcePixels[idx];

                    // Skip transparent pixels
                    if (pixel.a < 0.1f)
                        continue;

                    opaquePixelCount++;

                    // Check if near edge
                    bool isEdge = false;

                    for (int dy = -outlineWidth; dy <= outlineWidth && !isEdge; dy++)
                    {
                        for (int dx = -outlineWidth; dx <= outlineWidth && !isEdge; dx++)
                        {
                            // Circular kernel
                            if (dx * dx + dy * dy > outlineWidth * outlineWidth)
                                continue;

                            int nx = x + dx;
                            int ny = y + dy;

                            // Image boundary counts as edge
                            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                            {
                                isEdge = true;
                                continue;
                            }

                            int nidx = ny * width + nx;
                            if (sourcePixels[nidx].a < 0.1f)
                            {
                                isEdge = true;
                            }
                        }
                    }

                    if (isEdge)
                    {
                        // Use the provided outline color with the original alpha
                        outlinePixels[idx] = new Color(outlineColor.r, outlineColor.g, outlineColor.b, outlineColor.a);
                        edgePixelCount++;
                    }
                }
            }

            Debug.Log($"DEBUG-OUTLINE: [EDGE-DETECT] Results - OpaquePixels: {opaquePixelCount}, EdgePixels: {edgePixelCount}, Ratio: {(opaquePixelCount > 0 ? (float)edgePixelCount / opaquePixelCount * 100f : 0):F1}%");

            Texture2D outlineTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
            outlineTexture.SetPixels(outlinePixels);
            outlineTexture.Apply();
            outlineTexture.filterMode = FilterMode.Bilinear;

            Debug.Log($"DEBUG-OUTLINE: [EDGE-DETECT] Outline texture created successfully with custom color");

            return outlineTexture;
        }

        private Color GetMaskColorForClass(string className)
        {
            // Default color scheme - can be expanded
            return new Color(0, 1, 0, 0.5f); // Green with transparency
        }
        #endregion

        #region Public Functions
        public void OnPause(bool pause)
        {
            m_isPaused = pause;
        }

        public void ResetAllMarkers()
        {
            foreach (var e in m_spwanedEntities)
            {
                if (e != null)
                {
                    Destroy(e);
                }
            }
            m_spwanedEntities.Clear();

            foreach (var seg in m_spawnedSegmentations)
            {
                if (seg != null)
                {
                    Destroy(seg);
                }
            }
            m_spawnedSegmentations.Clear();

            Debug.Log("DEBUG-RESET: Reset all markers and segmentations");
            OnObjectsIdentified?.Invoke(-1);
        }

        public void ClearSegmentations()
        {
            foreach (var seg in m_spawnedSegmentations)
            {
                if (seg != null)
                {
                    Destroy(seg);
                }
            }
            m_spawnedSegmentations.Clear();
        }

        /// <summary>
        /// Check if there's an anchored segmentation at the given position for the specified class
        /// Used by SentisInferenceUiManager to hide live segmentations when anchored version exists
        /// </summary>
        public bool HasAnchoredSegmentationAt(Vector3 worldPos, string className, float maxDistance)
        {
            foreach (var seg in m_spawnedSegmentations)
            {
                if (seg == null)
                    continue;

                var billboard = seg.GetComponent<SpatialSegmentationBillboard>();
                if (billboard == null)
                    continue;

                // Check class name matches
                if (billboard.ClassName != className)
                    continue;

                // Check distance
                float dist = Vector3.Distance(seg.transform.position, worldPos);
                if (dist <= maxDistance)
                {
                    Debug.Log($"DEBUG-ANCHOR-FILTER: Found anchored '{className}' at distance {dist:F3}m (threshold: {maxDistance:F3}m)");
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Update distance threshold at runtime
        /// </summary>
        public void SetOutlineDistanceThreshold(float distance)
        {
            m_outlineDistanceThreshold = Mathf.Max(0.1f, distance);

            // Update all existing segmentations
            foreach (var seg in m_spawnedSegmentations)
            {
                if (seg != null)
                {
                    var distanceRenderer = seg.GetComponent<DistanceBasedSegmentation>();
                    if (distanceRenderer != null)
                    {
                        distanceRenderer.SetDistanceParameters(m_outlineDistanceThreshold, m_transitionBlendWidth);
                    }
                }
            }
        }

        /// <summary>
        /// Update outline width at runtime (requires re-generating textures)
        /// </summary>
        public void SetOutlineWidth(int widthPixels)
        {
            m_outlineWidthPixels = Mathf.Clamp(widthPixels, 1, 15);
            // Note: Existing segmentations won't update - only new ones will use the new width
        }
        #endregion
    }

    /// <summary>
    /// Billboard component for spatial segmentation - keeps it facing the camera
    /// Also tracks quality metrics for smart auto-update
    /// </summary>
    public class SpatialSegmentationBillboard : MonoBehaviour
    {
        public string ClassName { get; set; }

        public float Confidence { get; set; }
        public float Coverage { get; set; }
        public float QualityScore => Confidence * 0.7f + Coverage * 0.3f;

        [Tooltip("If true, billboard fully faces camera. If false, only rotates on Y axis (stays upright)")]
        public bool FullBillboard = true;

        private OVRCameraRig m_cameraRig;

        private void Update()
        {
            if (m_cameraRig == null)
            {
                m_cameraRig = FindFirstObjectByType<OVRCameraRig>();
                return;
            }
        }
    }
}