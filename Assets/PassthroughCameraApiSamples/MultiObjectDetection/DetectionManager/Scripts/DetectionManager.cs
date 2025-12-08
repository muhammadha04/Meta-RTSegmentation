// Copyright (c) Meta Platforms, Inc. and affiliates.
// Updated with proper spatial segmentation support

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

            Debug.Log($"Segmentation shader: {(m_segmentationShader != null ? m_segmentationShader.name : "NULL")}");
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
                        Debug.Log($"Replacing existing segmentation for '{box.ClassName}'");
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
                // Use raycast-estimated dimensions
                worldWidth = box.EstimatedWorldWidth;
                worldHeight = box.EstimatedWorldHeight;
            }
            else
            {
                // Fallback: estimate from distance to camera and normalized dimensions
                OVRCameraRig cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (cameraRig != null)
                {
                    float distance = Vector3.Distance(cameraRig.centerEyeAnchor.position, box.WorldPos.Value);
                    // Approximate FOV-based scaling
                    float fovScale = distance * 1.2f; // Rough approximation for ~90 degree FOV
                    worldWidth = box.NormalizedWidth * fovScale;
                    worldHeight = box.NormalizedHeight * fovScale;
                }
                else
                {
                    worldWidth = m_fallbackSegmentationSize * textureAspect;
                    worldHeight = m_fallbackSegmentationSize;
                }
            }

            // Ensure minimum size
            worldWidth = Mathf.Max(worldWidth, 0.05f);
            worldHeight = Mathf.Max(worldHeight, 0.05f);

            // Create the segmentation object
            GameObject segObj = CreateSegmentationMesh(maskTex, worldWidth, worldHeight, box.ClassName);

            if (segObj != null)
            {
                // Position at world location with small offset
                Vector3 position = box.WorldPos.Value;

                // Get direction from object to camera for offset
                OVRCameraRig cam = FindFirstObjectByType<OVRCameraRig>();
                if (cam != null)
                {
                    Vector3 toCamera = (cam.centerEyeAnchor.position - position).normalized;
                    position += toCamera * m_segmentationSurfaceOffset;
                }
                // Get direction from object to camera for offset AND orientation
                cam = FindFirstObjectByType<OVRCameraRig>();
                if (cam != null)
                {
                    Vector3 toCamera = (cam.centerEyeAnchor.position - position).normalized;

                    // Orient the segmentation to face the camera (but locked, not billboard)
                    // The quad's forward (Z) should point toward the camera
                    segObj.transform.rotation = Quaternion.LookRotation(toCamera, Vector3.up);

                    // Apply offset AFTER rotation
                    position += toCamera * m_segmentationSurfaceOffset;
                    segObj.transform.position = position;
                }
                else
                {
                    segObj.transform.position = position;
                }

                // Store quality metrics on the billboard component
                var billboard = segObj.GetComponent<SpatialSegmentationBillboard>();
                if (billboard != null)
                {
                    billboard.Confidence = box.Confidence;
                    billboard.Coverage = box.Coverage;
                }

                m_spawnedSegmentations.Add(segObj);

                Debug.Log($"Spawned segmentation '{box.ClassName}' at {position}, size: {worldWidth:F3}x{worldHeight:F3}m, quality: {billboard?.QualityScore:F3}");
            }
        }

        /// <summary>
        /// Auto-update segmentations for objects currently being detected (in view)
        /// Only updates if the new detection has better quality (confidence + coverage)
        /// </summary>
        private void AutoUpdateSegmentationsInView()
        {
            if (m_uiInference.BoxDrawn.Count == 0)
                return;

            // For each currently detected object, check if we should update its segmentation
            foreach (var box in m_uiInference.BoxDrawn)
            {
                if (!box.WorldPos.HasValue || box.SegmentationMask == null)
                    continue;

                // Calculate quality score for new detection
                float newQualityScore = box.Confidence * 0.7f + box.Coverage * 0.3f;

                // Check if we have an existing segmentation for this object
                foreach (var seg in m_spawnedSegmentations)
                {
                    if (seg == null)
                        continue;

                    var billboard = seg.GetComponent<SpatialSegmentationBillboard>();
                    if (billboard == null || billboard.ClassName != box.ClassName)
                        continue;

                    // Found existing segmentation for this class - check if it's nearby
                    var dist = Vector3.Distance(seg.transform.position, box.WorldPos.Value);
                    if (dist > m_spawnDistance * 2f)
                        continue;

                    // Compare quality scores - only update if significantly better
                    float existingQualityScore = billboard.QualityScore;
                    float improvement = newQualityScore - existingQualityScore;

                    if (improvement > m_qualityImprovementThreshold)
                    {
                        Debug.Log($"Auto-updating '{box.ClassName}': quality {existingQualityScore:F3} -> {newQualityScore:F3} (improvement: {improvement:F3})");
                        SpawnSpatialSegmentationWithQuality(box, newQualityScore);
                    }

                    break; // Found matching segmentation, move to next box
                }
            }
        }

        /// <summary>
        /// Spawn segmentation and store quality metrics for future comparison
        /// </summary>
        private void SpawnSpatialSegmentationWithQuality(SentisInferenceUiManager.BoundingBox box, float qualityScore)
        {
            if (!box.WorldPos.HasValue || box.SegmentationMask == null)
                return;

            // Remove existing segmentation for this object
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
                    float distance = Vector3.Distance(cameraRig.centerEyeAnchor.position, box.WorldPos.Value);
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

            GameObject segObj = CreateSegmentationMesh(maskTex, worldWidth, worldHeight, box.ClassName);

            if (segObj != null)
            {
                Vector3 position = box.WorldPos.Value;

                // Get direction from object to camera for offset AND orientation
                OVRCameraRig cam = FindFirstObjectByType<OVRCameraRig>();
                if (cam != null)
                {
                    Vector3 toCamera = (cam.centerEyeAnchor.position - position).normalized;

                    // Orient the segmentation to face the camera (but locked, not billboard)
                    segObj.transform.rotation = Quaternion.LookRotation(toCamera, Vector3.up);

                    // Apply offset AFTER rotation
                    position += toCamera * m_segmentationSurfaceOffset;
                }

                segObj.transform.position = position;

                // Store quality metrics on the billboard component
                var billboardComponent = segObj.GetComponent<SpatialSegmentationBillboard>();
                if (billboardComponent != null)
                {
                    billboardComponent.Confidence = box.Confidence;
                    billboardComponent.Coverage = box.Coverage;
                }

                m_spawnedSegmentations.Add(segObj);
            }
        }
        private GameObject CreateSegmentationMesh(Texture2D texture, float worldWidth, float worldHeight, string className)
        {
            GameObject obj = new GameObject($"SpatialSegmentation_{className}");

            // Create mesh (simple quad)
            MeshFilter meshFilter = obj.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = obj.AddComponent<MeshRenderer>();

            // Create quad mesh with correct dimensions
            Mesh mesh = new Mesh();

            float halfW = worldWidth / 2f;
            float halfH = worldHeight / 2f;

            // Vertices - quad centered at origin
            Vector3[] vertices = new Vector3[]
            {
                new Vector3(-halfW, -halfH, 0), // bottom-left
                new Vector3( halfW, -halfH, 0), // bottom-right
                new Vector3(-halfW,  halfH, 0), // top-left
                new Vector3( halfW,  halfH, 0)  // top-right
            };

            // UVs
            Vector2[] uvs = new Vector2[]
            {
                new Vector2(0, 0),
                new Vector2(1, 0),
                new Vector2(0, 1),
                new Vector2(1, 1)
            };

            // Triangles (two triangles for quad)
            int[] triangles = new int[]
            {
                0, 2, 1,  // first triangle
                2, 3, 1   // second triangle
            };

            mesh.vertices = vertices;
            mesh.uv = uvs;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            meshFilter.mesh = mesh;

            // Create material
            Material mat;
            if (m_segmentationShader != null)
            {
                mat = new Material(m_segmentationShader);
            }
            else
            {
                // Last resort fallback
                mat = new Material(Shader.Find("Sprites/Default"));
            }

            mat.mainTexture = texture;
            mat.color = Color.white;
            mat.renderQueue = 3000; // Transparent

            // Try to enable transparency
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

            return obj;
        }
        #endregion

        #region Public Functions
        public void OnPause(bool pause)
        {
            m_isPaused = pause;
        }

        /// <summary>
        /// Reset all markers and segmentations - called when pressing B
        /// </summary>
        public void ResetAllMarkers()
        {
            // Clear object markers
            foreach (var e in m_spwanedEntities)
            {
                if (e != null)
                {
                    Destroy(e);
                }
            }
            m_spwanedEntities.Clear();

            // Clear segmentations
            foreach (var seg in m_spawnedSegmentations)
            {
                if (seg != null)
                {
                    Destroy(seg);
                }
            }
            m_spawnedSegmentations.Clear();

            Debug.Log("Reset all markers and segmentations");
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
        #endregion
    }

    /// <summary>
    /// Billboard component for spatial segmentation - keeps it facing the camera
    /// Also tracks quality metrics for smart auto-update
    /// </summary>
    public class SpatialSegmentationBillboard : MonoBehaviour
    {
        public string ClassName { get; set; }

        // Quality metrics for smart auto-update
        public float Confidence { get; set; }
        public float Coverage { get; set; }  // Normalized width * height (how much of view the object covers)
        public float QualityScore => Confidence * 0.7f + Coverage * 0.3f; // Combined quality metric

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

            Vector3 cameraPos = m_cameraRig.centerEyeAnchor.position;

            //if (FullBillboard)
            //{
            //    // Full billboard - always face camera directly (like a sprite)
            //    // This handles sitting/standing and looking from any angle
            //    transform.LookAt(cameraPos);
            //    // Flip 180 degrees because LookAt makes Z point AT camera, but we want to FACE camera
            //    transform.Rotate(0, 180, 0);
            //}
            //else
            //{
            //    // Y-axis only billboard - stays upright
            //    Vector3 toCamera = cameraPos - transform.position;
            //    toCamera.y = 0;

            //    if (toCamera.sqrMagnitude > 0.001f)
            //    {
            //        transform.rotation = Quaternion.LookRotation(-toCamera.normalized, Vector3.up);
            //    }
            //}
        }
    }
}