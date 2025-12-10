// Distance-based segmentation rendering
// Switches between solid fill (far) and outline-only (near) based on user proximity

using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    /// <summary>
    /// Attach to segmentation mesh objects to enable distance-based rendering.
    /// When user is far: shows solid fill segmentation
    /// When user is near: shows outline-only segmentation
    /// </summary>
    public class DistanceBasedSegmentation : MonoBehaviour
    {
        [Header("Distance Thresholds")]
        [Tooltip("Distance (meters) at which segmentation switches to outline mode")]
        [SerializeField] private float m_outlineDistanceThreshold = 1.0f;

        [Tooltip("Blend zone width for smooth transition (meters)")]
        [SerializeField] private float m_transitionBlendWidth = 0.2f;

        [Header("Outline Settings")]
        [Tooltip("Width of outline border in mask-space pixels (1-10 recommended)")]
        [SerializeField] private int m_outlineWidthPixels = 3;

        [Header("Debug")]
        [SerializeField] private bool m_showDebugInfo = false;

        // Textures
        private Texture2D m_solidTexture;
        private Texture2D m_outlineTexture;

        // Current state
        private MeshRenderer m_meshRenderer;
        private Material m_material;
        private OVRCameraRig m_cameraRig;
        private SegmentationRenderMode m_currentMode = SegmentationRenderMode.SolidFill;

        // For smooth blending (optional advanced feature)
        private float m_currentBlendFactor = 0f; // 0 = solid, 1 = outline

        #region Properties

        /// <summary>
        /// Distance at which segmentation switches to outline mode
        /// </summary>
        public float OutlineDistanceThreshold
        {
            get => m_outlineDistanceThreshold;
            set => m_outlineDistanceThreshold = Mathf.Max(0.1f, value);
        }

        /// <summary>
        /// Width of outline in mask pixels
        /// </summary>
        public int OutlineWidthPixels
        {
            get => m_outlineWidthPixels;
            set => m_outlineWidthPixels = Mathf.Clamp(value, 1, 15);
        }

        /// <summary>
        /// Current render mode
        /// </summary>
        public SegmentationRenderMode CurrentMode => m_currentMode;

        /// <summary>
        /// Current distance to user
        /// </summary>
        public float CurrentDistance { get; private set; }

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            m_meshRenderer = GetComponent<MeshRenderer>();
            if (m_meshRenderer != null)
            {
                m_material = m_meshRenderer.material;
            }
        }

        private void Update()
        {
            UpdateDistanceBasedRendering();
        }

        private void Start()
        {
            Debug.Log($"DEBUG-OUTLINE: [INIT] DistanceBasedSegmentation started on '{gameObject.name}'");
            Debug.Log($"DEBUG-OUTLINE: [INIT] Threshold={m_outlineDistanceThreshold}m, BlendWidth={m_transitionBlendWidth}m, OutlinePixels={m_outlineWidthPixels}");
            Debug.Log($"DEBUG-OUTLINE: [INIT] SolidTexture={m_solidTexture != null}, OutlineTexture={m_outlineTexture != null}");
            Debug.Log($"DEBUG-OUTLINE: [INIT] MeshRenderer={m_meshRenderer != null}, Material={m_material != null}");
        }

        private void OnDestroy()
        {
            // Clean up textures we created
            if (m_solidTexture != null)
            {
                Destroy(m_solidTexture);
            }
            if (m_outlineTexture != null)
            {
                Destroy(m_outlineTexture);
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize with pre-generated textures (most efficient)
        /// </summary>
        public void Initialize(Texture2D solidTexture, Texture2D outlineTexture)
        {
            m_solidTexture = solidTexture;
            m_outlineTexture = outlineTexture;

            Debug.Log($"DEBUG-OUTLINE: [INITIALIZE] Received textures - Solid: {solidTexture != null} ({solidTexture?.width}x{solidTexture?.height}), Outline: {outlineTexture != null} ({outlineTexture?.width}x{outlineTexture?.height})");

            // Start with solid (far mode)
            SetRenderMode(SegmentationRenderMode.SolidFill);
            Debug.Log($"DEBUG-OUTLINE: [INITIALIZE] Initial mode set to SolidFill");
        }

        /// <summary>
        /// Initialize by generating textures from mask data
        /// </summary>
        public void InitializeFromMask(
            float[,] fullMask,
            Color maskColor,
            float boxCenterX,
            float boxCenterY,
            float boxWidth,
            float boxHeight,
            float imageWidth,
            float imageHeight)
        {
            SegmentationProcessor.GenerateBothTextures(
                fullMask,
                maskColor,
                boxCenterX, boxCenterY,
                boxWidth, boxHeight,
                imageWidth, imageHeight,
                m_outlineWidthPixels,
                out m_solidTexture,
                out m_outlineTexture);

            SetRenderMode(SegmentationRenderMode.SolidFill);
        }

        /// <summary>
        /// Configure distance parameters at runtime
        /// </summary>
        public void SetDistanceParameters(float outlineThreshold, float blendWidth = 0.2f)
        {
            float oldThreshold = m_outlineDistanceThreshold;
            float oldBlend = m_transitionBlendWidth;

            m_outlineDistanceThreshold = Mathf.Max(0.1f, outlineThreshold);
            m_transitionBlendWidth = Mathf.Max(0f, blendWidth);

            Debug.Log($"DEBUG-OUTLINE: [SET-PARAMS] Distance params changed: Threshold {oldThreshold}m -> {m_outlineDistanceThreshold}m, Blend {oldBlend}m -> {m_transitionBlendWidth}m");
        }

        /// <summary>
        /// Force a specific render mode (bypasses distance check)
        /// </summary>
        public void ForceRenderMode(SegmentationRenderMode mode)
        {
            SetRenderMode(mode);
        }

        #endregion

        #region Private Methods

        private void UpdateDistanceBasedRendering()
        {
            if (m_material == null || m_solidTexture == null || m_outlineTexture == null)
            {
                // Only log once per second to avoid spam
                if (Time.frameCount % 60 == 0)
                {
                    Debug.LogWarning($"DEBUG-OUTLINE: [UPDATE-FAIL] '{gameObject.name}' - Missing components: Material={m_material != null}, SolidTex={m_solidTexture != null}, OutlineTex={m_outlineTexture != null}");
                }
                return;
            }

            // Get camera reference
            if (m_cameraRig == null)
            {
                m_cameraRig = FindFirstObjectByType<OVRCameraRig>();
                if (m_cameraRig == null)
                {
                    if (Time.frameCount % 60 == 0)
                    {
                        Debug.LogWarning($"DEBUG-OUTLINE: [UPDATE-FAIL] '{gameObject.name}' - OVRCameraRig not found!");
                    }
                    return;
                }
                Debug.Log($"DEBUG-OUTLINE: [UPDATE] Found OVRCameraRig: {m_cameraRig.name}");
            }

            // Calculate distance to user
            Vector3 userPosition = m_cameraRig.centerEyeAnchor.position;
            Vector3 objectPosition = transform.position;
            CurrentDistance = Vector3.Distance(objectPosition, userPosition);

            // Log distance every 30 frames (roughly 0.5 sec at 60fps)
            if (Time.frameCount % 30 == 0)
            {
                Debug.Log($"DEBUG-OUTLINE: [DISTANCE] '{gameObject.name}' - UserPos={userPosition}, ObjPos={objectPosition}, Distance={CurrentDistance:F3}m, Threshold={m_outlineDistanceThreshold:F3}m");
            }

            // Determine target mode based on distance
            SegmentationRenderMode targetMode;

            if (m_transitionBlendWidth > 0.001f)
            {
                // Smooth transition with blend zone
                float nearEdge = m_outlineDistanceThreshold - m_transitionBlendWidth / 2f;
                float farEdge = m_outlineDistanceThreshold + m_transitionBlendWidth / 2f;

                if (CurrentDistance <= nearEdge)
                {
                    targetMode = SegmentationRenderMode.OutlineOnly;
                    m_currentBlendFactor = 1f;
                }
                else if (CurrentDistance >= farEdge)
                {
                    targetMode = SegmentationRenderMode.SolidFill;
                    m_currentBlendFactor = 0f;
                }
                else
                {
                    // In blend zone - use solid for now (could implement alpha blending)
                    m_currentBlendFactor = 1f - (CurrentDistance - nearEdge) / m_transitionBlendWidth;
                    targetMode = m_currentBlendFactor > 0.5f ? SegmentationRenderMode.OutlineOnly : SegmentationRenderMode.SolidFill;
                }

                if (Time.frameCount % 30 == 0)
                {
                    Debug.Log($"DEBUG-OUTLINE: [BLEND-CALC] NearEdge={nearEdge:F3}m, FarEdge={farEdge:F3}m, BlendFactor={m_currentBlendFactor:F3}, TargetMode={targetMode}");
                }
            }
            else
            {
                // Hard switch at threshold
                targetMode = CurrentDistance <= m_outlineDistanceThreshold
                    ? SegmentationRenderMode.OutlineOnly
                    : SegmentationRenderMode.SolidFill;

                if (Time.frameCount % 30 == 0)
                {
                    Debug.Log($"DEBUG-OUTLINE: [HARD-SWITCH] Distance={CurrentDistance:F3}m <= Threshold={m_outlineDistanceThreshold:F3}m ? -> TargetMode={targetMode}");
                }
            }

            // Apply mode if changed
            if (targetMode != m_currentMode)
            {
                Debug.Log($"DEBUG-OUTLINE: [MODE-CHANGE] '{gameObject.name}' switching from {m_currentMode} to {targetMode} at distance {CurrentDistance:F3}m (threshold={m_outlineDistanceThreshold:F3}m)");
                SetRenderMode(targetMode);
            }
        }

        private void SetRenderMode(SegmentationRenderMode mode)
        {
            SegmentationRenderMode previousMode = m_currentMode;
            m_currentMode = mode;

            if (m_material == null)
            {
                Debug.LogError($"DEBUG-OUTLINE: [SET-MODE-FAIL] '{gameObject.name}' - Material is null, cannot set mode!");
                return;
            }

            Texture2D targetTexture = mode == SegmentationRenderMode.SolidFill
                ? m_solidTexture
                : m_outlineTexture;

            if (targetTexture != null)
            {
                m_material.mainTexture = targetTexture;
                Debug.Log($"DEBUG-OUTLINE: [SET-MODE] '{gameObject.name}' - Mode: {previousMode} -> {mode}, Texture: {targetTexture.name} ({targetTexture.width}x{targetTexture.height})");
            }
            else
            {
                Debug.LogError($"DEBUG-OUTLINE: [SET-MODE-FAIL] '{gameObject.name}' - Target texture for mode {mode} is null! Solid={m_solidTexture != null}, Outline={m_outlineTexture != null}");
            }
        }

        #endregion

        #region Editor / Debug

        private void OnDrawGizmosSelected()
        {
            if (!m_showDebugInfo)
                return;

            // Draw outline threshold sphere
            Gizmos.color = new Color(0f, 1f, 0f, 0.3f);
            Gizmos.DrawWireSphere(transform.position, m_outlineDistanceThreshold);

            // Draw blend zone
            if (m_transitionBlendWidth > 0.001f)
            {
                Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
                Gizmos.DrawWireSphere(transform.position, m_outlineDistanceThreshold - m_transitionBlendWidth / 2f);
                Gizmos.DrawWireSphere(transform.position, m_outlineDistanceThreshold + m_transitionBlendWidth / 2f);
            }
        }

        #endregion
    }
}