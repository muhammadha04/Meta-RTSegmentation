// Copyright (c) Meta Platforms, Inc. and affiliates.
// Spatially anchored segmentation mask that billboards towards the camera

using Meta.XR.Samples;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SegmentationSpatialMarker : MonoBehaviour
    {
        [SerializeField] private MeshRenderer m_quadRenderer;
        [SerializeField] private float m_fadeInDuration = 0.3f;
        [SerializeField] private float m_offsetFromSurface = 0.02f; // Small offset to prevent z-fighting
        
        private OVRCameraRig m_camera;
        private string m_className;
        private Material m_material;
        private float m_fadeTimer = 0f;
        private float m_targetAlpha = 0.6f;
        private Vector3 m_surfaceNormal = Vector3.forward;
        
        private void Awake()
        {
            // Create material instance for this marker
            if (m_quadRenderer != null)
            {
                m_material = new Material(Shader.Find("Unlit/Transparent"));
                m_material.renderQueue = 3000; // Transparent queue
                m_quadRenderer.material = m_material;
            }
        }

        private void Update()
        {
            // Fade in effect
            if (m_fadeTimer < m_fadeInDuration)
            {
                m_fadeTimer += Time.deltaTime;
                float alpha = Mathf.Lerp(0f, m_targetAlpha, m_fadeTimer / m_fadeInDuration);
                if (m_material != null)
                {
                    Color c = m_material.color;
                    c.a = alpha;
                    m_material.color = c;
                }
            }

            // Billboard towards camera while respecting surface normal
            if (!m_camera)
            {
                m_camera = FindFirstObjectByType<OVRCameraRig>();
            }
            else
            {
                // Get direction to camera
                Vector3 toCamera = m_camera.centerEyeAnchor.position - transform.position;
                
                // Project onto plane perpendicular to surface normal for constrained billboard
                // Or use full billboard if no surface normal constraint
                Vector3 flattenedDir = toCamera;
                flattenedDir.y = 0; // Keep upright
                
                if (flattenedDir != Vector3.zero)
                {
                    // Look at camera but stay upright
                    transform.rotation = Quaternion.LookRotation(-flattenedDir, Vector3.up);
                }
            }
        }

        private void OnDestroy()
        {
            if (m_material != null)
            {
                Destroy(m_material);
            }
        }

        /// <summary>
        /// Initialize the segmentation marker with a mask texture and world-space dimensions
        /// </summary>
        public void Initialize(Texture2D maskTexture, string className, Vector3 worldSize, Vector3 surfaceNormal)
        {
            m_className = className;
            m_surfaceNormal = surfaceNormal.normalized;
            
            if (m_material != null && maskTexture != null)
            {
                m_material.mainTexture = maskTexture;
                Color c = m_material.color;
                c.a = 0f; // Start invisible for fade-in
                m_material.color = c;
            }

            // Scale the quad to match the world-space size of the detected object
            transform.localScale = new Vector3(worldSize.x, worldSize.y, 1f);
            
            // Offset slightly from surface to prevent z-fighting
            transform.position += m_surfaceNormal * m_offsetFromSurface;
        }

        /// <summary>
        /// Simple initialization with just texture and size
        /// </summary>
        public void Initialize(Texture2D maskTexture, string className, float worldWidth, float worldHeight)
        {
            Initialize(maskTexture, className, new Vector3(worldWidth, worldHeight, 1f), Vector3.forward);
        }

        public string GetClassName()
        {
            return m_className;
        }
    }
}
