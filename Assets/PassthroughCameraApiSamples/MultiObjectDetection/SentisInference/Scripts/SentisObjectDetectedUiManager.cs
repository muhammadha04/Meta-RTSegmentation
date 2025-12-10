// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Collections;
using Meta.XR;
using Meta.XR.Samples;
using UnityEngine;
namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisObjectDetectedUiManager : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;
        [SerializeField] private GameObject m_detectionCanvas;
        [SerializeField] private float m_canvasDistance = 1f;
        private Pose m_captureCameraPose;
        private Vector3 m_capturePosition;
        private Quaternion m_captureRotation;
        private IEnumerator Start()
        {
            if (m_cameraAccess == null)
            {
                Debug.LogError($"PCA: {nameof(m_cameraAccess)} field is required "
                            + $"for the component {nameof(SentisObjectDetectedUiManager)} to operate properly");
                enabled = false;
                yield break;
            }
            // CRITICAL FIX: Detach canvas from camera so it doesn't follow headset
            Canvas canvas = m_detectionCanvas.GetComponent<Canvas>();
            if (canvas != null)
            {
                canvas.worldCamera = null;  // Remove camera reference
                Debug.Log("Canvas camera reference removed - canvas is now independent");
            }
            while (!m_cameraAccess.IsPlaying)
            {
                yield return null;
            }
            var cameraCanvasRectTransform = m_detectionCanvas.GetComponentInChildren<RectTransform>();
            var leftSidePointInCamera = m_cameraAccess.ViewportPointToRay(new Vector2(0f, 0.5f));
            var rightSidePointInCamera = m_cameraAccess.ViewportPointToRay(new Vector2(1f, 0.5f));
            var horizontalFoVDegrees = Vector3.Angle(leftSidePointInCamera.direction, rightSidePointInCamera.direction);
            var horizontalFoVRadians = horizontalFoVDegrees / 180 * Math.PI;
            var newCanvasWidthInMeters = 2 * m_canvasDistance * Math.Tan(horizontalFoVRadians / 2);
            var localScale = (float)(newCanvasWidthInMeters / cameraCanvasRectTransform.sizeDelta.x);
            cameraCanvasRectTransform.localScale = new Vector3(localScale, localScale, localScale);
        }
        public void UpdatePosition()
        {
            // Position the canvas in front of the camera
            m_detectionCanvas.transform.position = m_capturePosition;
            m_detectionCanvas.transform.rotation = m_captureRotation;
        }
        public void CapturePosition()
        {
            // 1. Capture the camera pose
            m_captureCameraPose = m_cameraAccess.GetCameraPose();
            // 2. Position canvas directly in front of camera (in camera's local space)
            m_capturePosition = m_captureCameraPose.position +
                               m_captureCameraPose.rotation * Vector3.forward * m_canvasDistance;
            // 3. Canvas rotation should exactly match camera rotation when captured
            m_captureRotation = m_captureCameraPose.rotation;
        }
        public Vector3 GetCapturedCameraPosition()
        {
            return m_captureCameraPose.position;
        }
    }
}