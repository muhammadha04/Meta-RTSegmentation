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

            // 2. Calculate the position (correctly uses full camera rotation)
            m_capturePosition = m_captureCameraPose.position + m_captureCameraPose.rotation * Vector3.forward * m_canvasDistance;

            // 3. Calculate the forward direction from the camera to the canvas position
            Vector3 forwardDirection = m_capturePosition - m_captureCameraPose.position;

            // 4. CRITICAL FIX FOR TILT: Flatten the direction vector to the world X-Z plane.
            // This removes the camera's pitch (X) component from the rotation calculation.
            forwardDirection.y = 0;

            // 5. Calculate the rotation: Look along the flattened direction, using WORLD UP (0, 1, 0)
            // This explicitly prevents any roll (Z) component, locking the canvas upright.
            m_captureRotation = Quaternion.LookRotation(forwardDirection, Vector3.up);

            // If forwardDirection is zero (shouldn't happen), default to identity
            if (forwardDirection == Vector3.zero)
            {
                m_captureRotation = Quaternion.identity;
            }
        }

        public Vector3 GetCapturedCameraPosition()
        {
            return m_captureCameraPose.position;
        }

    }
}
