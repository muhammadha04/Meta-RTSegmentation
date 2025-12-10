// Copyright (c) Meta Platforms, Inc. and affiliates.

using Meta.XR.Samples;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SegmentationSpatialMarker : MonoBehaviour
    {
        private Vector3 m_fixedPosition;
        private Quaternion m_fixedRotation;

        public void SetFixedPositionAndRotation(Vector3 pos, Quaternion rot)
        {
            m_fixedPosition = pos;
            m_fixedRotation = rot;
            transform.position = pos;
            transform.rotation = rot;
        }

        private void Update()
        {
            transform.position = m_fixedPosition;
            transform.rotation = m_fixedRotation;
        }

        private void LateUpdate()
        {
            transform.position = m_fixedPosition;
            transform.rotation = m_fixedRotation;
        }
    }
}