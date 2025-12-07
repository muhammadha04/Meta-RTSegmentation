// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Collections;
using Meta.XR;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceRunManager : MonoBehaviour
    {
        [Header("Sentis Model config")]
        [SerializeField] private Vector2Int m_inputSize = new(640, 640);
        [SerializeField] private BackendType m_backend = BackendType.CPU;
        [SerializeField] private ModelAsset m_sentisModel;
        [SerializeField] private int m_layersPerFrame = 25;
        [SerializeField] private TextAsset m_labelsAsset;
        public bool IsModelLoaded { get; private set; } = false;

        [Header("UI display references")]
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel;
        [SerializeField, Range(0, 1)] private float m_iouThreshold = 0.6f;
        [SerializeField, Range(0, 1)] private float m_scoreThreshold = 0.23f;
        [Space(40)]

        private Worker m_engine;
        private IEnumerator m_schedule;
        private bool m_started = false;
        private Tensor<float> m_input;
        private Model m_model;
        private int m_download_state = 0;

        // CHANGED: Output0 contains boxes + class scores + mask coefficients (1, 116, 8400)
        private Tensor<float> m_output;

        // CHANGED: Output1 is now prototype masks (1, 32, 160, 160) instead of label IDs
        private Tensor<float> m_prototypeMasks;

        private Tensor<float> m_pullOutput;
        private Tensor<float> m_pullPrototypeMasks;  // CHANGED: Was Tensor<int> m_pullLabelIDs

        private bool m_isWaiting = false;
        private Pose m_imageCameraPose;

        #region Unity Functions
        private IEnumerator Start()
        {
            yield return new WaitForSeconds(0.05f);
            m_uiInference.SetLabels(m_labelsAsset);
            LoadModel();
        }

        private void Update()
        {
            InferenceUpdate();
        }

        private void OnDestroy()
        {
            if (m_schedule != null)
            {
                StopCoroutine(m_schedule);
            }
            m_input?.Dispose();
            m_prototypeMasks?.Dispose();  // CHANGED: Was m_labelIDs
            m_engine?.Dispose();
        }
        #endregion

        #region Public Functions
        public void RunInference(PassthroughCameraAccess cameraAccess)
        {
            if (!m_started)
            {
                m_imageCameraPose = cameraAccess.GetCameraPose();
                m_input?.Dispose();

                Texture targetTexture = cameraAccess.GetTexture();
                m_uiInference.SetDetectionCapture(targetTexture);

                // FIXED: Removed deprecated SetDimensions() - dimensions auto-detected from tensor shape
                m_input = new Tensor<float>(new TensorShape(1, 3, m_inputSize.x, m_inputSize.y));
                TextureConverter.ToTensor(targetTexture, m_input);  // FIXED: No SetDimensions needed

                m_schedule = m_engine.ScheduleIterable(m_input);
                m_download_state = 0;
                m_started = true;
            }
        }

        public bool IsRunning()
        {
            return m_started;
        }
        #endregion

        #region Inference Functions
        private void LoadModel()
        {
            var model = ModelLoader.Load(m_sentisModel);
            Debug.Log($"Sentis segmentation model loaded - Outputs: {model.outputs.Count}");

            // DEBUG: Print detailed model information
            Debug.Log($"DEBUG MODEL: Input count = {model.inputs.Count}");
            for (int i = 0; i < model.inputs.Count; i++)
            {
                Debug.Log($"DEBUG MODEL: Input[{i}] name='{model.inputs[i].name}'");
            }

            for (int i = 0; i < model.outputs.Count; i++)
            {
                Debug.Log($"DEBUG MODEL: Output[{i}] name='{model.outputs[i].name}'");
            }

            m_engine = new Worker(model, m_backend);

            // Warmup inference - FIXED: Removed deprecated SetDimensions()
            Texture m_loadingTexture = new Texture2D(m_inputSize.x, m_inputSize.y, TextureFormat.RGBA32, false);
            m_input = new Tensor<float>(new TensorShape(1, 3, m_inputSize.x, m_inputSize.y));
            TextureConverter.ToTensor(m_loadingTexture, m_input);

            m_engine.Schedule(m_input);

            Debug.Log("DEBUG MODEL: Warmup inference completed - checking actual output shapes...");

            // DEBUG: After warmup, peek at actual output shapes
            var testOutput0 = m_engine.PeekOutput(0) as Tensor<float>;
            var testOutput1 = m_engine.PeekOutput(1) as Tensor<float>;
            if (testOutput0 != null)
            {
                Debug.Log($"DEBUG MODEL: Actual Output[0] shape = {testOutput0.shape}");
            }
            if (testOutput1 != null)
            {
                Debug.Log($"DEBUG MODEL: Actual Output[1] shape = {testOutput1.shape}");
            }

            IsModelLoaded = true;
        }

        private void InferenceUpdate()
        {
            if (m_started)
            {
                try
                {
                    if (m_download_state == 0)
                    {
                        var it = 0;
                        while (m_schedule.MoveNext())
                        {
                            if (++it % m_layersPerFrame == 0)
                                return;
                        }
                        m_download_state = 1;
                    }
                    else
                    {
                        GetInferencesResults();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Sentis error: {e.Message}");
                }
            }
        }

        private void PollRequestOuput()
        {
            // CHANGED: Get output 0 (boxes + class scores + mask coefficients)
            m_pullOutput = m_engine.PeekOutput(0) as Tensor<float>;
            if (m_pullOutput.dataOnBackend != null)
            {
                m_pullOutput.ReadbackRequest();
                m_isWaiting = true;
            }
            else
            {
                Debug.LogError("Sentis: No data output m_output");
                m_download_state = 4;
            }
        }

        // CHANGED: Renamed from PollRequestLabelIDs to PollRequestPrototypeMasks
        private void PollRequestPrototypeMasks()
        {
            // CHANGED: Get output 1 (prototype masks) as Tensor<float> not Tensor<int>
            m_pullPrototypeMasks = m_engine.PeekOutput(1) as Tensor<float>;
            if (m_pullPrototypeMasks.dataOnBackend != null)
            {
                m_pullPrototypeMasks.ReadbackRequest();
                m_isWaiting = true;
            }
            else
            {
                Debug.LogError("Sentis: No data output m_prototypeMasks");
                m_download_state = 4;
            }
        }

        private void GetInferencesResults()
        {
            switch (m_download_state)
            {
                case 1:
                    if (!m_isWaiting)
                    {
                        PollRequestOuput();
                    }
                    else
                    {
                        if (m_pullOutput.IsReadbackRequestDone())
                        {
                            m_output = m_pullOutput.ReadbackAndClone();
                            m_isWaiting = false;

                            if (m_output.shape[0] > 0)
                            {
                                Debug.Log($"DEBUG OUTPUT: Shape = {m_output.shape}");
                                Debug.Log($"DEBUG OUTPUT: Dimensions = [{m_output.shape[0]}, {m_output.shape[1]}, {m_output.shape[2]}]");
                                // Log first few values to understand the data
                                Debug.Log($"DEBUG OUTPUT: [0,0,0]={m_output[0, 0, 0]} [0,1,0]={m_output[0, 1, 0]} [0,0,1]={m_output[0, 0, 1]}");
                                m_download_state = 2;
                            }
                            else
                            {
                                Debug.Log("Sentis: m_output empty");
                                m_download_state = 4;
                            }
                        }
                    }
                    break;
                case 2:
                    if (!m_isWaiting)
                    {
                        PollRequestPrototypeMasks();  // CHANGED: Was PollRequestLabelIDs()
                    }
                    else
                    {
                        // CHANGED: Check prototype masks instead of label IDs
                        if (m_pullPrototypeMasks.IsReadbackRequestDone())
                        {
                            m_prototypeMasks = m_pullPrototypeMasks.ReadbackAndClone();
                            m_isWaiting = false;

                            if (m_prototypeMasks.shape[0] > 0)
                            {
                                Debug.Log($"DEBUG PROTOTYPES: Shape = {m_prototypeMasks.shape}");
                                Debug.Log($"DEBUG PROTOTYPES: Dimensions = [{m_prototypeMasks.shape[0]}, {m_prototypeMasks.shape[1]}, {m_prototypeMasks.shape[2]}, {m_prototypeMasks.shape[3]}]");
                                m_download_state = 3;
                            }
                            else
                            {
                                Debug.LogError("Sentis: m_prototypeMasks empty");
                                m_download_state = 4;
                            }
                        }
                    }
                    break;
                case 3:
                    // CHANGED: Pass prototype masks instead of label IDs
                    m_uiInference.DrawUIBoxes(m_output, m_prototypeMasks, m_inputSize.x, m_inputSize.y, m_imageCameraPose);
                    m_download_state = 5;
                    break;
                case 4:
                    m_uiInference.OnObjectDetectionError();
                    m_download_state = 5;
                    break;
                case 5:
                    m_download_state++;
                    m_started = false;
                    m_output?.Dispose();
                    m_prototypeMasks?.Dispose();  // CHANGED: Was m_labelIDs
                    break;
            }
        }
        #endregion
    }
}