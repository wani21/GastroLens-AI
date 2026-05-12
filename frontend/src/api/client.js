import axios from "axios";

import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000'
});

export default client;

// Change this if your backend runs on a different host/port
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes — video inference can be slow
});

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiClient.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function predictVideo(file, frameSampleRate = 10) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiClient.post(
    `/predict/video?frame_sample_rate=${frameSampleRate}`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return res.data;
}

export async function checkHealth() {
  const res = await apiClient.get("/health");
  return res.data;
}

export const explainImage = async (file, classIndex = null) => {
  const formData = new FormData();
  formData.append("file", file);
  if (classIndex !== null) {
    formData.append("class_index", classIndex);
  }
  const res = await apiClient.post("/explain", formData, { headers: { "Content-Type": "multipart/form-data" } });
  return res.data;
};

export const predictVideoLSTM = async (file, frameSampleRate = 10) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiClient.post(`/predict/video/lstm?frame_sample_rate=${frameSampleRate}`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 180000,
  });
  return res.data;
};

export const segmentImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiClient.post("/segment", formData, { headers: { "Content-Type": "multipart/form-data" } });
  return res.data;
};

export const shapExplain = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await apiClient.post("/shap", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
  });
  return res.data;
};
