import axios from "axios";

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
