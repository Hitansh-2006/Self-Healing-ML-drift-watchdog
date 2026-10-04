// Reads from a .env file (VITE_API_URL=https://your-real-server.com) at
// build time, so the same dashboard code works locally (defaults to
// localhost) and when deployed against a real, remote serving API.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

async function postEmpty(path) {
  const res = await fetch(`${API_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

async function postFile(path, file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // existing endpoints
  getStatus: () => get("/status"),
  getPredictions: (limit = 50) => get(`/admin/predictions?limit=${limit}`),
  getDriftEvents: (limit = 20) => get(`/admin/drift_events?limit=${limit}`),
  getPromotionRequests: () => get("/admin/promotion_requests"),
  decidePromotion: (promotion_request_id, decision) =>
    post("/admin/promote", { promotion_request_id, decision, decided_by: "dashboard" }),

  // new endpoints — KPI stats and action buttons
  getStats: () => get("/admin/stats"),
  uploadStream: (file) => postFile("/admin/upload_stream", file),
  trainBaseline: () => postEmpty("/admin/train_baseline"),
  runDriftCheck: () => postEmpty("/admin/run_drift_check"),
  retrain: () => postEmpty("/admin/retrain"),
};
