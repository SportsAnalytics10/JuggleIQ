// Kaggle/ngrok API base — no trailing slash
export const API_BASE =
  import.meta.env.VITE_API_BASE || "https://lakisha-deltaic-conception.ngrok-free.dev";

// Build full URL for annotated video (API returns path like /download/{job_id})
export function getAnnotatedVideoUrl(annotatedPath) {
  if (!annotatedPath) return null;
  if (annotatedPath.startsWith("http")) return annotatedPath;
  const base = API_BASE.replace(/\/$/, "");
  const path = annotatedPath.startsWith("/") ? annotatedPath : `/${annotatedPath}`;
  return `${base}${path}`;
}
