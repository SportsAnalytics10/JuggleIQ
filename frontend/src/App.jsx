import { useState, useRef } from "react";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
  CartesianGrid,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { API_BASE, getAnnotatedVideoUrl } from "./config";
import "./App.css";

const SAMPLE_VIDEOS = [
  { name: "Drill 1", src: "/samples/sample1.mov" },
  { name: "Drill 2", src: "/samples/sample2.mov" },
  { name: "Drill 3", src: "/samples/sample3.mov" },
  { name: "Drill 4", src: "/samples/sample4.mov" },
  { name: "Drill 5", src: "/samples/sample5.mov" },
];

const YOUTUBE_LINKS = [
  { label: "Juggling basics", url: "https://www.youtube.com/results?search_query=soccer+juggling+tutorial" },
  { label: "Footwork drills", url: "https://www.youtube.com/results?search_query=footwork+juggling+drills" },
];

const FIGMA_LINK = "https://www.figma.com";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const resultsRef = useRef(null);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select a video file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || res.statusText || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `juggleiq-session-${result.job_id || "session"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const videoUrl = result ? getAnnotatedVideoUrl(result.annotated_video_url) : null;

  // Chart data from result
  const footData = result
    ? [
        { name: "Left", value: result.left_foot ?? 0, color: "#22c55e" },
        { name: "Right", value: result.right_foot ?? 0, color: "#3b82f6" },
        { name: "Unknown", value: result.unknown ?? 0, color: "#94a3b8" },
      ].filter((d) => d.value > 0)
    : [];

  const rhythmData =
    result?.intervals?.map((val, i) => ({ index: i + 1, interval: val, label: `Touch ${i + 1}→${i + 2}` })) ?? [];

  const scoreBreakdown = result?.score_breakdown
    ? [
        { metric: "Touch", score: result.score_breakdown.touch_score, fullMark: 100 },
        { metric: "Rhythm", score: result.score_breakdown.rhythm_score ?? result.rhythm_score, fullMark: 100 },
        { metric: "Peak", score: result.score_breakdown.peak_score, fullMark: 100 },
        { metric: "Drift", score: result.score_breakdown.drift_score, fullMark: 100 },
      ]
    : [];

  return (
    <div className="app">
      <header className="header">
        <h1>JuggleIQ</h1>
        <p className="tagline">CV-powered juggling coach — upload, analyze, improve.</p>
      </header>

      <nav className="nav">
        <a href="#samples">Samples</a>
        <a href="#upload">Upload</a>
        <a href="#results">Results</a>
        <a href={FIGMA_LINK} target="_blank" rel="noreferrer">
          Figma Dashboard
        </a>
      </nav>

      <section id="samples" className="section">
        <h2>Sample videos</h2>
        <p className="section-desc">Videos from the drills folder. Upload your own in the section below to analyze.</p>
        <div className="sample-grid">
          {SAMPLE_VIDEOS.map((s) => (
            <div key={s.name} className="sample-card">
              <video controls src={s.src} className="sample-video" />
              <span>{s.name}</span>
            </div>
          ))}
        </div>
      </section>

      <section id="upload" className="section upload-section">
        <h2>Upload & analyze</h2>
        <div className="upload-box">
          <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm" onChange={handleFileChange} className="file-input" />
          <p className="file-name">{file ? file.name : "No file chosen"}</p>
          <button onClick={handleAnalyze} disabled={loading} className="btn btn-primary">
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
        {loading && <p className="loading-msg">Processing video… this may take a minute.</p>}
      </section>

      <section id="results" className="section results-section" ref={resultsRef}>
        <h2>Results</h2>
        {!result && !loading && <p className="muted">Upload a video and click Analyze to see results here.</p>}

        {result && (
          <>
            <div className="results-actions">
              <button onClick={handleDownloadJson} className="btn btn-secondary">
                Download session JSON
              </button>
            </div>

            <div className="result-video-block">
              <h3>Annotated video</h3>
              {videoUrl ? (
                <video controls src={videoUrl} className="result-video" />
              ) : (
                <p className="muted">Video URL not available.</p>
              )}
            </div>

            <div className="metrics-grid">
              <MetricCard label="Touch count" value={result.touch_count} />
              <MetricCard label="Best streak" value={result.best_streak} />
              <MetricCard label="Skill score" value={result.skill_score != null ? result.skill_score.toFixed(1) : "—"} />
              <MetricCard label="Rhythm score" value={result.rhythm_score != null ? result.rhythm_score.toFixed(1) : "—"} />
              <MetricCard label="Avg drift (px)" value={result.avg_drift_px != null ? result.avg_drift_px.toFixed(1) : "—"} />
              <MetricCard
                label="Knee stiffness"
                value={result.knee_feedback?.stiffness_label?.replace(/_/g, " ") ?? "—"}
              />
            </div>

            {result.coaching_tips?.length > 0 && (
              <div className="coaching-block">
                <h3>Coaching tips</h3>
                <ul>
                  {result.coaching_tips.map((tip, i) => (
                    <li key={i}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.knee_feedback?.stiffness_tip && (
              <div className="knee-tip">
                <strong>Knee tip:</strong> {result.knee_feedback.stiffness_tip}
              </div>
            )}

            <div className="charts-row">
              {footData.length > 0 && (
                <div className="chart-card">
                  <h4>Foot usage</h4>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={footData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                        {footData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {rhythmData.length > 0 && (
                <div className="chart-card">
                  <h4>Rhythm intervals (s)</h4>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={rhythmData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="interval" stroke="#8b5cf6" strokeWidth={2} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {scoreBreakdown.length > 0 && (
                <div className="chart-card">
                  <h4>Score breakdown</h4>
                  <ResponsiveContainer width="100%" height={220}>
                    <RadarChart data={scoreBreakdown}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar name="Score" dataKey="score" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.5} />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {result.touches?.length > 0 && (
              <div className="touches-table-wrap">
                <h3>Touch timeline</h3>
                <table className="touches-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Foot</th>
                      <th>Time (s)</th>
                      <th>Frame</th>
                      <th>X, Y</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.touches.map((t) => (
                      <tr key={t.touch_num}>
                        <td>{t.touch_num}</td>
                        <td>{t.foot}</td>
                        <td>{typeof t.t === "number" ? t.t.toFixed(2) : t.t}</td>
                        <td>{t.frame_idx}</td>
                        <td>{t.x != null && t.y != null ? `${t.x}, ${t.y}` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>

      <section className="section links-section">
        <h2>Resources</h2>
        <ul className="resource-list">
          {YOUTUBE_LINKS.map((l) => (
            <li key={l.url}>
              <a href={l.url} target="_blank" rel="noreferrer">
                {l.label}
              </a>
            </li>
          ))}
          <li>
            <a href={FIGMA_LINK} target="_blank" rel="noreferrer">
              Figma dashboard
            </a>
          </li>
        </ul>
      </section>

      <footer className="footer">
        <p>JuggleIQ — Hacklytics 2026</p>
      </footer>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

export default App;
