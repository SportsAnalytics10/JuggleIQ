import { useState, useRef, useEffect } from "react";
import { animate, stagger } from "animejs";
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
  const headerRef = useRef(null);

  // Anime.js: subtle header entrance on load (https://animejs.com/)
  useEffect(() => {
    if (!headerRef.current) return;
    const h1 = headerRef.current.querySelector("h1");
    const tag = headerRef.current.querySelector(".tagline");
    if (h1) animate(h1, { opacity: 1, translateY: 0, duration: 600, ease: "outExpo" });
    if (tag) animate(tag, { opacity: 1, duration: 500, delay: 150, ease: "outExpo" });
  }, []);

  // Anime.js: subtle entrance animations when results load (https://animejs.com/)
  useEffect(() => {
    if (!result) return;
    const el = resultsRef.current;
    if (!el) return;
    const run = () => {
      const actions = el.querySelector(".results-actions");
      const cards = el.querySelectorAll(".metric-card");
      const coaching = el.querySelectorAll(".coaching-block, .knee-tip");
      const charts = el.querySelectorAll(".chart-card");
      const table = el.querySelector(".touches-table-wrap");
      const figma = el.querySelector(".figma-make-block");
      if (actions) animate(actions, { opacity: 1, translateY: 0, duration: 400, ease: "outExpo" });
      if (cards.length) animate(cards, { opacity: 1, translateY: 0, duration: 450, delay: stagger(70, { start: 80 }), ease: "outExpo" });
      if (coaching.length) animate(coaching, { opacity: 1, translateY: 0, duration: 450, delay: 280, ease: "outExpo" });
      if (charts.length) animate(charts, { opacity: 1, translateY: 0, duration: 500, delay: stagger(120, { start: 350 }), ease: "outExpo" });
      if (table) animate(table, { opacity: 1, translateY: 0, duration: 450, delay: 500, ease: "outExpo" });
      if (figma) animate(figma, { opacity: 1, translateY: 0, duration: 500, delay: 550, ease: "outExpo" });
    };
    const t = requestAnimationFrame(() => requestAnimationFrame(run));
    return () => cancelAnimationFrame(t);
  }, [result]);

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
    const payload = {
      ...result,
      annotated_video_full_url: getAnnotatedVideoUrl(result.annotated_video_url) || null,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `juggleiq-session-${result.job_id || "session"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

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
      <div className="app-bg" aria-hidden="true">
        <div className="app-bg-gradient" />
        <div className="app-bg-grid" />
        <div className="app-bg-blob app-bg-blob-1" />
        <div className="app-bg-blob app-bg-blob-2" />
      </div>

      <header className="header" ref={headerRef}>
        <div className="header-graphic">
          <svg viewBox="0 0 120 48" fill="none" xmlns="http://www.w3.org/2000/svg" className="header-svg">
            <circle cx="24" cy="24" r="10" stroke="url(#headerGrad)" strokeWidth="2" fill="none" opacity="0.8" />
            <circle cx="60" cy="24" r="10" stroke="url(#headerGrad)" strokeWidth="2" fill="none" opacity="0.5" />
            <circle cx="96" cy="24" r="10" stroke="url(#headerGrad)" strokeWidth="2" fill="none" opacity="0.8" />
            <path d="M34 24h22M66 24h22" stroke="url(#headerGrad)" strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
            <defs>
              <linearGradient id="headerGrad" x1="0" y1="0" x2="120" y2="48" gradientUnits="userSpaceOnUse">
                <stop stopColor="var(--accent)" />
                <stop offset="1" stopColor="var(--purple)" />
              </linearGradient>
            </defs>
          </svg>
        </div>
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
        <h2 className="section-title"><span className="section-title-bar" />Sample videos</h2>
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
        <h2 className="section-title"><span className="section-title-bar" />Upload & analyze</h2>
        <div className="upload-box">
          <div className="upload-icon">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M24 8v32M8 24h32" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
              <path d="M24 16l-8 8 8 8 8-8-8-8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />
              <circle cx="24" cy="24" r="18" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.4" />
            </svg>
          </div>
          <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm" onChange={handleFileChange} className="file-input" id="file-upload" />
          <label htmlFor="file-upload" className="file-label">Choose video file</label>
          <p className="file-name">{file ? file.name : "No file chosen"}</p>
          <button onClick={handleAnalyze} disabled={loading} className="btn btn-primary">
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
        {loading && <p className="loading-msg">Processing video… this may take a minute.</p>}
      </section>

      <section id="results" className="section results-section" ref={resultsRef}>
        <h2 className="section-title"><span className="section-title-bar" />Results</h2>
        {!result && !loading && <p className="muted">Upload a video and click Analyze to see results here.</p>}

        {result && (
          <>
            <div className="results-actions">
              <button onClick={handleDownloadJson} className="btn btn-secondary">
                Download session JSON
              </button>
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

            <div className="figma-make-block">
              <h3>Figma Make — Animate your analysis</h3>
              <p className="figma-make-desc">
                Skip the in-app video and build a <strong>unique animated dashboard</strong> in Figma Make. Use your session JSON as the dataset and bring the numbers to life.
              </p>
              <a href={FIGMA_LINK} target="_blank" rel="noreferrer" className="btn btn-primary figma-cta">
                Open Figma Make dashboard →
              </a>
              <div className="figma-ideas">
                <h4>Animation ideas from your API response</h4>
                <ul>
                  <li><strong>Tap a touch</strong> → Reveal foot (L/R), time, and coordinates with a quick highlight or pulse.</li>
                  <li><strong>Touch timeline</strong> → Staggered reveal of <code>touches[]</code> one by one (e.g. dots or steps) with labels.</li>
                  <li><strong>Skill score gauge</strong> → Animate a needle or bar from 0 to <code>skill_score</code> when the frame loads.</li>
                  <li><strong>Foot usage donut</strong> → Build the donut from <code>left_foot</code> / <code>right_foot</code> with a short draw animation.</li>
                  <li><strong>Rhythm chart</strong> → Animate <code>intervals[]</code> as a line or bars that draw in sequence.</li>
                  <li><strong>Knee stiffness badge</strong> → Show <code>stiffness_label</code> + <code>stiffness_tip</code> with a subtle entrance animation.</li>
                  <li><strong>Coaching tips</strong> → Cycle or reveal <code>coaching_tips[]</code> with type-on or fade-in.</li>
                </ul>
                <p className="figma-hint">Download the session JSON above and import it as your Figma Make dataset (&lt;5MB).</p>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="section links-section">
        <h2 className="section-title"><span className="section-title-bar" />Resources</h2>
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
              Figma Make dashboard
            </a>
          </li>
        </ul>
      </section>

      <footer className="footer">
        <div className="footer-line" />
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
