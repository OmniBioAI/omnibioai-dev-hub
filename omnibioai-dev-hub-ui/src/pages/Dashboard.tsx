import { useEffect, useState } from "react";
import { getStatus } from "../api/client";

interface Props {
  onNavigate: (page: string) => void;
}

interface StatusData {
  control_plane: {
    status: string;
    uptime_sec: number;
    engine_ready: boolean;
    error: string | null;
    metrics: { init_time: number; build_time_ms: number };
  };
  index_vectors: number;
  graph_edges: number;
  plugins_loaded: number;
}

function Sparkline({ data, color }: { data: number[]; color?: string }) {
  const max = Math.max(...data, 1);
  return (
    <div className="sparkline">
      {data.map((v, i) => (
        <div
          key={i}
          className={`spark-bar${color ? " " + color : ""}`}
          style={{ height: `${Math.round((v / max) * 100)}%` }}
        />
      ))}
    </div>
  );
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

export default function DashboardPage({ onNavigate }: Props) {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [uptimeHistory, setUptimeHistory] = useState<number[]>([0]);
  const [docHistory, setDocHistory] = useState<number[]>([0]);

  const fetchStatus = async () => {
    try {
      const data: StatusData = await getStatus();
      setStatus(data);
      setError(null);
      setLastRefresh(new Date());
      setUptimeHistory(h => [...h.slice(-14), data.control_plane.uptime_sec]);
      setDocHistory(h => [...h.slice(-14), data.index_vectors]);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 10000);
    return () => clearInterval(id);
  }, []);

  if (loading) {
    return (
      <div className="surface" style={{ textAlign: "center", padding: 40 }}>
        <span className="spinner" />
        <span style={{ color: "var(--text-muted)" }}>Connecting to backend on :8082…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="surface" style={{ borderColor: "rgba(225,29,72,.3)" }}>
        <div style={{ color: "var(--danger)", fontWeight: 600, marginBottom: 8 }}>Backend unreachable</div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>{error}</div>
        <code style={{ fontSize: 12, color: "var(--accent)" }}>
          uvicorn api.main:app --host 0.0.0.0 --port 8082 --reload
        </code>
        <br />
        <button className="btn-primary" onClick={fetchStatus} style={{ marginTop: 12 }}>↻ Retry</button>
      </div>
    );
  }

  const cp = status!.control_plane;
  const indexVectors = status!.index_vectors;

  const pipelineSteps = [
    { label: "Indexing",   sub: `${indexVectors} vectors`,                                 icon: "◈", state: indexVectors > 0 ? "done" : "active" },
    { label: "Plugins",    sub: `${status!.plugins_loaded} loaded`,                        icon: "◫", state: status!.plugins_loaded > 0 ? "done" : "active" },
    { label: "Graph",      sub: `${status!.graph_edges} edges`,                            icon: "⧉", state: status!.graph_edges > 0 ? "done" : "active" },
    { label: "RAG Engine", sub: cp.engine_ready ? "Ready" : "Initializing",               icon: "✦", state: cp.engine_ready ? "active" : "done" },
  ];

  return (
    <>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div className="page-header">
          <div className="page-title">Overview</div>
          <div className="page-sub">Last updated {lastRefresh.toLocaleTimeString()}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`badge ${cp.status === "READY" ? "success" : "warn"}`}>{cp.status}</span>
          <button className="section-action" onClick={fetchStatus}>↻ Refresh</button>
        </div>
      </div>

      {/* STAT CARDS */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Index Vectors</div>
          <div className="stat-value">{indexVectors}</div>
          <div className="stat-delta">● Indexed</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Graph Edges</div>
          <div className="stat-value">{status!.graph_edges}</div>
          <div className="stat-delta muted">total</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Plugins</div>
          <div className="stat-value">{status!.plugins_loaded}</div>
          <div className="stat-delta muted">loaded</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Uptime</div>
          <div className="stat-value">{formatUptime(cp.uptime_sec)}</div>
          <div className="stat-delta">● {cp.engine_ready ? "Engine ready" : "Starting"}</div>
        </div>
      </div>

      {/* PIPELINE */}
      <div className="surface">
        <div className="section-header">
          <div className="section-title">RAG Pipeline</div>
          <button className="section-action" onClick={() => onNavigate("chat")}>Ask OmniBioAI →</button>
        </div>
        <div className="pipeline-row">
          {pipelineSteps.map((step, i) => (
            <div key={i} className={`pipeline-step ${step.state}`}>
              <div className="pipeline-node">{step.icon}</div>
              <div className="pipeline-step-label">{step.label}</div>
              <div className="pipeline-step-sub">{step.sub}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="two-col">
        {/* LIVE METRICS */}
        <div className="surface">
          <div className="section-header">
            <div className="section-title">Live Metrics</div>
            <span className="badge success">● polling 10s</span>
          </div>
          <div className="metric-row">
            <div className="metric-item">
              <div className="metric-header">
                <span className="metric-name">Index Vectors</span>
                <span className="metric-val">{indexVectors}</span>
              </div>
              <Sparkline data={docHistory} />
            </div>
            <div className="metric-item">
              <div className="metric-header">
                <span className="metric-name">Uptime</span>
                <span className="metric-val">{formatUptime(cp.uptime_sec)}</span>
              </div>
              <Sparkline data={uptimeHistory} color="blue" />
            </div>
            <div className="metric-item">
              <div className="metric-header">
                <span className="metric-name">Graph Edges</span>
                <span className="metric-val">{status!.graph_edges}</span>
              </div>
              <Sparkline data={[1, status!.graph_edges]} color="amber" />
            </div>
            <div className="metric-item">
              <div className="metric-header">
                <span className="metric-name">Plugins Loaded</span>
                <span className="metric-val">{status!.plugins_loaded}</span>
              </div>
              <Sparkline data={[status!.plugins_loaded]} />
            </div>
          </div>
        </div>

        {/* SYSTEM INFO TABLE */}
        <div className="surface">
          <div className="section-header">
            <div className="section-title">System Info</div>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <tbody>
              {[
                ["Engine status",   cp.status],
                ["Engine ready",    cp.engine_ready ? "Yes" : "No"],
                ["Index vectors",   String(indexVectors)],
                ["Plugins loaded",  String(status!.plugins_loaded)],
                ["Graph edges",     String(status!.graph_edges)],
                ["Build time",      `${cp.metrics.build_time_ms?.toFixed(2) ?? "—"} ms`],
                ["Uptime",          `${cp.uptime_sec.toFixed(1)} s`],
              ].map(([k, v]) => (
                <tr key={k} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "7px 0", fontSize: 11, color: "var(--text-muted)", width: "50%" }}>{k}</td>
                  <td style={{ padding: "7px 0", fontSize: 12, fontFamily: "var(--mono)", color: "var(--text)", textAlign: "right" }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
            <button className="btn-primary" onClick={() => onNavigate("chat")} style={{ flex: 1 }}>✦ Ask OmniBioAI</button>
            <button className="btn-primary" onClick={() => onNavigate("search")} style={{ flex: 1 }}>⌕ Vector Search</button>
          </div>
        </div>
      </div>
    </>
  );
}