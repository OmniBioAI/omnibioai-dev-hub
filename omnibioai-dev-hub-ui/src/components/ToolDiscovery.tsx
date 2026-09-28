import { useEffect, useState } from "react";
import {
  discoveryCategories,
  discoveryCatalogStatus,
  discoveryDataTypes,
  discoverySearch,
} from "../api/client";
import type { DiscoveryFacetItem, DiscoveryTool } from "../api/client";

const displayStatus = (value: unknown) => value == null ? "Unknown" : String(value);
const nameOf = (value: Record<string, unknown>) => String(value.name || value.input_name || value.id || "input");
const typeOf = (value: Record<string, unknown>) => String(value.normalized_type || value.type || value.data_type || "Unknown");

function FacetSelect({ label, value, items, onChange }: { label: string; value: string; items: DiscoveryFacetItem[]; onChange: (v: string) => void }) {
  return <label style={{ fontSize: 12, color: "var(--text-secondary)" }}>{label}{" "}<select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}><option value="">Any</option>{items.map((item) => <option key={item.value} value={item.value}>{item.value} ({item.count})</option>)}</select></label>;
}

function ToolCard({ tool }: { tool: DiscoveryTool }) {
  const values = (items: Array<Record<string, unknown>> | undefined) => items?.length ? items.map((item) => `${nameOf(item)} — ${typeOf(item)}`).join(", ") : "None listed";
  const types = (items: Array<Record<string, unknown>> | undefined) => items?.length ? items.map(typeOf).join(", ") : "Unknown";
  const backends = (tool.backend_capabilities || []).map((item) => String(item.backend || item.type || "Unknown")).join(", ") || "Unknown";
  const servers = (tool.compatible_servers || []).map((item) => String(item.server_id || item.server || "Unknown")).join(", ") || "Unknown";
  return <div className="result-card" data-testid={`tool-result-${tool.tool_id}`}>
    <div className="result-source">{tool.display_name || tool.tool_id}</div>
    <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--teal)", marginBottom: 8 }}>{tool.tool_id}</div>
    {tool.description && <div className="result-text">{tool.description}</div>}
    {tool.classification && <div className="result-text"><strong>Category:</strong> {tool.classification}</div>}
    <div className="result-text"><strong>Required inputs:</strong> {values(tool.required_inputs)}</div>
    <div className="result-text"><strong>Optional inputs:</strong> {values(tool.optional_inputs)}</div>
    <div className="result-text"><strong>Accepted types:</strong> {types(tool.normalized_inputs)}</div>
    <div className="result-text"><strong>Outputs:</strong> {types(tool.normalized_outputs)}</div>
    <div className="result-text"><strong>Execution:</strong> {backends} · {tool.architecture || "Unknown architecture"}</div>
    <div className="result-text"><strong>Servers:</strong> {servers}</div>
    <div className="result-text"><strong>Resources:</strong> {tool.resource_requirements ? JSON.stringify(tool.resource_requirements) : "Unknown"}</div>
    <div className="result-text"><strong>Status:</strong> Configured: {displayStatus(tool.configuration_status)} · Serving: {displayStatus(tool.serving_status)} · Registered: {displayStatus(tool.registration_status)} · Tested: {displayStatus(tool.tested_status)} · Operationally verified: {displayStatus(tool.operational_verification_status)}</div>
  </div>;
}

export default function ToolDiscovery() {
  const [q, setQ] = useState("");
  const [inputType, setInputType] = useState("");
  const [outputType, setOutputType] = useState("");
  const [category, setCategory] = useState("");
  const [backend, setBackend] = useState("");
  const [architecture, setArchitecture] = useState("");
  const [types, setTypes] = useState<DiscoveryFacetItem[]>([]);
  const [categories, setCategories] = useState<DiscoveryFacetItem[]>([]);
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [result, setResult] = useState<{ items: DiscoveryTool[]; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    Promise.all([discoveryDataTypes(), discoveryCategories(), discoveryCatalogStatus()]).then(([t, c, s]) => {
      setTypes(t.items); setCategories(c.items); setStatus(s);
    }).catch(() => setStatus(null));
  }, []);

  const search = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const response = await discoverySearch({ q: q.trim() || undefined, input_type: inputType, output_type: outputType, category, backend, architecture, limit: 20, offset: 0 });
      setResult({ items: response.items, total: response.total });
    } catch {
      setError("TES tool discovery is unavailable. Documentation search remains available in Documentation mode.");
    } finally { setLoading(false); }
  };

  return <>
    <div className="page-header"><div className="page-title">Tool Discovery</div><div className="page-sub">Structured TES discovery · canonical tool metadata</div></div>
    <div className="search-bar-row"><input className="search-input" placeholder="Search TES tools... e.g. paired-end FASTQ alignment" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") search(); }} /><button className="btn-primary" onClick={search} disabled={loading}>{loading ? <><span className="spinner" />Searching</> : "Find tools"}</button></div>
    <div className="surface" style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
      <FacetSelect label="Input type" value={inputType} items={types} onChange={setInputType} /><FacetSelect label="Output type" value={outputType} items={types} onChange={setOutputType} /><FacetSelect label="Category" value={category} items={categories} onChange={setCategory} />
      <label style={{ fontSize: 12, color: "var(--text-secondary)" }}>Backend <input aria-label="Backend" value={backend} onChange={(e) => setBackend(e.target.value)} /></label><label style={{ fontSize: 12, color: "var(--text-secondary)" }}>Architecture <input aria-label="Architecture" value={architecture} onChange={(e) => setArchitecture(e.target.value)} /></label>
    </div>
    {status && <div className="page-sub" data-testid="catalog-status">TES catalog: {String(status.status || status.state || "READY")} · {String(status.tool_count || status.canonical_tool_count || "")} tools</div>}
    {error && <div className="result-card" style={{ borderLeftColor: "var(--red)" }}><div className="result-source" style={{ color: "var(--red)" }} role="alert">Tool discovery unavailable</div><div className="result-text">{error}</div></div>}
    {result && result.items.length === 0 && <div className="surface" role="status">No compatible tools found.</div>}
    {result && result.items.length > 0 && <div><div className="section-header"><div className="section-title">Tools ({result.total})</div></div>{result.items.map((tool) => <ToolCard key={tool.tool_id} tool={tool} />)}</div>}
    {!result && !error && !loading && <div className="surface" style={{ textAlign: "center", padding: "40px 20px" }}><div style={{ color: "var(--text-secondary)", fontSize: 14 }}>Describe your data and analysis goal to find configured TES tools.</div></div>}
  </>;
}
