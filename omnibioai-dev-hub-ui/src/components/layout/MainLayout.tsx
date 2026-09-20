import { useState } from "react";
import { useCurrentUser } from "../../hooks/useCurrentUser";

const NAV = [
  {
    section: "Overview",
    items: [
      { id: "dashboard", label: "Overview", icon: "⬡" },
    ],
  },
  {
    section: "Knowledge Engine",
    items: [
      { id: "ingest",  label: "Ingestion",    icon: "⇧" },
      { id: "vectors", label: "Vector Store",  icon: "◈" },
      { id: "rag",     label: "RAG Pipeline",  icon: "⟳" },
      { id: "models",  label: "Models",        icon: "◉" },
    ],
  },
  {
    section: "Playground",
    items: [
      { id: "chat",   label: "Ask OmniBioAI", icon: "✦" },
      { id: "search", label: "Vector Search",   icon: "⌕" },
      { id: "graph",  label: "Knowledge Graph", icon: "◎" },
    ],
  },
  {
    section: "Observability",
    items: [
      { id: "docs",    label: "System Status",  icon: "◫" },
    ],
  },
];

interface Props {
  page: string;
  setPage: (p: string) => void;
  children: React.ReactNode;
  breadcrumb?: string;
}

export default function MainLayout({ page, setPage, children, breadcrumb }: Props) {
  const user = useCurrentUser();
  return (
    <div className="app-shell">
      {/* TOPBAR */}
      <header className="topbar">
        <div className="topbar-logo">
          <div className="topbar-logo-icon">⬡</div>
          OmniBioAI
        </div>
        <div className="topbar-sep" />
        <span className="topbar-breadcrumb">Dev Hub</span>
        {breadcrumb && (
          <>
            <span style={{ color: "var(--text-muted)", fontSize: 13 }}>/</span>
            <span className="topbar-breadcrumb" style={{ color: "var(--text-primary)" }}>
              {breadcrumb}
            </span>
          </>
        )}
        <div className="topbar-right">
          <div className="status-pill">
            <span className="status-dot" />
            System Healthy
          </div>
          <span className="version-tag">v6-faiss</span>
        </div>
      </header>

      <div className="app-body">
        {/* SIDEBAR */}
        <nav className="sidebar">
          {NAV.map((group) => (
            <div key={group.section}>
              <div className="nav-section-label">{group.section}</div>
              {group.items.map((item) => (
                <div
                  key={item.id}
                  className={`nav-item${page === item.id ? " active" : ""}`}
                  onClick={() => setPage(item.id)}
                >
                  <span style={{ fontSize: 14, width: 16, textAlign: "center" }}>
                    {item.icon}
                  </span>
                  {item.label}
                </div>
              ))}
            </div>
          ))}

          <div className="sidebar-footer">
            <div className="sidebar-footer-user">
              <div className="avatar">{user.initials}</div>
              <div>
                <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 500 }}>{user.name}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{user.role}</div>
              </div>
            </div>
          </div>
        </nav>

        {/* PAGE AREA */}
        <main className="page-content">
          {children}
        </main>
      </div>
    </div>
  );
}