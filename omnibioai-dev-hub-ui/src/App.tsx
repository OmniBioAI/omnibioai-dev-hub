import { useState } from "react";
import MainLayout from "./components/layout/MainLayout";
import DashboardPage from "./pages/Dashboard";
import ChatPage from "./pages/ChatPage";
import SearchPage from "./pages/SearchPage";
import DocsExplorer from "./pages/DocsExplorer";
import GraphView from "./pages/GraphView";

const PAGE_LABELS: Record<string, string> = {
  dashboard: "Overview",
  chat:      "Ask OmniBioAI",
  search:    "Vector Search",
  graph:     "Knowledge Graph",
  docs:      "System Status",
};

// Deep-link contract for embedders (Studio's ServiceViewer opens
// /_svc/devhub?view=chat): ?view=<page id> selects that page on initial
// load, same ids as PAGE_LABELS/renderPage below. Only ever read once, at
// mount -- Dev Hub has no router and this isn't meant to track the URL
// afterward, just to seed where the existing internal navigation starts.
// Absent or unrecognized values fall back to the pre-existing default
// ("dashboard") rather than ever passing an unvalidated value through.
function initialPageFromUrl(): string {
  if (typeof window === "undefined") return "dashboard";
  const requested = new URLSearchParams(window.location.search).get("view");
  return requested && requested in PAGE_LABELS ? requested : "dashboard";
}

export default function App() {
  const [page, setPage] = useState(initialPageFromUrl);

  const renderPage = () => {
    switch (page) {
      case "dashboard": return <DashboardPage onNavigate={setPage} />;
      case "chat":      return <ChatPage />;
      case "search":    return <SearchPage />;
      case "graph":     return <GraphView />;
      case "docs":      return <DocsExplorer />;
      default:          return <DashboardPage onNavigate={setPage} />;
    }
  };

  return (
    <MainLayout page={page} setPage={setPage} breadcrumb={PAGE_LABELS[page]}>
      {renderPage()}
    </MainLayout>
  );
}