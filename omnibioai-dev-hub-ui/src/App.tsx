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

export default function App() {
  const [page, setPage] = useState("dashboard");

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