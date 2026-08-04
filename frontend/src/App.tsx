import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import Dashboard from "./pages/Index";
import RunManager from "./pages/RunManager";
import RunDetails from "./pages/RunDetails";
import Results from "./pages/Results";
import Explorer from "./pages/Explorer";
import SettingsPage from "./pages/Settings";
import DocsPage from "./pages/Docs";
import Analysis from "./pages/Analysis";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Migrate localhost to 127.0.0.1 in settings to avoid IPv6 loopback issues on Windows
try {
  const saved = localStorage.getItem('knf-settings');
  if (saved) {
    const settings = JSON.parse(saved);
    if (settings && settings.apiBaseUrl && settings.apiBaseUrl.includes('localhost')) {
      settings.apiBaseUrl = settings.apiBaseUrl.replace('localhost', '127.0.0.1');
      localStorage.setItem('knf-settings', JSON.stringify(settings));
      console.log('Migrated knf-settings apiBaseUrl from localhost to 127.0.0.1');
    }
  }
} catch (e) {
  console.error('Failed to migrate settings:', e);
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <HashRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/runs" element={<RunManager />} />
            <Route path="/runs/:runId" element={<RunDetails />} />
            <Route path="/results" element={<Results />} />
            <Route path="/explorer" element={<Explorer />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/docs" element={<DocsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AppShell>
      </HashRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
