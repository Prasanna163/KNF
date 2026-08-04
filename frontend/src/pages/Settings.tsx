import { useState } from 'react';
import { Save, Settings, Server, Cpu, BellRing, Check } from 'lucide-react';
import type { AppSettings } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

const defaultSettings: AppSettings = {
  apiBaseUrl: 'http://127.0.0.1:8000',
  defaultProcessingMode: 'auto',
  defaultBackend: 'torch',
  defaultWorkers: 4,
  notificationsEnabled: true,
};

const SettingsPage = () => {
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem('knf-settings');
    const loaded = saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    if (loaded.apiBaseUrl && loaded.apiBaseUrl.includes(':8766')) {
      loaded.apiBaseUrl = loaded.apiBaseUrl.replace(':8766', ':8000');
    }
    return loaded;
  });
  const [saved, setSaved] = useState(false);
  const { toast } = useToast();

  const save = () => {
    localStorage.setItem('knf-settings', JSON.stringify(settings));
    toast({ title: 'Settings Persisted', description: 'Workspace preferences have been saved to local storage.' });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="p-6 lg:p-8 max-w-2xl space-y-8 animate-fade-in-up">
      <header>
        <div className="flex items-center gap-2 text-primary font-mono tracking-widest text-[9px] uppercase mb-2">
          <Settings className="w-3.5 h-3.5" />
          Workspace Preferences
        </div>
        <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">
          System <span className="text-gradient">Settings</span>
        </h1>
        <p className="text-xs text-muted-foreground mt-1">Configure API endpoints, computation defaults, and notification behavior</p>
      </header>

      {/* API Configuration Card */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground uppercase tracking-wider border-b border-glass pb-2">
          <Server className="w-4 h-4 text-primary" />
          Backend Connection
        </div>
        <div className="rounded-lg border border-glass bg-card/45 p-5 space-y-4">
          <Field label="API Base URL" hint="knf_core FastAPI server endpoint (default: http://127.0.0.1:8000)">
            <input
              type="text"
              value={settings.apiBaseUrl}
              onChange={e => setSettings(s => ({ ...s, apiBaseUrl: e.target.value }))}
              className="w-full rounded-lg bg-background/50 border border-glass px-3 py-2 text-xs text-foreground font-mono glass-input outline-none"
              placeholder="http://127.0.0.1:8000"
            />
          </Field>
        </div>
      </section>

      {/* Computation Defaults Card */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground uppercase tracking-wider border-b border-glass pb-2">
          <Cpu className="w-4 h-4 text-primary" />
          Computation Defaults
        </div>
        <div className="rounded-lg border border-glass bg-card/45 p-5 space-y-4">
          <Field label="Default Processing Mode" hint="Strategy used when launching descriptor runs">
            <select
              value={settings.defaultProcessingMode}
              onChange={e => setSettings(s => ({ ...s, defaultProcessingMode: e.target.value as AppSettings['defaultProcessingMode'] }))}
              className="w-full rounded-lg bg-background/50 border border-glass px-3 py-2 text-xs text-foreground glass-input outline-none"
            >
              <option value="auto">Auto (System Decides)</option>
              <option value="single">Single Process</option>
              <option value="multi">Parallel Batch</option>
            </select>
          </Field>
          <Field label="Default NCI Density Backend" hint="Computation engine for NCI analysis">
            <select
              value={settings.defaultBackend}
              onChange={e => setSettings(s => ({ ...s, defaultBackend: e.target.value as 'torch' | 'multiwfn' }))}
              className="w-full rounded-lg bg-background/50 border border-glass px-3 py-2 text-xs text-foreground glass-input outline-none"
            >
              <option value="torch">PyTorch (GPU/CPU accelerated)</option>
              <option value="multiwfn">Multiwfn Integration</option>
            </select>
          </Field>
          <Field label="Default Worker Processes" hint="Parallel processes for batch computation (leave 0 for system default)">
            <input
              type="number"
              min={1}
              max={64}
              value={settings.defaultWorkers}
              onChange={e => setSettings(s => ({ ...s, defaultWorkers: +e.target.value }))}
              className="w-full rounded-lg bg-background/50 border border-glass px-3 py-2 text-xs text-foreground font-mono glass-input outline-none"
            />
          </Field>
        </div>
      </section>

      {/* Notification Settings Card */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground uppercase tracking-wider border-b border-glass pb-2">
          <BellRing className="w-4 h-4 text-primary" />
          Notifications
        </div>
        <div className="rounded-lg border border-glass bg-card/45 p-5">
          <label className="flex items-start justify-between gap-4 cursor-pointer">
            <div>
              <span className="text-sm font-medium text-foreground block">Enable Toast Notifications</span>
              <span className="text-xs text-muted-foreground block mt-0.5 font-mono">
                Show status alerts for run completion, errors, and backend events.
              </span>
            </div>
            <input
              type="checkbox"
              checked={settings.notificationsEnabled}
              onChange={e => setSettings(s => ({ ...s, notificationsEnabled: e.target.checked }))}
              className="mt-0.5 rounded border-glass bg-background/50 text-primary focus:ring-primary"
            />
          </label>
        </div>
      </section>

      {/* Save Button */}
      <button
        onClick={save}
        className={cn(
          'inline-flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold transition-all shadow-sm border border-transparent',
          saved
            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
            : 'bg-primary text-primary-foreground hover:opacity-90'
        )}
      >
        {saved ? <Check className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
        {saved ? 'Saved!' : 'Save Preferences'}
      </button>
    </div>
  );
};

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">{label}</label>
      {children}
      {hint && <p className="text-[10px] text-muted-foreground/60 mt-1 font-mono">{hint}</p>}
    </div>
  );
}

export default SettingsPage;
