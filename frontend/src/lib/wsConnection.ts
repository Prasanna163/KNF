/**
 * wsConnection.ts — Module-level singleton WebSocket.
 *
 * Unlike the useWebSocket hook (which closes on component unmount), this
 * singleton lives for the entire browser session. RunManager opens it and
 * sends start_run; after navigate('/analysis') the connection stays alive
 * and keeps publishing every message into wsStore — which Analysis reads.
 */
import { publishWsMessage } from './wsStore';

let _ws: WebSocket | null = null;
let _url = 'ws://127.0.0.1:8000/ws/run';

// ── Status listeners ──────────────────────────────────────────────────────
const _statusListeners = new Set<(connected: boolean) => void>();

export function setWsUrl(url: string): void {
  _url = url;
}

export function isWsConnected(): boolean {
  return _ws?.readyState === WebSocket.OPEN;
}

/** Subscribe to connect/disconnect events. Returns an unsubscribe fn. */
export function onWsStatusChange(fn: (connected: boolean) => void): () => void {
  _statusListeners.add(fn);
  return () => _statusListeners.delete(fn);
}

function _notify(connected: boolean) {
  _statusListeners.forEach(fn => fn(connected));
}

// ── Lifecycle ─────────────────────────────────────────────────────────────

export function connectWs(): void {
  if (_ws?.readyState === WebSocket.OPEN || _ws?.readyState === WebSocket.CONNECTING) {
    return;
  }

  try {
    const ws = new WebSocket(_url);

    ws.onopen = () => {
      _ws = ws;
      _notify(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as Record<string, unknown>;
        // Publish every message into the singleton bus — Analysis reads from there
        publishWsMessage(data);
        // Auto-close when the run finishes
        if (
          data.type === 'completed' || data.status === 'completed' ||
          data.type === 'error'     || data.status === 'error'
        ) {
          ws.close();
        }
      } catch {
        publishWsMessage({ type: 'raw', raw: event.data });
      }
    };

    ws.onclose = () => {
      if (_ws === ws) _ws = null;
      _notify(false);
    };

    ws.onerror = () => {
      if (_ws === ws) _ws = null;
      _notify(false);
    };

    _ws = ws;
  } catch (e) {
    console.error('[wsConnection] Failed to open:', e);
  }
}

export function disconnectWs(): void {
  _ws?.close();
  _ws = null;
}

export function sendWsMessage(msg: Record<string, unknown>): void {
  if (_ws?.readyState === WebSocket.OPEN) {
    _ws.send(JSON.stringify(msg));
  } else {
    console.warn('[wsConnection] Not connected — message dropped:', msg);
  }
}
