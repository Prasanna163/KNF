/**
 * wsStore.ts — Global singleton message bus for WebSocket events.
 *
 * RunManager publishes each WS message here.
 * Analysis (and any other page) subscribes to receive them.
 * Because this is a module-level singleton it survives React page navigation
 * inside the same browser session.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WsMessage = Record<string, any>;

/** Rolling buffer of the last 500 WS messages. */
export const wsMessageBuffer: WsMessage[] = [];

const _subscribers = new Set<(msg: WsMessage) => void>();

/** Called by RunManager when a new WS message arrives. */
export function publishWsMessage(msg: WsMessage): void {
  wsMessageBuffer.push(msg);
  if (wsMessageBuffer.length > 500) wsMessageBuffer.shift();
  _subscribers.forEach(fn => fn(msg));
}

/** Called by any page that wants to receive live messages. Returns an unsubscribe fn. */
export function subscribeWsMessages(fn: (msg: WsMessage) => void): () => void {
  _subscribers.add(fn);
  return () => _subscribers.delete(fn);
}

/** How many messages are buffered (for initial hydration on page mount). */
export function getBufferedMessages(): WsMessage[] {
  return [...wsMessageBuffer];
}

/** Clear the buffer — call when a new run starts. */
export function clearWsBuffer(): void {
  wsMessageBuffer.length = 0;
}
