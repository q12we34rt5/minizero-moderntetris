import type { Placement } from '../engine/engine.ts';

export type AiConnectionStatus = 'disconnected' | 'connecting' | 'connected';

interface MoveMessage {
  type: 'move';
  placement: {
    use_hold: number;
    lock_x: number;
    lock_y: number;
    orientation: number;
    spin_type: number;
  } | null;
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

/**
 * WebSocket client for the AI backend. One request in flight at a time — the
 * game loop gates this, and the backend serves requests serially anyway.
 */
export class AiClient {
  private ws: WebSocket | null = null;
  private pending: {
    resolve: (p: Placement | null) => void;
    reject: (e: Error) => void;
  } | null = null;
  private statusListeners = new Set<(s: AiConnectionStatus) => void>();
  status: AiConnectionStatus = 'disconnected';

  onStatus(cb: (s: AiConnectionStatus) => void): () => void {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  private setStatus(s: AiConnectionStatus) {
    this.status = s;
    for (const cb of this.statusListeners) cb(s);
  }

  connect(url: string) {
    this.disconnect();
    this.setStatus('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this.setStatus('disconnected');
      return;
    }
    this.ws = ws;

    ws.onopen = () => this.setStatus('connected');
    ws.onclose = () => {
      if (this.ws === ws) {
        this.ws = null;
        this.setStatus('disconnected');
      }
      this.failPending(new Error('connection closed'));
    };
    ws.onerror = () => {
      this.failPending(new Error('connection error'));
    };
    ws.onmessage = (ev) => {
      let msg: MoveMessage | ErrorMessage;
      try {
        msg = JSON.parse(ev.data as string);
      } catch {
        this.failPending(new Error('invalid response JSON'));
        return;
      }
      const p = this.pending;
      this.pending = null;
      if (!p) return;
      if (msg.type === 'error') {
        p.reject(new Error(msg.message));
      } else if (msg.type === 'move') {
        p.resolve(
          msg.placement === null
            ? null
            : {
                useHold: msg.placement.use_hold !== 0,
                lockX: msg.placement.lock_x,
                lockY: msg.placement.lock_y,
                orientation: msg.placement.orientation,
                spinType: msg.placement.spin_type,
              },
        );
      } else {
        p.reject(new Error('unknown response type'));
      }
    };
  }

  disconnect() {
    this.failPending(new Error('disconnected'));
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      this.ws.onopen = null;
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  private failPending(e: Error) {
    const p = this.pending;
    this.pending = null;
    p?.reject(e);
  }

  /** Request the AI's placement for a serialized engine state. */
  requestMove(state: Int32Array): Promise<Placement | null> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('not connected'));
    }
    if (this.pending) {
      return Promise.reject(new Error('a request is already in flight'));
    }
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject };
      this.ws!.send(JSON.stringify({ type: 'request_move', state: Array.from(state) }));
    });
  }
}
