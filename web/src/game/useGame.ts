import { useCallback, useEffect, useRef, useState } from 'react';
import { Engine } from '../engine/engine.ts';
import type { GameView } from '../engine/view.ts';
import { InputController, DEFAULT_SETTINGS, Action, type InputSettings } from '../input/controller.ts';
import {
  DEFAULT_GAMEPAD_MAPPING,
  DEFAULT_GAMEPAD_OPTIONS,
  type GamepadMapping,
  type GamepadOptions,
  type GamepadStatus,
} from '../input/gamepad.ts';
import { DEFAULT_KEYBOARD_MAPPING, type KeyboardMapping } from '../input/keyboard.ts';
import { AiClient, type AiConnectionStatus } from '../ai/client.ts';
import { ColorTracker } from './color-tracker.ts';

export type GameMode = 'single' | 'pve' | 'eve';
export type GameStatus = 'loading' | 'playing' | 'paused' | 'gameover';
/** Which board won: 'A' (left) or 'B' (right); null = no opponent (single). */
export type Winner = 'A' | 'B' | null;

/** How a board is driven. */
type Control = 'human' | 'ai' | 'none';

export interface GameHud {
  view: GameView;
  pps: number;
  apm: number;
  app: number; // attack per piece
  /** Extra AI info for this board (key->value), rendered generically. Empty for human boards. */
  aiInfo: Record<string, string>;
  /** Per-cell piece colors for locked blocks, aligned with view.board. */
  colors: Int8Array | null;
}

/** One selectable model, as advertised by the router's GET /models. */
export interface ModelInfo {
  id: string;
  displayName: string;
  desc?: string;
}

export interface PveSettings {
  /** Base URL of the model router (ws:// or wss://). /models and /model/<id> hang off this. */
  routerUrl: string;
  /**
   * Selected model ids, kept separate per mode so tweaking the PvE opponent
   * doesn't disturb an EvE matchup (and vice versa). Empty string = fall back
   * to the first available model.
   */
  pveModel: string; // PvE AI opponent (board B)
  eveModelA: string; // EvE board A
  eveModelB: string; // EvE board B
  /**
   * Per-mode timing, independent between PvE and EvE. PvE has a single AI
   * (board B) so it keeps one pair; EvE splits timing per board (A/B) so the
   * two AIs can run at different speeds.
   */
  pveAiIntervalMs: number;
  pveGarbageDelay: number;
  eveAiIntervalMsA: number;
  eveGarbageDelayA: number;
  eveAiIntervalMsB: number;
  eveGarbageDelayB: number;
  /** EvE only: start both boards from the same seed (identical piece sequence). */
  eveSeedSync: boolean;
}

const DEFAULT_PVE: PveSettings = {
  routerUrl: 'ws://localhost:8000',
  pveModel: '',
  eveModelA: '',
  eveModelB: '',
  pveAiIntervalMs: 400,
  pveGarbageDelay: 1,
  eveAiIntervalMsA: 400,
  eveGarbageDelayA: 1,
  eveAiIntervalMsB: 400,
  eveGarbageDelayB: 1,
  eveSeedSync: false,
};

/** HTTP(S) base for the router's /models endpoint, derived from the ws(s):// URL. */
function routerHttpBase(routerUrl: string): string {
  return routerUrl.replace(/\/+$/, '').replace(/^ws/, 'http');
}

/** WS URL for a specific model id via the router, or null if unroutable. */
function modelWsUrl(routerUrl: string, id: string): string | null {
  const base = routerUrl.replace(/\/+$/, '');
  if (!base || !id) return null;
  return `${base}/model/${encodeURIComponent(id)}`;
}

// Game-rule settings that change engine behavior in every mode (single/pve/eve).
// all_spin rides the serialized state to the AI backend, so it must match the
// backend's env_modern_tetris_all_spin in pve/eve.
export interface RulesSettings {
  allSpin: boolean;
}

const DEFAULT_RULES: RulesSettings = {
  allSpin: false,
};

const SETTINGS_KEY = 'moderntetris-web-settings';
const PVE_KEY = 'moderntetris-web-pve';
const RULES_KEY = 'moderntetris-web-rules';
const GAMEPAD_KEY = 'moderntetris-web-gamepad';
const GAMEPAD_OPTS_KEY = 'moderntetris-web-gamepad-opts';
const KEYBOARD_KEY = 'moderntetris-web-keyboard';

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw) return { ...fallback, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return fallback;
}

function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

function controlsForMode(m: GameMode): [Control, Control] {
  if (m === 'single') return ['human', 'none'];
  if (m === 'pve') return ['human', 'ai'];
  return ['ai', 'ai'];
}

/** Per-board running stats (pieces-per-second, attack-per-minute). */
interface BoardClock {
  pieceCount: number;
  startTime: number;
  firstPiece: boolean;
}

function newClock(): BoardClock {
  return { pieceCount: 0, startTime: 0, firstPiece: false };
}

function makeHud(
  view: GameView,
  clock: BoardClock,
  aiInfo: Record<string, string> = {},
  colors: Int8Array | null = null,
): GameHud {
  let pps = 0;
  let apm = 0;
  if (clock.firstPiece) {
    const elapsed = (performance.now() - clock.startTime) / 1000;
    if (elapsed > 0) {
      pps = view.pieceCount / elapsed;
      apm = (view.totalAttack / elapsed) * 60;
    }
  }
  const app = view.pieceCount > 0 ? view.totalAttack / view.pieceCount : 0;
  return { view, pps, apm, app, aiInfo, colors };
}

/** An in-progress AI move being played out step-by-step for animation. */
interface AiAnim {
  actions: number[]; // step-action ids: [HOLD?] + path moves + HARD_DROP
  index: number; // next action to apply
  lastStepTime: number;
  stepInterval: number;
}

/** Runtime state for one board (A = left, B = right). */
interface BoardRuntime {
  id: 'A' | 'B';
  engine: Engine;
  client: AiClient;
  control: Control;
  clock: BoardClock;
  pending: boolean; // an AI request is in flight
  lastRequest: number;
  anim: AiAnim | null; // an AI move currently being animated
  aiInfo: Record<string, string>; // latest extra info from the AI backend (value/winloss/...)
  colors: ColorTracker; // piece colors of the locked cells (the engine board has none)
  connectedUrl: string | null; // the /model/<id> URL the client is currently pointed at
}

export interface UseGame {
  mode: GameMode;
  setMode: (m: GameMode) => void;
  status: GameStatus;
  winner: Winner;
  hud: GameHud | null;
  aiHud: GameHud | null;
  aiStatusA: AiConnectionStatus;
  aiStatusB: AiConnectionStatus;
  settings: InputSettings;
  setSettings: (s: InputSettings) => void;
  pveSettings: PveSettings;
  setPveSettings: (s: PveSettings) => void;
  /** Models advertised by the router (for the per-board model picker). */
  models: ModelInfo[];
  /** Re-fetch the model list from the router (e.g. after editing the registry). */
  refreshModels: () => void;
  rules: RulesSettings;
  setRules: (r: RulesSettings) => void;
  gamepadMapping: GamepadMapping;
  setGamepadMapping: (m: GamepadMapping) => void;
  gamepadOptions: GamepadOptions;
  setGamepadOptions: (o: GamepadOptions) => void;
  keyboardMapping: KeyboardMapping;
  setKeyboardMapping: (m: KeyboardMapping) => void;
  gamepadStatus: GamepadStatus;
  seed: string;
  setSeed: (s: string) => void;
  reset: () => void;
  reconnectAi: () => void;
  togglePause: () => void;
  dumpState: () => string;
}

export function useGame(): UseGame {
  const [mode, setModeState] = useState<GameMode>('single');
  const [status, setStatus] = useState<GameStatus>('loading');
  const [winner, setWinner] = useState<Winner>(null);
  const [hud, setHud] = useState<GameHud | null>(null);
  const [aiHud, setAiHud] = useState<GameHud | null>(null);
  const [aiStatusA, setAiStatusA] = useState<AiConnectionStatus>('disconnected');
  const [aiStatusB, setAiStatusB] = useState<AiConnectionStatus>('disconnected');
  const [settings, setSettingsState] = useState<InputSettings>(() => load(SETTINGS_KEY, DEFAULT_SETTINGS));
  const [pveSettings, setPveSettingsState] = useState<PveSettings>(() => load(PVE_KEY, DEFAULT_PVE));
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [rules, setRulesState] = useState<RulesSettings>(() => load(RULES_KEY, DEFAULT_RULES));
  const [gamepadMapping, setGamepadMappingState] = useState<GamepadMapping>(() => load(GAMEPAD_KEY, DEFAULT_GAMEPAD_MAPPING));
  const [gamepadOptions, setGamepadOptionsState] = useState<GamepadOptions>(() => load(GAMEPAD_OPTS_KEY, DEFAULT_GAMEPAD_OPTIONS));
  const [keyboardMapping, setKeyboardMappingState] = useState<KeyboardMapping>(() => load(KEYBOARD_KEY, DEFAULT_KEYBOARD_MAPPING));
  const [gamepadStatus, setGamepadStatus] = useState<GamepadStatus>({ connected: false, id: null });
  const [seed, setSeedState] = useState('');

  const settingsRef = useRef(settings);
  const pveSettingsRef = useRef(pveSettings);
  const modelsRef = useRef<ModelInfo[]>(models); // effect-side mirror of the fetched model list
  const rulesRef = useRef(rules);
  const gamepadMappingRef = useRef(gamepadMapping);
  const gamepadOptionsRef = useRef(gamepadOptions);
  const keyboardMappingRef = useRef(keyboardMapping);
  const modeRef = useRef(mode);
  const seedRef = useRef(seed);
  const statusRef = useRef<GameStatus>(status);

  // Filled in by the init effect.
  const resetRef = useRef<() => void>(() => {});
  const reconnectRef = useRef<() => void>(() => {});
  const syncClientsRef = useRef<() => void>(() => {}); // re-point AI boards at their selected models
  const pauseRef = useRef<() => void>(() => {});
  const dumpStateRef = useRef<() => string>(() => '');

  const setSettings = useCallback((s: InputSettings) => {
    settingsRef.current = s;
    setSettingsState(s);
    save(SETTINGS_KEY, s);
  }, []);

  const setPveSettings = useCallback((s: PveSettings) => {
    pveSettingsRef.current = s;
    setPveSettingsState(s);
    save(PVE_KEY, s);
  }, []);

  const setRules = useCallback((r: RulesSettings) => {
    rulesRef.current = r;
    setRulesState(r);
    save(RULES_KEY, r);
  }, []);

  const setGamepadMapping = useCallback((m: GamepadMapping) => {
    gamepadMappingRef.current = m;
    setGamepadMappingState(m);
    save(GAMEPAD_KEY, m);
  }, []);

  const setGamepadOptions = useCallback((o: GamepadOptions) => {
    gamepadOptionsRef.current = o;
    setGamepadOptionsState(o);
    save(GAMEPAD_OPTS_KEY, o);
  }, []);

  const setKeyboardMapping = useCallback((m: KeyboardMapping) => {
    keyboardMappingRef.current = m;
    setKeyboardMappingState(m);
    save(KEYBOARD_KEY, m);
  }, []);

  const setSeed = useCallback((s: string) => {
    seedRef.current = s;
    setSeedState(s);
  }, []);

  // Fetch the router's model list. Keeps both the React state (for the UI) and
  // the effect-side ref (for URL resolution) in sync. On any failure the list
  // is emptied, so the picker shows "no models" and boards stay disconnected.
  const refreshModels = useCallback(() => {
    const url = routerHttpBase(pveSettingsRef.current.routerUrl) + '/models';
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: Array<{ id: string; display_name?: string; desc?: string }>) => {
        const list: ModelInfo[] = Array.isArray(data)
          ? data.map((m) => ({ id: String(m.id), displayName: String(m.display_name ?? m.id), desc: m.desc }))
          : [];
        modelsRef.current = list;
        setModels(list);
        // Drop selections this router doesn't advertise. A stale id (registry
        // edited, or a different router) would otherwise sit in localStorage
        // forever and silently outvote the picker: <select> shows the first
        // option when its value matches none, so re-picking that option fires
        // no change and the old id keeps getting connected. Only when the fetch
        // actually returned models -- an empty list means the router is down,
        // and that must not wipe the operator's choice.
        if (list.length > 0) {
          const s = pveSettingsRef.current;
          const keep = (id: string) => (id === '' || list.some((m) => m.id === id) ? id : '');
          const [pve, a, b] = [keep(s.pveModel), keep(s.eveModelA), keep(s.eveModelB)];
          if (pve !== s.pveModel || a !== s.eveModelA || b !== s.eveModelB) {
            setPveSettings({ ...s, pveModel: pve, eveModelA: a, eveModelB: b });
          }
        }
      })
      .catch(() => {
        modelsRef.current = [];
        setModels([]);
      });
  }, [setPveSettings]);

  const reset = useCallback(() => resetRef.current(), []);
  const reconnectAi = useCallback(() => {
    refreshModels(); // operator may have edited the registry; pick it up on reconnect
    reconnectRef.current();
  }, [refreshModels]);
  const togglePause = useCallback(() => pauseRef.current(), []);
  const dumpState = useCallback(() => dumpStateRef.current(), []);

  // (Re)fetch models whenever the router URL changes (and once on mount).
  useEffect(() => {
    refreshModels();
  }, [pveSettings.routerUrl, refreshModels]);

  // Live-apply model switches: as soon as a board's selected model (or the
  // router URL / mode) changes, re-point that AI board at the new /model/<id>
  // without waiting for a Reset. The model list also arrives after the first
  // reset, so re-sync when it does. syncClient only reconnects boards whose
  // target URL actually changed, so unrelated tweaks (interval, garbage) don't
  // fire it. No-op before the init effect has filled syncClientsRef in.
  useEffect(() => {
    syncClientsRef.current();
  }, [models, mode, pveSettings.routerUrl, pveSettings.pveModel, pveSettings.eveModelA, pveSettings.eveModelB]);

  const setMode = useCallback((m: GameMode) => {
    modeRef.current = m;
    setModeState(m);
    resetRef.current(); // switching mode starts a fresh game in that mode
  }, []);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let gen = 0; // bumped on every reset; stale async AI responses are dropped
    let boardA: BoardRuntime | null = null;
    let boardB: BoardRuntime | null = null;
    let input: InputController | null = null;

    const clientA = new AiClient();
    const clientB = new AiClient();
    const offA = clientA.onStatus(setAiStatusA);
    const offB = clientB.onStatus(setAiStatusB);

    const setStatusBoth = (s: GameStatus) => {
      statusRef.current = s;
      setStatus(s);
    };

    const endGame = (w: Winner) => {
      if (statusRef.current === 'gameover') return;
      setStatusBoth('gameover');
      setWinner(w);
      input?.setEnabled(false);
    };

    Promise.all([Engine.create(), Engine.create()]).then(([ea, eb]) => {
      if (cancelled) {
        ea.dispose();
        eb.dispose();
        return;
      }
      boardA = { id: 'A', engine: ea, client: clientA, control: 'human', clock: newClock(), pending: false, lastRequest: 0, anim: null, aiInfo: {}, colors: new ColorTracker(), connectedUrl: null };
      boardB = { id: 'B', engine: eb, client: clientB, control: 'none', clock: newClock(), pending: false, lastRequest: 0, anim: null, aiInfo: {}, colors: new ColorTracker(), connectedUrl: null };

      // Every board mutation the color plane cares about happens on a hard drop
      // (the engine locks pieces only in hardDrop()), so track around that step
      // and let every other action go straight through.
      const stepBoard = (board: BoardRuntime, action: number) => {
        if (action !== Action.HARD_DROP) {
          board.engine.step(action);
          return;
        }
        board.colors.noteLock(board.engine.read());
        board.engine.step(action);
        board.colors.sync(board.engine.read());
      };

      // Timing knobs are per-mode; single has no AI so it never reads these. In
      // EvE they're also per-board so the two AIs can differ; PvE's single pair
      // applies to its one AI (and to garbage landing on either board).
      const aiIntervalMs = (board: BoardRuntime) => {
        const s = pveSettingsRef.current;
        if (modeRef.current !== 'eve') return s.pveAiIntervalMs;
        return board.id === 'A' ? s.eveAiIntervalMsA : s.eveAiIntervalMsB;
      };
      const garbageDelay = (board: BoardRuntime) => {
        const s = pveSettingsRef.current;
        if (modeRef.current !== 'eve') return s.pveGarbageDelay;
        return board.id === 'A' ? s.eveGarbageDelayA : s.eveGarbageDelayB;
      };
      // Garbage lands on the opponent, so use the receiving board's delay.
      const sendGarbage = (lines: number, opponent: BoardRuntime) => {
        if (lines > 0 && opponent.control !== 'none') opponent.engine.addGarbage(lines, garbageDelay(opponent));
      };

      // Request one AI placement, then play it out as an animation (see
      // advanceAnim) rather than snapping to the final position.
      const aiTick = (board: BoardRuntime, opponent: BoardRuntime, now: number) => {
        if (board.control !== 'ai' || board.client.status !== 'connected') return;
        if (board.pending || board.anim) return;
        if (now - board.lastRequest < aiIntervalMs(board)) return;
        board.pending = true;
        board.lastRequest = now;
        const requestGen = gen;
        const state = board.engine.serializeFull();
        // In two-player play, hand the AI the real opponent board so its
        // two-player search sees it. With no opponent (single board) the
        // backend falls back to an empty opponent.
        const opponentState = opponent.control === 'none' ? undefined : opponent.engine.serializeFull();
        board.client
          .requestMove(state, opponentState)
          .then((move) => {
            board.pending = false;
            board.aiInfo = move.info;
            const placement = move.placement;
            if (cancelled || requestGen !== gen || statusRef.current !== 'playing') return;
            const loser: Winner = opponent.control === 'none' ? null : opponent.id;
            if (placement === null) {
              endGame(loser); // AI resigned / topped out
              return;
            }
            const path = board.engine.placementPath(placement);
            if (path.length === 0) {
              endGame(loser); // no matching placement (should not happen)
              return;
            }
            // Spread the animation over ~60% of the AI interval, capped at
            // 60ms/step. No lower bound: at AI Interval 0 this is 0ms/step, so
            // the whole path applies in one frame (effectively a snap).
            const interval = aiIntervalMs(board);
            board.anim = {
              actions: path,
              index: 0,
              lastStepTime: performance.now(),
              stepInterval: Math.min(60, Math.floor((interval * 0.6) / path.length)),
            };
          })
          .catch(() => {
            board.pending = false; // backend/connection error -> status chip shows it
          });
      };

      // Play out an AI move one step-action per stepInterval. The final action
      // (HARD_DROP) gets a longer beat so the lock reads clearly. When it lands,
      // exchange garbage and check for a top-out.
      const HARD_DROP_FACTOR = 2.5;
      const advanceAnim = (board: BoardRuntime, opponent: BoardRuntime, now: number) => {
        while (board.anim) {
          const anim = board.anim;
          const isHardDrop = anim.index === anim.actions.length - 1;
          const required = isHardDrop
            ? Math.round(anim.stepInterval * HARD_DROP_FACTOR)
            : anim.stepInterval;
          if (now - anim.lastStepTime < required) break;
          stepBoard(board, anim.actions[anim.index]);
          anim.index += 1;
          anim.lastStepTime += required;
          if (anim.index >= anim.actions.length) {
            board.anim = null;
            const av = board.engine.read();
            if (!board.clock.firstPiece) {
              board.clock.firstPiece = true;
              board.clock.startTime = now;
            }
            board.clock.pieceCount = av.pieceCount;
            const loser: Winner = opponent.control === 'none' ? null : opponent.id;
            sendGarbage(av.linesSent, opponent);
            if (!av.isAlive) endGame(loser);
          }
        }
      };

      // Keyboard-driven board: collect actions, step, exchange garbage on lock.
      const runHumanBoard = (board: BoardRuntime, opponent: BoardRuntime, now: number) => {
        if (!input) return;
        const actions = input.collect(now);
        for (const a of actions) {
          stepBoard(board, a);
          if (a !== Action.HARD_DROP) continue;
          const v = board.engine.read();
          if (v.pieceCount <= board.clock.pieceCount) continue;
          board.clock.pieceCount = v.pieceCount;
          if (!board.clock.firstPiece) {
            board.clock.firstPiece = true;
            board.clock.startTime = now;
          }
          sendGarbage(v.linesSent, opponent);
          for (const e of input.onNewPiece(now)) stepBoard(board, e);
        }
        if (!board.engine.read().isAlive) {
          endGame(opponent.control === 'none' ? null : opponent.id);
        }
      };

      // The /model/<id> URL an AI board should be pointed at, from the router
      // base + its selected model (empty selection falls back to the first
      // advertised model). null = unroutable (no base / no models loaded yet).
      const resolveUrl = (boardId: 'A' | 'B'): string | null => {
        const s = pveSettingsRef.current;
        // PvE only has one AI (board B) with its own model; EvE picks per board.
        const selected =
          modeRef.current === 'pve' ? s.pveModel : boardId === 'A' ? s.eveModelA : s.eveModelB;
        // An id the router doesn't advertise resolves like "auto" rather than
        // being sent as-is, so the connection always matches what the picker
        // shows (which falls back to the first model for an unknown value).
        const known = modelsRef.current.some((m) => m.id === selected);
        const id = (known ? selected : '') || modelsRef.current[0]?.id || '';
        return modelWsUrl(s.routerUrl, id);
      };

      // Connect an AI board to its resolved model URL; reconnect if the target
      // changed (model switch) and disconnect non-AI boards. Idempotent when
      // already pointed at the right URL, so it's safe to call every reset.
      const syncClient = (board: BoardRuntime, url: string | null) => {
        if (board.control !== 'ai' || !url) {
          board.client.disconnect();
          board.connectedUrl = null;
          return;
        }
        if (board.client.status === 'disconnected' || board.connectedUrl !== url) {
          board.client.connect(url);
          board.connectedUrl = url;
        }
      };

      // Re-point both boards at their currently-selected models. Idempotent, so
      // the "model changed" effect can call it every render without churn.
      syncClientsRef.current = () => {
        if (!boardA || !boardB) return;
        syncClient(boardA, resolveUrl('A'));
        syncClient(boardB, resolveUrl('B'));
      };

      const doReset = () => {
        if (!boardA || !boardB || !input) return;
        gen++;
        const m = modeRef.current;
        const [cA, cB] = controlsForMode(m);
        boardA.control = cA;
        boardB.control = cB;

        const trimmed = seedRef.current.trim();
        const parsed = trimmed === '' ? NaN : Number(trimmed);
        const seedA = Number.isFinite(parsed) ? parsed >>> 0 : (Math.random() * 0x100000000) >>> 0;
        // pve always shares the bag (fair race). eve defaults to a distinct seed
        // for B so a same-model match still diverges into a real game, but the
        // operator can force an identical start via eveSeedSync.
        const seedB =
          m === 'eve' && !pveSettingsRef.current.eveSeedSync ? (seedA ^ 0x5bd1e995) >>> 0 : seedA;

        const allSpin = rulesRef.current.allSpin;
        boardA.engine.setConfig(0, false, allSpin); // piece_life disabled, client-side gravity
        boardA.engine.reset(seedA);
        boardA.clock = newClock();
        boardA.pending = false;
        boardA.lastRequest = 0;
        boardA.anim = null;
        boardA.colors.reset();

        if (cB !== 'none') {
          boardB.engine.setConfig(0, false, allSpin);
          boardB.engine.reset(seedB);
          boardB.clock = newClock();
          boardB.pending = false;
          boardB.lastRequest = 0;
          boardB.anim = null;
          boardB.colors.reset();
        }

        syncClient(boardA, resolveUrl('A'));
        syncClient(boardB, resolveUrl('B'));

        const now = performance.now();
        input.reset(now);
        input.setEnabled(cA === 'human');

        setWinner(null);
        setStatusBoth('playing');
        setHud(makeHud(boardA.engine.read(), boardA.clock, boardA.aiInfo, boardA.colors.visible()));
        setAiHud(cB !== 'none' ? makeHud(boardB.engine.read(), boardB.clock, boardB.aiInfo, boardB.colors.visible()) : null);
      };
      resetRef.current = doReset;

      // Dump the live engine state(s) as backend `set_state` commands. The flat
      // codec ints are exactly what cmdSetState expects, so a dumped line can be
      // pasted into the console to reproduce a position (e.g. a sudden game over).
      dumpStateRef.current = () => {
        if (!boardA) return '';
        const dumpBoard = (b: BoardRuntime) => `# Board ${b.id} (alive=${b.engine.read().isAlive})\nset_state ${Array.from(b.engine.serializeFull()).join(' ')}`;
        const parts = [dumpBoard(boardA)];
        if (boardB && boardB.control !== 'none') parts.push(dumpBoard(boardB));
        return parts.join('\n\n');
      };

      reconnectRef.current = () => {
        if (!boardA || !boardB) return;
        for (const board of [boardA, boardB]) {
          if (board.control !== 'ai') continue;
          const url = resolveUrl(board.id);
          if (!url) continue;
          board.client.connect(url);
          board.connectedUrl = url;
        }
      };

      pauseRef.current = () => {
        if (statusRef.current === 'playing') {
          setStatusBoth('paused');
          input?.setEnabled(false);
        } else if (statusRef.current === 'paused') {
          setStatusBoth('playing');
          if (boardA && boardA.control === 'human') input?.setEnabled(true);
        }
      };

      input = new InputController({
        getSettings: () => settingsRef.current,
        getGamepadMapping: () => gamepadMappingRef.current,
        getGamepadOptions: () => gamepadOptionsRef.current,
        getKeyboardMapping: () => keyboardMappingRef.current,
        onReset: () => doReset(),
        onGamepadStatus: (s) => setGamepadStatus(s),
      });
      input.attach();

      const loop = () => {
        raf = requestAnimationFrame(loop);
        if (statusRef.current !== 'playing' || !boardA || !boardB) return;
        const now = performance.now();

        if (boardA.control === 'human') {
          runHumanBoard(boardA, boardB, now);
        } else if (boardA.control === 'ai') {
          advanceAnim(boardA, boardB, now);
          aiTick(boardA, boardB, now);
        }
        if (boardB.control === 'ai') {
          advanceAnim(boardB, boardA, now);
          aiTick(boardB, boardA, now);
        }

        // Publish every frame so stats / garbage meters stay current even when
        // a board is idle (e.g. garbage arriving from the opponent).
        setHud(makeHud(boardA.engine.read(), boardA.clock, boardA.aiInfo, boardA.colors.visible()));
        setAiHud(boardB.control !== 'none' ? makeHud(boardB.engine.read(), boardB.clock, boardB.aiInfo, boardB.colors.visible()) : null);
      };

      doReset();
      raf = requestAnimationFrame(loop);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      offA();
      offB();
      clientA.disconnect();
      clientB.disconnect();
      input?.detach();
      boardA?.engine.dispose();
      boardB?.engine.dispose();
    };
  }, []);

  return {
    mode,
    setMode,
    status,
    winner,
    hud,
    aiHud,
    aiStatusA,
    aiStatusB,
    settings,
    setSettings,
    pveSettings,
    setPveSettings,
    models,
    refreshModels,
    rules,
    setRules,
    gamepadMapping,
    setGamepadMapping,
    gamepadOptions,
    setGamepadOptions,
    keyboardMapping,
    setKeyboardMapping,
    gamepadStatus,
    seed,
    setSeed,
    reset,
    reconnectAi,
    togglePause,
    dumpState,
  };
}
