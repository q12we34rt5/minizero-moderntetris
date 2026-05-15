import { useCallback, useEffect, useRef, useState } from 'react';
import { Engine } from '../engine/engine.ts';
import type { GameView } from '../engine/view.ts';
import { InputController, DEFAULT_SETTINGS, Action, type InputSettings } from '../input/controller.ts';
import { DEFAULT_GAMEPAD_MAPPING, type GamepadMapping, type GamepadStatus } from '../input/gamepad.ts';
import { AiClient, type AiConnectionStatus } from '../ai/client.ts';

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
}

export interface PveSettings {
  aiIntervalMs: number;
  garbageDelay: number;
  backendUrlA: string;
  backendUrlB: string;
}

const DEFAULT_PVE: PveSettings = {
  aiIntervalMs: 400,
  garbageDelay: 1,
  backendUrlA: 'ws://localhost:8001',
  backendUrlB: 'ws://localhost:8001',
};

const SETTINGS_KEY = 'moderntetris-web-settings';
const PVE_KEY = 'moderntetris-web-pve';
const GAMEPAD_KEY = 'moderntetris-web-gamepad';

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

function makeHud(view: GameView, clock: BoardClock): GameHud {
  let pps = 0;
  let apm = 0;
  if (clock.firstPiece) {
    const elapsed = (performance.now() - clock.startTime) / 1000;
    if (elapsed > 0) {
      pps = view.pieceCount / elapsed;
      apm = (view.totalAttack / elapsed) * 60;
    }
  }
  return { view, pps, apm };
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
  gamepadMapping: GamepadMapping;
  setGamepadMapping: (m: GamepadMapping) => void;
  gamepadStatus: GamepadStatus;
  seed: string;
  setSeed: (s: string) => void;
  reset: () => void;
  reconnectAi: () => void;
  togglePause: () => void;
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
  const [gamepadMapping, setGamepadMappingState] = useState<GamepadMapping>(() => load(GAMEPAD_KEY, DEFAULT_GAMEPAD_MAPPING));
  const [gamepadStatus, setGamepadStatus] = useState<GamepadStatus>({ connected: false, id: null });
  const [seed, setSeedState] = useState('');

  const settingsRef = useRef(settings);
  const pveSettingsRef = useRef(pveSettings);
  const gamepadMappingRef = useRef(gamepadMapping);
  const modeRef = useRef(mode);
  const seedRef = useRef(seed);
  const statusRef = useRef<GameStatus>(status);

  // Filled in by the init effect.
  const resetRef = useRef<() => void>(() => {});
  const reconnectRef = useRef<() => void>(() => {});
  const pauseRef = useRef<() => void>(() => {});

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

  const setGamepadMapping = useCallback((m: GamepadMapping) => {
    gamepadMappingRef.current = m;
    setGamepadMappingState(m);
    save(GAMEPAD_KEY, m);
  }, []);

  const setSeed = useCallback((s: string) => {
    seedRef.current = s;
    setSeedState(s);
  }, []);

  const reset = useCallback(() => resetRef.current(), []);
  const reconnectAi = useCallback(() => reconnectRef.current(), []);
  const togglePause = useCallback(() => pauseRef.current(), []);

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
      boardA = { id: 'A', engine: ea, client: clientA, control: 'human', clock: newClock(), pending: false, lastRequest: 0, anim: null };
      boardB = { id: 'B', engine: eb, client: clientB, control: 'none', clock: newClock(), pending: false, lastRequest: 0, anim: null };

      const garbageDelay = () => pveSettingsRef.current.garbageDelay;
      const sendGarbage = (lines: number, opponent: BoardRuntime) => {
        if (lines > 0 && opponent.control !== 'none') opponent.engine.addGarbage(lines, garbageDelay());
      };

      // Request one AI placement, then play it out as an animation (see
      // advanceAnim) rather than snapping to the final position.
      const aiTick = (board: BoardRuntime, opponent: BoardRuntime, now: number) => {
        if (board.control !== 'ai' || board.client.status !== 'connected') return;
        if (board.pending || board.anim) return;
        if (now - board.lastRequest < pveSettingsRef.current.aiIntervalMs) return;
        board.pending = true;
        board.lastRequest = now;
        const requestGen = gen;
        const state = board.engine.serializeFull();
        board.client
          .requestMove(state)
          .then((placement) => {
            board.pending = false;
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
            const interval = pveSettingsRef.current.aiIntervalMs;
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
          board.engine.step(anim.actions[anim.index]);
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
          board.engine.step(a);
          if (a !== Action.HARD_DROP) continue;
          const v = board.engine.read();
          if (v.pieceCount <= board.clock.pieceCount) continue;
          board.clock.pieceCount = v.pieceCount;
          if (!board.clock.firstPiece) {
            board.clock.firstPiece = true;
            board.clock.startTime = now;
          }
          sendGarbage(v.linesSent, opponent);
          for (const e of input.onNewPiece(now)) board.engine.step(e);
        }
        if (!board.engine.read().isAlive) {
          endGame(opponent.control === 'none' ? null : opponent.id);
        }
      };

      const syncClient = (board: BoardRuntime, url: string) => {
        if (board.control === 'ai') {
          if (board.client.status === 'disconnected') board.client.connect(url);
        } else {
          board.client.disconnect();
        }
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
        // pve: both boards share the bag (fair race). eve: derive a distinct
        // seed for B so a same-model match still diverges into a real game.
        const seedB = m === 'eve' ? (seedA ^ 0x5bd1e995) >>> 0 : seedA;

        boardA.engine.setConfig(0, false); // piece_life disabled, client-side gravity
        boardA.engine.reset(seedA);
        boardA.clock = newClock();
        boardA.pending = false;
        boardA.lastRequest = 0;
        boardA.anim = null;

        if (cB !== 'none') {
          boardB.engine.setConfig(0, false);
          boardB.engine.reset(seedB);
          boardB.clock = newClock();
          boardB.pending = false;
          boardB.lastRequest = 0;
          boardB.anim = null;
        }

        syncClient(boardA, pveSettingsRef.current.backendUrlA);
        syncClient(boardB, pveSettingsRef.current.backendUrlB);

        const now = performance.now();
        input.reset(now);
        input.setEnabled(cA === 'human');

        setWinner(null);
        setStatusBoth('playing');
        setHud(makeHud(boardA.engine.read(), boardA.clock));
        setAiHud(cB !== 'none' ? makeHud(boardB.engine.read(), boardB.clock) : null);
      };
      resetRef.current = doReset;

      reconnectRef.current = () => {
        if (!boardA || !boardB) return;
        if (boardA.control === 'ai') boardA.client.connect(pveSettingsRef.current.backendUrlA);
        if (boardB.control === 'ai') boardB.client.connect(pveSettingsRef.current.backendUrlB);
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
        setHud(makeHud(boardA.engine.read(), boardA.clock));
        setAiHud(boardB.control !== 'none' ? makeHud(boardB.engine.read(), boardB.clock) : null);
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
    gamepadMapping,
    setGamepadMapping,
    gamepadStatus,
    seed,
    setSeed,
    reset,
    reconnectAi,
    togglePause,
  };
}
