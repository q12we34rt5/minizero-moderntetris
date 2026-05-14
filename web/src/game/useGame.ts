import { useCallback, useEffect, useRef, useState } from 'react';
import { Engine } from '../engine/engine.ts';
import type { GameView } from '../engine/view.ts';
import { InputController, DEFAULT_SETTINGS, Action, type InputSettings } from '../input/keyboard.ts';
import { AiClient, type AiConnectionStatus } from '../ai/client.ts';

export type GameMode = 'single' | 'pve';
export type GameStatus = 'loading' | 'playing' | 'gameover';
export type Winner = 'player' | 'ai' | null;

export interface GameHud {
  view: GameView;
  pps: number;
  apm: number;
}

export interface PveSettings {
  aiIntervalMs: number;
  garbageDelay: number;
  backendUrl: string;
}

const DEFAULT_PVE: PveSettings = {
  aiIntervalMs: 400,
  garbageDelay: 1,
  backendUrl: 'ws://localhost:8001',
};

const SETTINGS_KEY = 'moderntetris-web-settings';
const PVE_KEY = 'moderntetris-web-pve';

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

export interface UseGame {
  mode: GameMode;
  setMode: (m: GameMode) => void;
  status: GameStatus;
  winner: Winner;
  hud: GameHud | null;
  aiHud: GameHud | null;
  aiStatus: AiConnectionStatus;
  settings: InputSettings;
  setSettings: (s: InputSettings) => void;
  pveSettings: PveSettings;
  setPveSettings: (s: PveSettings) => void;
  seed: string;
  setSeed: (s: string) => void;
  reset: () => void;
  reconnectAi: () => void;
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

export function useGame(): UseGame {
  const [mode, setModeState] = useState<GameMode>('single');
  const [status, setStatus] = useState<GameStatus>('loading');
  const [winner, setWinner] = useState<Winner>(null);
  const [hud, setHud] = useState<GameHud | null>(null);
  const [aiHud, setAiHud] = useState<GameHud | null>(null);
  const [aiStatus, setAiStatus] = useState<AiConnectionStatus>('disconnected');
  const [settings, setSettingsState] = useState<InputSettings>(() => load(SETTINGS_KEY, DEFAULT_SETTINGS));
  const [pveSettings, setPveSettingsState] = useState<PveSettings>(() => load(PVE_KEY, DEFAULT_PVE));
  const [seed, setSeedState] = useState('');

  const settingsRef = useRef(settings);
  const pveSettingsRef = useRef(pveSettings);
  const modeRef = useRef(mode);
  const seedRef = useRef(seed);
  const statusRef = useRef<GameStatus>(status);

  // Filled in by the init effect.
  const resetRef = useRef<() => void>(() => {});
  const reconnectRef = useRef<() => void>(() => {});

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

  const setSeed = useCallback((s: string) => {
    seedRef.current = s;
    setSeedState(s);
  }, []);

  const reset = useCallback(() => resetRef.current(), []);
  const reconnectAi = useCallback(() => reconnectRef.current(), []);

  const setMode = useCallback((m: GameMode) => {
    modeRef.current = m;
    setModeState(m);
    // Switching mode starts a fresh game in that mode.
    resetRef.current();
  }, []);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let playerEngine: Engine | null = null;
    let aiEngine: Engine | null = null;
    let input: InputController | null = null;
    const aiClient = new AiClient();

    // Mutable loop state.
    let gen = 0; // bumped on every reset; stale async AI responses are dropped
    let playerClock = newClock();
    let aiClock = newClock();
    let aiPending = false;
    let aiLastRequest = 0;

    const setStatusBoth = (s: GameStatus) => {
      statusRef.current = s;
      setStatus(s);
    };

    const offStatus = aiClient.onStatus((s) => setAiStatus(s));

    const endGame = (w: Winner) => {
      setStatusBoth('gameover');
      setWinner(w);
      input?.setEnabled(false);
    };

    Promise.all([Engine.create(), Engine.create()]).then(([pe, ae]) => {
      if (cancelled) {
        pe.dispose();
        ae.dispose();
        return;
      }
      playerEngine = pe;
      aiEngine = ae;

      const doReset = () => {
        if (!playerEngine || !input) return;
        gen++;
        const trimmed = seedRef.current.trim();
        const parsed = trimmed === '' ? NaN : Number(trimmed);
        const useSeed = Number.isFinite(parsed) ? parsed >>> 0 : (Math.random() * 0x100000000) >>> 0;

        playerEngine.setConfig(0, false); // piece_life disabled, client-side gravity
        playerEngine.reset(useSeed);
        const now = performance.now();
        input.reset(now);
        input.setEnabled(true);
        playerClock = newClock();

        if (modeRef.current === 'pve' && aiEngine) {
          aiEngine.setConfig(0, false);
          aiEngine.reset(useSeed); // same bag sequence -> fair race
          aiClock = newClock();
          aiPending = false;
          aiLastRequest = 0;
          if (aiClient.status === 'disconnected') aiClient.connect(pveSettingsRef.current.backendUrl);
          setAiHud(makeHud(aiEngine.read(), aiClock));
        } else {
          setAiHud(null);
        }

        setWinner(null);
        setStatusBoth('playing');
        setHud(makeHud(playerEngine.read(), playerClock));
      };
      resetRef.current = doReset;
      reconnectRef.current = () => aiClient.connect(pveSettingsRef.current.backendUrl);

      input = new InputController({
        getSettings: () => settingsRef.current,
        onReset: () => doReset(),
      });
      input.attach();

      const loop = () => {
        raf = requestAnimationFrame(loop);
        if (statusRef.current !== 'playing' || !playerEngine || !input) return;
        const now = performance.now();
        const pve = modeRef.current === 'pve';

        // --- player board (step-level, keyboard) ---
        const actions = input.collect(now);
        for (const a of actions) {
          playerEngine.step(a);
          if (a !== Action.HARD_DROP) continue;
          const v = playerEngine.read();
          if (v.pieceCount <= playerClock.pieceCount) continue;
          playerClock.pieceCount = v.pieceCount;
          if (!playerClock.firstPiece) {
            playerClock.firstPiece = true;
            playerClock.startTime = now;
          }
          if (pve && aiEngine && v.linesSent > 0) {
            aiEngine.addGarbage(v.linesSent, pveSettingsRef.current.garbageDelay);
          }
          for (const e of input.onNewPiece(now)) playerEngine.step(e);
        }
        const playerView = playerEngine.read();
        // Publish every frame so the garbage queue indicator stays current even
        // when garbage arrives from the AI while the player is idle.
        setHud(makeHud(playerView, playerClock));
        if (pve && aiEngine) setAiHud(makeHud(aiEngine.read(), aiClock));
        if (!playerView.isAlive) {
          endGame(pve ? 'ai' : null);
          return;
        }

        // --- AI board (placement-level, backend-driven) ---
        if (
          pve &&
          aiEngine &&
          aiClient.status === 'connected' &&
          !aiPending &&
          now - aiLastRequest >= pveSettingsRef.current.aiIntervalMs
        ) {
          aiPending = true;
          aiLastRequest = now;
          const requestGen = gen;
          const state = aiEngine.serializeFull();
          aiClient
            .requestMove(state)
            .then((placement) => {
              aiPending = false;
              if (cancelled || requestGen !== gen || statusRef.current !== 'playing' || !aiEngine || !playerEngine) {
                return;
              }
              if (placement === null || !aiEngine.applyPlacement(placement)) {
                endGame('player'); // AI resigned / topped out / illegal move
                return;
              }
              const av = aiEngine.read();
              if (!aiClock.firstPiece) {
                aiClock.firstPiece = true;
                aiClock.startTime = performance.now();
              }
              aiClock.pieceCount = av.pieceCount;
              if (av.linesSent > 0) playerEngine.addGarbage(av.linesSent, pveSettingsRef.current.garbageDelay);
              setAiHud(makeHud(av, aiClock));
              if (!av.isAlive) endGame('player');
            })
            .catch(() => {
              aiPending = false;
              // connection/backend error -> stop ticking; status chip shows it
            });
        }
      };

      doReset();
      raf = requestAnimationFrame(loop);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      offStatus();
      aiClient.disconnect();
      input?.detach();
      playerEngine?.dispose();
      aiEngine?.dispose();
    };
  }, []);

  return {
    mode,
    setMode,
    status,
    winner,
    hud,
    aiHud,
    aiStatus,
    settings,
    setSettings,
    pveSettings,
    setPveSettings,
    seed,
    setSeed,
    reset,
    reconnectAi,
  };
}
