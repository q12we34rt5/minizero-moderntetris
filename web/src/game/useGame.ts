import { useCallback, useEffect, useRef, useState } from 'react';
import { Engine } from '../engine/engine.ts';
import type { GameView } from '../engine/view.ts';
import { InputController, DEFAULT_SETTINGS, type InputSettings } from '../input/keyboard.ts';

export type GameStatus = 'loading' | 'playing' | 'gameover';

export interface GameHud {
  view: GameView;
  pps: number;
  apm: number;
}

const SETTINGS_KEY = 'moderntetris-web-settings';

function loadSettings(): InputSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return DEFAULT_SETTINGS;
}

export interface UseGame {
  status: GameStatus;
  hud: GameHud | null;
  settings: InputSettings;
  setSettings: (s: InputSettings) => void;
  seed: string;
  setSeed: (s: string) => void;
  reset: () => void;
}

export function useGame(): UseGame {
  const [status, setStatus] = useState<GameStatus>('loading');
  const [hud, setHud] = useState<GameHud | null>(null);
  const [settings, setSettingsState] = useState<InputSettings>(loadSettings);
  const [seed, setSeedState] = useState('');

  const settingsRef = useRef(settings);
  const seedRef = useRef(seed);
  const statusRef = useRef<GameStatus>(status);
  const resetRef = useRef<() => void>(() => {});

  const setSettings = useCallback((s: InputSettings) => {
    settingsRef.current = s;
    setSettingsState(s);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch {
      /* ignore */
    }
  }, []);

  const setSeed = useCallback((s: string) => {
    seedRef.current = s;
    setSeedState(s);
  }, []);

  const reset = useCallback(() => resetRef.current(), []);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let engine: Engine | null = null;
    let input: InputController | null = null;

    let lastPieceCount = 0;
    let startTime = 0;
    let firstPiece = false;

    const setStatusBoth = (s: GameStatus) => {
      statusRef.current = s;
      setStatus(s);
    };

    Engine.create().then((eng) => {
      if (cancelled) {
        eng.dispose();
        return;
      }
      engine = eng;

      const doReset = () => {
        if (!engine || !input) return;
        const trimmed = seedRef.current.trim();
        const parsed = trimmed === '' ? NaN : Number(trimmed);
        const useSeed = Number.isFinite(parsed) ? parsed >>> 0 : (Math.random() * 0x100000000) >>> 0;
        engine.setConfig(0, false); // piece_life disabled, gravity handled client-side
        engine.reset(useSeed);
        const now = performance.now();
        input.reset(now);
        input.setEnabled(true);
        lastPieceCount = 0;
        firstPiece = false;
        startTime = now;
        setStatusBoth('playing');
        setHud({ view: engine.read(), pps: 0, apm: 0 });
      };
      resetRef.current = doReset;

      input = new InputController({
        getSettings: () => settingsRef.current,
        onReset: () => doReset(),
      });
      input.attach();

      const publish = (view: GameView) => {
        let pps = 0;
        let apm = 0;
        if (firstPiece) {
          const elapsed = (performance.now() - startTime) / 1000;
          if (elapsed > 0) {
            pps = view.pieceCount / elapsed;
            apm = (view.totalAttack / elapsed) * 60;
          }
        }
        setHud({ view, pps, apm });
      };

      const loop = () => {
        raf = requestAnimationFrame(loop);
        if (statusRef.current !== 'playing' || !engine || !input) return;

        const now = performance.now();
        const actions = input.collect(now);
        for (const a of actions) engine.step(a);

        let view = engine.read();
        if (view.pieceCount > lastPieceCount) {
          lastPieceCount = view.pieceCount;
          if (!firstPiece) {
            firstPiece = true;
            startTime = now;
          }
          const extra = input.onNewPiece(now);
          if (extra.length > 0) {
            for (const a of extra) engine.step(a);
            view = engine.read();
          }
        }

        if (!view.isAlive) {
          setStatusBoth('gameover');
          input.setEnabled(false);
          publish(view);
          return;
        }

        if (actions.length > 0) publish(view);
      };

      doReset();
      raf = requestAnimationFrame(loop);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      input?.detach();
      engine?.dispose();
    };
  }, []);

  return { status, hud, settings, setSettings, seed, setSeed, reset };
}
