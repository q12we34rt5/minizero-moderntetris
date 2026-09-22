import { useEffect, useRef, useState } from 'react';
import {
  buttonLabel,
  captureNextButton,
  DEFAULT_GAMEPAD_MAPPING,
  GAMEPAD_INPUT_LABELS,
  LOGICAL_INPUTS,
  type GamepadMapping,
  type GamepadOptions,
  type GamepadStatus,
  type LogicalInput,
} from '../input/gamepad.ts';
import {
  captureNextKey,
  DEFAULT_KEYBOARD_MAPPING,
  keyLabel,
  type KeyboardMapping,
} from '../input/keyboard.ts';

type Tab = 'keyboard' | 'gamepad';

interface Props {
  gamepadStatus: GamepadStatus;
  gamepadMapping: GamepadMapping;
  onGamepadChange: (m: GamepadMapping) => void;
  gamepadOptions: GamepadOptions;
  onGamepadOptionsChange: (o: GamepadOptions) => void;
  keyboardMapping: KeyboardMapping;
  onKeyboardChange: (m: KeyboardMapping) => void;
}

export function InputBindingsPanel({
  gamepadStatus,
  gamepadMapping,
  onGamepadChange,
  gamepadOptions,
  onGamepadOptionsChange,
  keyboardMapping,
  onKeyboardChange,
}: Props) {
  const [tab, setTab] = useState<Tab>('keyboard'); // keyboard first
  // Which input is capturing, and the cancel handle for the active capture.
  const [capturing, setCapturing] = useState<LogicalInput | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const cancelCapture = () => {
    cancelRef.current?.();
    cancelRef.current = null;
    setCapturing(null);
  };

  // Cancel an in-progress capture on unmount or tab switch.
  useEffect(() => () => cancelRef.current?.(), []);
  const switchTab = (t: Tab) => {
    if (t === tab) return;
    cancelCapture();
    setTab(t);
  };

  const startKeyCapture = (input: LogicalInput) => {
    cancelRef.current?.();
    setCapturing(input);
    cancelRef.current = captureNextKey((key) => {
      cancelRef.current = null;
      setCapturing(null);
      onKeyboardChange({ ...keyboardMapping, [input]: key });
    });
  };

  const startButtonCapture = (input: LogicalInput) => {
    cancelRef.current?.();
    setCapturing(input);
    cancelRef.current = captureNextButton((buttonIndex) => {
      cancelRef.current = null;
      setCapturing(null);
      onGamepadChange({ ...gamepadMapping, [input]: buttonIndex });
    });
  };

  return (
    <div className="panel" style={{ flex: 1, minWidth: 320 }}>
      <div className="tab-row">
        <button
          className={`btn ${tab === 'keyboard' ? 'btn-primary' : ''}`}
          onClick={() => switchTab('keyboard')}
        >
          Keyboard
        </button>
        <button
          className={`btn ${tab === 'gamepad' ? 'btn-primary' : ''}`}
          onClick={() => switchTab('gamepad')}
        >
          Gamepad
        </button>
      </div>

      {tab === 'keyboard' ? (
        <>
          <div className="gamepad-status">
            <span className="status-dot on" />
            Click a binding, then press a key
          </div>
          <div className="gamepad-rows">
            {LOGICAL_INPUTS.map((input) => (
              <div className="gamepad-row" key={input}>
                <span className="gamepad-input-label">{GAMEPAD_INPUT_LABELS[input]}</span>
                <button
                  className={`gamepad-bind ${capturing === input ? 'capturing' : ''}`}
                  onClick={() => (capturing === input ? cancelCapture() : startKeyCapture(input))}
                >
                  {capturing === input ? 'Press a key…' : keyLabel(keyboardMapping[input])}
                </button>
              </div>
            ))}
          </div>
          <button
            className="btn gamepad-reset"
            onClick={() => {
              cancelCapture();
              onKeyboardChange({ ...DEFAULT_KEYBOARD_MAPPING });
            }}
          >
            Reset to Defaults
          </button>
          <p className="hint">R resets the game (fixed). Changes apply immediately.</p>
        </>
      ) : (
        <>
          <div className="gamepad-status">
            <span className={`status-dot ${gamepadStatus.connected ? 'on' : 'off'}`} />
            {gamepadStatus.connected ? gamepadStatus.id : 'No gamepad detected'}
          </div>
          <div className="gamepad-rows">
            {LOGICAL_INPUTS.map((input) => (
              <div className="gamepad-row" key={input}>
                <span className="gamepad-input-label">{GAMEPAD_INPUT_LABELS[input]}</span>
                <button
                  className={`gamepad-bind ${capturing === input ? 'capturing' : ''}`}
                  disabled={!gamepadStatus.connected && capturing !== input}
                  onClick={() => (capturing === input ? cancelCapture() : startButtonCapture(input))}
                >
                  {capturing === input ? 'Press a button…' : buttonLabel(gamepadMapping[input])}
                </button>
              </div>
            ))}
          </div>
          <div className="controls-row" style={{ marginTop: 12 }}>
            <label htmlFor="set-gp-block-harddrop">Block hard drop while other input is held</label>
            <input
              id="set-gp-block-harddrop"
              type="checkbox"
              checked={gamepadOptions.blockHardDropWhileInput}
              onChange={(e) =>
                onGamepadOptionsChange({ ...gamepadOptions, blockHardDropWhileInput: e.target.checked })
              }
            />
          </div>

          <button
            className="btn gamepad-reset"
            onClick={() => {
              cancelCapture();
              onGamepadChange({ ...DEFAULT_GAMEPAD_MAPPING });
            }}
          >
            Reset to Defaults
          </button>
          <p className="hint">
            Keyboard and gamepad work at the same time — either can play. The hard-drop guard
            ignores a hard-drop press while any direction is held or another button fires, to
            prevent accidental drops.
          </p>
        </>
      )}
    </div>
  );
}
