import { useEffect, useRef, useState } from 'react';
import {
  buttonLabel,
  captureNextButton,
  DEFAULT_GAMEPAD_MAPPING,
  GAMEPAD_INPUT_LABELS,
  LOGICAL_INPUTS,
  type GamepadMapping,
  type GamepadStatus,
  type LogicalInput,
} from '../input/gamepad.ts';

interface Props {
  status: GamepadStatus;
  mapping: GamepadMapping;
  onChange: (m: GamepadMapping) => void;
}

export function GamepadPanel({ status, mapping, onChange }: Props) {
  const [capturing, setCapturing] = useState<LogicalInput | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  // Cancel an in-progress capture if the panel unmounts.
  useEffect(() => () => cancelRef.current?.(), []);

  const startCapture = (input: LogicalInput) => {
    cancelRef.current?.();
    setCapturing(input);
    cancelRef.current = captureNextButton((buttonIndex) => {
      cancelRef.current = null;
      setCapturing(null);
      onChange({ ...mapping, [input]: buttonIndex });
    });
  };

  const cancelCapture = () => {
    cancelRef.current?.();
    cancelRef.current = null;
    setCapturing(null);
  };

  const resetDefaults = () => {
    cancelCapture();
    onChange({ ...DEFAULT_GAMEPAD_MAPPING });
  };

  return (
    <div className="panel" style={{ flex: 1, minWidth: 320 }}>
      <div className="panel-title">Gamepad</div>
      <div className="gamepad-status">
        <span className={`status-dot ${status.connected ? 'on' : 'off'}`} />
        {status.connected ? status.id : 'No gamepad detected'}
      </div>

      <div className="gamepad-rows">
        {LOGICAL_INPUTS.map((input) => (
          <div className="gamepad-row" key={input}>
            <span className="gamepad-input-label">{GAMEPAD_INPUT_LABELS[input]}</span>
            <button
              className={`gamepad-bind ${capturing === input ? 'capturing' : ''}`}
              disabled={!status.connected && capturing !== input}
              onClick={() => (capturing === input ? cancelCapture() : startCapture(input))}
            >
              {capturing === input ? 'Press a button…' : buttonLabel(mapping[input])}
            </button>
          </div>
        ))}
      </div>

      <button className="btn gamepad-reset" onClick={resetDefaults}>
        Reset to Defaults
      </button>
    </div>
  );
}
