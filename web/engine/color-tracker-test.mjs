// Randomized check of the color tracker (src/game/color-tracker.ts) against the
// real WASM engine. Drives the board exactly like useGame does -- placement path
// replayed step by step, tracking around HARD_DROP -- and asserts that the
// tracked color plane stays in sync with the engine's own board.
//
//   node engine/color-tracker-test.mjs        (run from web/)
//
// Imports a .ts module directly, so it needs a Node with type stripping (>=22.18).
import { ColorTracker, COLOR_UNKNOWN, COLOR_GARBAGE } from '../src/game/color-tracker.ts';
import { parseView, VIEW_SIZE, BOARD_W, BOARD_H } from '../src/engine/view.ts';
import createEngineModule from '../src/engine/engine-wasm.js';

const HARD_DROP = 3;
const HOLD = 7;
const CELLS = BOARD_W * BOARD_H;

const mod = await createEngineModule();
const ctx = mod._et_create();
const viewPtr = mod._malloc(VIEW_SIZE * 4);
const placementsPtr = mod._malloc(512 * 4 * 4);
const pathPtr = mod._malloc(256 * 4);

const read = () => {
  mod._et_serialize(ctx, viewPtr);
  const base = viewPtr >> 2;
  return parseView(mod.HEAP32.subarray(base, base + VIEW_SIZE));
};

// Deterministic RNG so a failure is reproducible.
let rngState = 0x2f6e2b1;
const rnd = () => {
  rngState ^= rngState << 13;
  rngState ^= rngState >>> 17;
  rngState ^= rngState << 5;
  return (rngState >>> 0) / 0x100000000;
};

let failures = 0;
const fail = (msg) => {
  if (failures < 10) console.log(`  FAIL ${msg}`);
  failures++;
};

let drops = 0;
let clears = 0;
let garbageEvents = 0;
let strictChecks = 0;
let unknownCells = 0;

// The color plane must agree with the engine on which cells are occupied and on
// the block/garbage split. A COLOR_UNKNOWN cell means the tracker gave up and
// fell back to a rebuild -- tolerated in the UI, but a bug in this test.
const checkPlane = (tracker, view, tag) => {
  const colors = tracker.visible();
  for (let i = 0; i < CELLS; i++) {
    if ((colors[i] !== -1) !== (view.board[i] !== 0)) {
      fail(`${tag}: occupancy mismatch at r${Math.floor(i / BOARD_W)} c${i % BOARD_W} (color=${colors[i]}, cell=${view.board[i]})`);
      return;
    }
    if (view.board[i] === 3 && colors[i] !== COLOR_GARBAGE) fail(`${tag}: garbage cell ${i} not labeled garbage`);
    if (view.board[i] === 1 && colors[i] === COLOR_GARBAGE) fail(`${tag}: block cell ${i} labeled garbage`);
    if (colors[i] === COLOR_UNKNOWN) unknownCells++;
  }
};

for (let game = 0; game < 40; game++) {
  const tracker = new ColorTracker();
  mod._et_set_config(ctx, 0, 0, game % 2); // piece_life off, auto_drop off, alternate all_spin
  mod._et_reset(ctx, (game * 2654435761) >>> 0);
  tracker.reset();

  for (let move = 0; move < 300; move++) {
    if (!read().isAlive) break;

    // Garbage from an imaginary opponent: half the games play light (so lines
    // actually clear), half heavy (so the garbage shift path gets hammered).
    if (rnd() < (game % 2 === 0 ? 0.02 : 0.25)) {
      mod._et_add_garbage(ctx, 1 + Math.floor(rnd() * 4), Math.floor(rnd() * 2));
      garbageEvents++;
    }
    if (rnd() < 0.15) mod._et_step(ctx, HOLD); // never touches the board

    const n = mod._et_find_placements(ctx, placementsPtr, 512);
    if (n === 0) break;
    // Greedy "lowest landing" with a random tie-break: keeps the stack flat
    // enough that line clears happen regularly.
    const base = placementsPtr >> 2;
    let pick = 0;
    let bestY = -Infinity;
    for (let c = 0; c < n; c++) {
      const y = mod.HEAP32[base + c * 4 + 1] + rnd();
      if (y > bestY) {
        bestY = y;
        pick = c;
      }
    }
    const raw = mod.HEAP32.subarray(base + pick * 4, base + pick * 4 + 4);
    const pathLen = mod._et_placement_path(ctx, 0, raw[0], raw[1], raw[2], raw[3], pathPtr, 256);
    const actions = pathLen > 0
      ? Array.from(mod.HEAP32.subarray(pathPtr >> 2, (pathPtr >> 2) + pathLen))
      : [HARD_DROP]; // canonicalization dropped this pose; just drop in place

    for (const a of actions) {
      if (a !== HARD_DROP) {
        mod._et_step(ctx, a);
        continue;
      }
      const before = read();
      const colorsBefore = Int8Array.from(tracker.visible());
      const lockedType = before.current;
      tracker.noteLock(before);
      mod._et_step(ctx, a);
      const after = read();
      tracker.sync(after);
      drops++;
      if (after.linesCleared > 0) clears++;
      checkPlane(tracker, after, `game${game} move${move}`);

      // Strict oracle for the simple case (nothing cleared, nothing shifted):
      // the cells that turned occupied must be exactly the ones painted with the
      // dropped piece's type, and no other cell may change color.
      let added = 0;
      let removed = 0;
      for (let i = 0; i < CELLS; i++) {
        if (before.board[i] === 0 && after.board[i] !== 0) added++;
        else if (before.board[i] !== 0 && after.board[i] === 0) removed++;
      }
      if (removed === 0 && added === 4 && after.linesCleared === 0) {
        strictChecks++;
        const colorsAfter = tracker.visible();
        for (let i = 0; i < CELLS; i++) {
          const isNew = before.board[i] === 0 && after.board[i] !== 0;
          if (isNew && colorsAfter[i] !== lockedType) {
            fail(`game${game} move${move}: new cell ${i} colored ${colorsAfter[i]}, expected piece ${lockedType}`);
          } else if (!isNew && colorsAfter[i] !== colorsBefore[i]) {
            fail(`game${game} move${move}: untouched cell ${i} changed ${colorsBefore[i]} -> ${colorsAfter[i]}`);
          }
        }
      }
    }
  }
}

console.log(`drops=${drops} clears=${clears} garbage=${garbageEvents} strictChecks=${strictChecks} fallbackCells=${unknownCells} failures=${failures}`);
const ok = failures === 0 && unknownCells === 0 && clears > 0 && garbageEvents > 0;
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
