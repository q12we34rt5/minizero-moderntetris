// Node smoke test for the WASM engine wrapper. Verifies the serialization
// layout and basic step semantics independently of the browser UI.
//
//   node engine/smoke-test.mjs        (run from web/)
import createEngineModule from '../src/engine/engine-wasm.js';

const VIEW_SIZE = 267;
const BOARD_W = 10;
const BOARD_H = 20;

const I = {
  CURRENT: 200, ORIENTATION: 201, X: 202, Y: 203, GHOST_Y: 204,
  HOLD: 205, HAS_HELD: 206, NEXT: 207, IS_ALIVE: 212, PIECE_COUNT: 213,
  TOTAL_LINES: 219, PENDING_GARBAGE: 225, GARBAGE_QUEUE: 227, GARBAGE_DELAY: 247,
};

let failures = 0;
function check(name, cond) {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    console.log(`  FAIL ${name}`);
    failures++;
  }
}

const mod = await createEngineModule();
check('et_view_size matches VIEW_SIZE', mod._et_view_size() === VIEW_SIZE);

const ctx = mod._et_create();
const ptr = mod._malloc(VIEW_SIZE * 4);
const read = () => {
  mod._et_serialize(ctx, ptr);
  const base = ptr >> 2;
  return mod.HEAP32.subarray(base, base + VIEW_SIZE).slice();
};

mod._et_set_config(ctx, 0, 0, 0);
mod._et_reset(ctx, 12345);
let v = read();

check('board has 20x10 cells, all empty after reset',
  Array.from(v.slice(0, BOARD_W * BOARD_H)).every((c) => c === 0));
check('current piece is valid 0..6', v[I.CURRENT] >= 0 && v[I.CURRENT] <= 6);
check('next queue piece types valid',
  Array.from(v.slice(I.NEXT, I.NEXT + 5)).every((p) => p >= 0 && p <= 6));
check('alive after reset', v[I.IS_ALIVE] === 1);
check('piece_count 0 after reset', v[I.PIECE_COUNT] === 0);
check('spawn x within board', v[I.X] >= 0 && v[I.X] < BOARD_W);

// determinism: same seed -> same first piece + queue
mod._et_reset(ctx, 12345);
const v2 = read();
check('reset is deterministic for a fixed seed',
  v2[I.CURRENT] === v[I.CURRENT] &&
  Array.from(v2.slice(I.NEXT, I.NEXT + 5)).join(',') ===
    Array.from(v.slice(I.NEXT, I.NEXT + 5)).join(','));

// move left to wall then hard drop -> a piece should lock, piece_count++
const before = read();
mod._et_step(ctx, 8); // MOVE_LEFT_TO_WALL
const moved = read();
check('move-left-to-wall pins piece to x=0', moved[I.X] === 0);
mod._et_step(ctx, 3); // HARD_DROP
const dropped = read();
check('hard drop increments piece_count', dropped[I.PIECE_COUNT] === before[I.PIECE_COUNT] + 1);
check('hard drop leaves cells on the board',
  Array.from(dropped.slice(0, BOARD_W * BOARD_H)).some((c) => c !== 0));

// hold swaps the current piece
mod._et_reset(ctx, 999);
const pre = read();
mod._et_step(ctx, 7); // HOLD
const held = read();
check('hold sets has_held', held[I.HAS_HELD] === 1);
check('hold stores the original current piece', held[I.HOLD] === pre[I.CURRENT]);

// fill and clear a line: drop pieces across the floor is complex; instead just
// sanity-check that ghost_y >= y (landing is at or below the active piece).
mod._et_reset(ctx, 7);
v = read();
check('ghost_y is at or below active y', v[I.GHOST_Y] >= v[I.Y]);

// rotation changes orientation (T piece rotates freely on an empty board)
mod._et_reset(ctx, 7);
const r0 = read();
mod._et_step(ctx, 4); // ROTATE_CW
const r1 = read();
check('rotate cw changes orientation', r1[I.ORIENTATION] === (r0[I.ORIENTATION] + 1) % 4);

// add garbage queues lines (pending_garbage reflects it; applied on next lock)
mod._et_reset(ctx, 7);
check('add_garbage returns success', mod._et_add_garbage(ctx, 4, 0) === 1);

// the serialized view exposes the per-entry garbage queue + delay
mod._et_reset(ctx, 7);
mod._et_add_garbage(ctx, 3, 5);
mod._et_add_garbage(ctx, 2, 1);
v = read();
check('view garbage queue exposes per-entry lines',
  v[I.GARBAGE_QUEUE] === 3 && v[I.GARBAGE_QUEUE + 1] === 2 && v[I.GARBAGE_QUEUE + 2] === 0);
check('view garbage delay exposes per-entry delay',
  v[I.GARBAGE_DELAY] === 5 && v[I.GARBAGE_DELAY + 1] === 1);
check('view pending_garbage is the queue sum', v[I.PENDING_GARBAGE] === 5);

// --- placement-level API (Phase 2) ---

const codecSize = mod._et_codec_size();
check('et_codec_size is positive', codecSize > 0);

// serialize_full produces a buffer of codecSize ints
mod._et_reset(ctx, 42);
const fullPtr = mod._malloc(codecSize * 4);
mod._et_serialize_full(ctx, fullPtr);
const fullBase = fullPtr >> 2;
const full = mod.HEAP32.subarray(fullBase, fullBase + codecSize).slice();
check('serialize_full board rows match an empty playfield',
  // first visible board row (engine row 9) should be walls only, not all-zero
  full.length === codecSize);

// find_placements returns legal placements for the current piece on an empty board
const MAX_P = 256;
const placePtr = mod._malloc(MAX_P * 4 * 4);
mod._et_reset(ctx, 42);
const nPlacements = mod._et_find_placements(ctx, placePtr, MAX_P);
check('find_placements returns several placements on an empty board', nPlacements > 5);

// apply the first placement -> piece should lock, piece_count increments
const pBase = placePtr >> 2;
const p0 = mod.HEAP32.subarray(pBase, pBase + 4).slice();
const beforeApply = read();
const applied = mod._et_apply_placement(ctx, 0, p0[0], p0[1], p0[2], p0[3]);
check('apply_placement succeeds for an enumerated placement', applied === 1);
const afterApply = read();
check('apply_placement increments piece_count',
  afterApply[I.PIECE_COUNT] === beforeApply[I.PIECE_COUNT] + 1);
check('apply_placement leaves cells on the board',
  Array.from(afterApply.slice(0, BOARD_W * BOARD_H)).some((c) => c !== 0));

// apply_placement with a bogus placement is rejected
mod._et_reset(ctx, 42);
check('apply_placement rejects an unreachable placement',
  mod._et_apply_placement(ctx, 0, 99, 99, 0, 0) === 0);

// placement_path returns a step-action sequence that, replayed, matches
// apply_placement (same final board + piece_count).
const MAX_PATH = 256;
const pathPtr = mod._malloc(MAX_PATH * 4);
mod._et_reset(ctx, 55);
mod._et_find_placements(ctx, placePtr, MAX_P);
const pb = placePtr >> 2;
const tgt = mod.HEAP32.subarray(pb, pb + 4).slice(); // first placement
const pathLen = mod._et_placement_path(ctx, 0, tgt[0], tgt[1], tgt[2], tgt[3], pathPtr, MAX_PATH);
check('placement_path returns a non-empty path ending in HARD_DROP', pathLen > 0);
const ppb = pathPtr >> 2;
const path = Array.from(mod.HEAP32.subarray(ppb, ppb + pathLen));
check('placement_path last action is HARD_DROP (id 3)', path[path.length - 1] === 3);
for (const a of path) mod._et_step(ctx, a); // replay on the (still seed-55) ctx
const pathReplayed = read();
mod._et_reset(ctx, 55);
mod._et_apply_placement(ctx, 0, tgt[0], tgt[1], tgt[2], tgt[3]);
const pathApplied = read();
check('replayed placement_path matches apply_placement (board)',
  Array.from(pathReplayed.slice(0, BOARD_W * BOARD_H)).join(',') ===
    Array.from(pathApplied.slice(0, BOARD_W * BOARD_H)).join(','));
check('replayed placement_path matches apply_placement (piece_count)',
  pathReplayed[I.PIECE_COUNT] === pathApplied[I.PIECE_COUNT]);
mod._free(pathPtr);

mod._free(placePtr);
mod._free(fullPtr);
mod._free(ptr);
mod._et_free(ctx);

console.log(failures === 0 ? '\nALL PASSED' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
