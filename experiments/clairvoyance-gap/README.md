# Clairvoyance gap (2026-09-21)

## 問題

`moderntetris_placement` 的 MCTS 直接 clone 真實 env 往下展開，engine 的方塊 seed 與
`garbage_seed` 都在 state 裡，所以每個 simulation 看到的是**這一局真正會發生的**未來
（preview 之後的方塊、垃圾行的有無/行數/洞位置）。想量化：訓練與 eval 分數裡有多少是
來自「偷看真實未來」。

## 設計

新增 `actor_mcts_resample_hidden_future`（預設 false = 原本行為）。開啟時每一步搜尋前
複製 root env 並呼叫 `ModernTetrisPlacementEnv::resampleHiddenFuture()`；
`actor_mcts_resample_hidden_future_parts` 選要重抽哪些（`pieces`、`garbage`、
`garbage-iid`、`none`）。

- **pieces**：preview（5 個）之後的方塊在各自 7-bag 內重新洗牌（= 可觀察資訊下的
  posterior 抽樣），並重抽方塊 seed。
- **garbage**：在 root 抽好新的 `garbage_seed`，但等 root 這一步 hard drop **之後**才換上。
  原因：afterstate 特徵是用 engine hard drop 算的，這一步進場的垃圾洞位已經出現在
  網路輸入裡；在 root 就換 seed 會讓 root 的輸入和真實 env 不同（見下方 rootleak）。
  所有 root 動作共用同一個新 seed（CRN）。
- **garbage-iid**：同上，但每個 root 動作 hard drop 後各自抽新 seed（獨立抽樣）。
- **none**：只走重抽流程、不改任何東西（檢查流程本身）。

實際對局一律走真實 env。每一步搜尋前 `ZeroActor::checkSearchRootObservation()`
比對重抽後的 root 與真實 env 的整份網路輸入（盤面、current/hold/preview、combo、
b2b、pending garbage、所有合法動作的 descriptor 與 afterstate），不同就直接中止。
曾以「連 preview 一起洗」的故意 bug 驗證過這個檢查會觸發。

其餘設定沿用模型 cfg（gumbel on、384 sim、sample 64、max 200 步、garbage p=0.1）。

- 模型：`moderntetris_placement_gpz_3bx256_n50-098904-dirty-rs-lst-nopc-0903` iter 1687040
- `run.sh <dir> <iter> <label> <gpu> <num_games> [extra_conf]` 跑、`summarize.py` 彙整
- `tests/run.sh`：env 層級的檢查（重抽保留可見資訊與 bag 組成、afterstate 選項）

## 結果

| 組別 | 搜尋看到的未來 | 局數 | 平均分數 | 死亡率 | 平均長度 |
|---|---|---|---|---|---|
| true-future | 真實未來（訓練時的設定） | 256 | 100.73 ± 2.79 | 96.5% | 110.9 |
| resample-none | 同上，走重抽流程但不改 | 128 | 99.16 ± 4.27 | 93.0% | 109.3 |
| resample-pieces | 方塊重抽 | 128 | 56.29 ± 2.45 | 99.2% | 98.3 |
| resample-garbage | 垃圾重抽（CRN） | 128 | 47.88 ± 2.62 | 100% | 87.4 |
| resample-garbage-iid | 垃圾重抽（各動作獨立） | 128 | 39.79 ± 3.43 | 99.2% | 81.9 |
| **resample-all** | **方塊 + 垃圾重抽（CRN）** | 256 | **26.05 ± 1.54** | 99.6% | 78.9 |
| resample-all-rootleak | 同上，但 root 就換 garbage seed（root 洞位錯） | 256 | 23.14 ± 1.60 | 100% | 77.2 |
| resample-all-garbageiid | 方塊 + 垃圾重抽（垃圾各動作獨立） | 256 | 10.06 ± 1.48 | 100% | 65.2 |
| policy-only | 不搜尋（sim=1、無 gumbel noise） | 128 | −5.13 ± 1.10 | 100% | 38.4 |

Sanity check：true-future 與訓練最後一個 iteration 的 self-play 統計一致
（avg return 97.3、avg length 108.2）；resample-none 與 true-future 無顯著差異。

## 結論

1. **Clairvoyance gap = 74.7 ± 3.2（true-future − resample-all），佔 74%。**
   目前模型的分數大部分來自搜尋偷看真實未來。方塊（−44）與垃圾（−53）各自都很大，
   合起來不是相加（兩者重疊）。
2. **root 洞位洩漏影響很小**：resample-all vs rootleak = 2.9 ± 2.2，不顯著。
   最早那次 23.1 的結果基本上是對的。
3. **CRN 很重要**：垃圾未來在 root 動作間共用 vs 各自獨立，resample-all 26.1 vs 10.1、
   garbage-only 47.9 vs 39.8。獨立抽樣時 gumbel sequential halving 會挑到「抽到好運
   未來」的動作。與研究筆記 §4.2 第 4 點一致。
4. **看錯未來的搜尋仍遠勝不搜尋**（26.1 vs −5.1）；此模型的 policy 本身很弱，強度幾乎
   都來自搜尋。

## Caveat

1. resample-all 仍是**下界**：搜尋在一個抽樣出來的未來上做完美資訊搜尋（單一
   determinization，PIMC n=1），仍有 strategy fusion 與樂觀偏差。
2. 模型是在 clairvoyant 搜尋下訓練的，policy / value target 都來自偷看的搜尋；
   重抽組同時承受「搜尋不能偷看」與「網路學到的東西預設了會偷看」，這次無法分開。
   → 需要用 resample 訓練一個 model 才能拆開。
3. **環境/特徵本身的洩漏**：afterstate 特徵是 hard drop（含這一步進場的垃圾）之後的
   盤面，會透露垃圾洞位；真實玩家落子前不知道。只有一步、影響看起來小（結論 2），
   已新增 `nn_placement_afterstate_before_garbage`（預設 false，舊模型不受影響）：
   落點 BFS 改在清空垃圾佇列的副本上跑，afterstate 就是垃圾進場前的盤面。
   測試：2736 個有待進場垃圾的狀態中，關閉時 178 個（6.5%）的 descriptor 隨
   `garbage_seed` 改變，開啟時 0 個；合法動作集合不變。新訓練應開啟。
4. 各組局面沒有配對（重抽會消耗同一條 RNG stream），靠局數壓變異數。
