// connect4_engine — High-performance Connect-X engine for Kaggle competition.
//
// Exposes a Python class `Engine` via PyO3 that performs bitboard-based
// alpha-beta search with iterative deepening, transposition table,
// principal variation search (PVS), killer moves, history heuristic,
// late move reductions (LMR), aspiration windows, and time management.

use pyo3::prelude::*;
use std::time::{Duration, Instant};

// ── Constants ────────────────────────────────────────────────────────────────
const MAX_COLS: usize = 7;
const MAX_PLY: usize = 50;
const TT_SIZE: usize = 1 << 22; // 4M entries (~64 MB)
const TT_MASK: usize = TT_SIZE - 1;
const WIN_SCORE: i32 = 1_000_000;
const INF: i32 = WIN_SCORE + 100;

const EXACT: u8 = 0;
const LOWERBOUND: u8 = 1;
const UPPERBOUND: u8 = 2;

// ── Transposition Table Entry ────────────────────────────────────────────────
#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,       // full hash for collision detection
    best_move: u8,
    depth: i8,
    score: i32,
    flag: u8,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry { key: 0, best_move: 0, depth: -1, score: 0, flag: EXACT }
    }
}

// ── Engine ───────────────────────────────────────────────────────────────────
#[pyclass]
struct Engine {
    // Precomputed data (set once in new())
    rows: usize,
    cols: usize,
    inarow: usize,
    col_height: usize,
    window_masks: Vec<u64>,
    center_mask: u64,
    col_order: Vec<usize>,

    // Transposition table (persists across moves)
    tt: Vec<TTEntry>,

    // Per-search mutable state
    my_bb: u64,
    opp_bb: u64,
    heights: [usize; MAX_COLS],
    nodes: u64,
    deadline: Instant,
    timed_out: bool,

    // Search heuristics (cleared per find_best_move call)
    killers: [[Option<usize>; 2]; MAX_PLY],
    history: [[i32; MAX_COLS]; 2],
}

// ── Internal engine methods (not exposed to Python) ──────────────────────────
impl Engine {
    /// Mix two bitboards into a single hash key.
    #[inline]
    fn hash_key(&self) -> u64 {
        let mut h = self.my_bb;
        h ^= self.opp_bb.wrapping_mul(0x9e3779b97f4a7c15);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    /// Fast win detection using bitboard shifts.
    #[inline]
    fn has_won(&self, board: u64) -> bool {
        let ch = self.col_height;
        // Vertical
        let m = board & (board >> 1);
        if m & (m >> 2) != 0 { return true; }
        // Horizontal
        let m = board & (board >> ch);
        if m & (m >> (2 * ch)) != 0 { return true; }
        // Diagonal /
        let m = board & (board >> (ch - 1));
        if m & (m >> (2 * (ch - 1))) != 0 { return true; }
        // Diagonal \
        let m = board & (board >> (ch + 1));
        if m & (m >> (2 * (ch + 1))) != 0 { return true; }
        false
    }

    /// Generalized win detection for inarow != 4.
    #[inline]
    fn has_won_general(&self, board: u64) -> bool {
        let ch = self.col_height;
        for &shift in &[1, ch, ch - 1, ch + 1] {
            let mut m = board;
            for i in 1..self.inarow {
                m &= board >> (i * shift);
            }
            if m != 0 { return true; }
        }
        false
    }

    /// Dispatch to optimized or general win check.
    #[inline]
    fn check_win(&self, board: u64) -> bool {
        if self.inarow == 4 {
            self.has_won(board)
        } else {
            self.has_won_general(board)
        }
    }

    /// Heuristic evaluation of current position (from "my" perspective).
    fn evaluate(&self) -> i32 {
        let mut score: i32 = 0;
        let inarow = self.inarow as u32;

        // Center column bonus
        let my_center = (self.my_bb & self.center_mask).count_ones() as i32;
        let opp_center = (self.opp_bb & self.center_mask).count_ones() as i32;
        score += (my_center - opp_center) * 12;

        // Window evaluation
        for &mask in &self.window_masks {
            let mc = (self.my_bb & mask).count_ones();
            let oc = (self.opp_bb & mask).count_ones();
            if mc > 0 && oc > 0 { continue; } // blocked window
            let empty = inarow - mc - oc;
            if mc == inarow - 1 && empty == 1 {
                score += 50;
            } else if mc == inarow - 2 && empty == 2 {
                score += 10;
            } else if oc == inarow - 1 && empty == 1 {
                score -= 45;
            } else if oc == inarow - 2 && empty == 2 {
                score -= 8;
            }
        }
        score
    }

    /// Get list of valid (non-full) columns.
    #[inline]
    fn valid_cols_vec(&self) -> Vec<usize> {
        (0..self.cols).filter(|&c| self.heights[c] < self.rows).collect()
    }

    /// Drop a piece into a column.
    #[inline]
    fn drop(&mut self, col: usize, mine: bool) {
        let pos = col * self.col_height + self.heights[col];
        if mine {
            self.my_bb |= 1u64 << pos;
        } else {
            self.opp_bb |= 1u64 << pos;
        }
        self.heights[col] += 1;
    }

    /// Undo a piece drop.
    #[inline]
    fn undo(&mut self, col: usize, mine: bool) {
        self.heights[col] -= 1;
        let pos = col * self.col_height + self.heights[col];
        if mine {
            self.my_bb &= !(1u64 << pos);
        } else {
            self.opp_bb &= !(1u64 << pos);
        }
    }

    /// Order moves: TT best → killers → remaining sorted by history (center-bias tiebreak).
    fn order_moves(&self, valid: &[usize], tt_move: Option<usize>, ply: usize, side: usize) -> Vec<usize> {
        let mut ordered = Vec::with_capacity(self.cols);
        let mut used = [false; MAX_COLS];

        // 1. TT best move
        if let Some(tm) = tt_move {
            if valid.contains(&tm) {
                ordered.push(tm);
                used[tm] = true;
            }
        }

        // 2. Killer moves
        if ply < MAX_PLY {
            for i in 0..2 {
                if let Some(km) = self.killers[ply][i] {
                    if km < MAX_COLS && valid.contains(&km) && !used[km] {
                        ordered.push(km);
                        used[km] = true;
                    }
                }
            }
        }

        // 3. Remaining: center-biased order, stable-sorted by history descending
        let mut rest: Vec<usize> = self.col_order.iter()
            .filter(|&&c| valid.contains(&c) && !used[c])
            .cloned()
            .collect();
        rest.sort_by(|&a, &b| self.history[side][b].cmp(&self.history[side][a]));
        ordered.extend(rest);

        ordered
    }

    /// Store a TT entry (depth-preferred replacement).
    #[inline]
    fn tt_store(&mut self, key: u64, depth: i32, best_move: usize, score: i32, flag: u8) {
        let idx = (key as usize) & TT_MASK;
        // Replace if new depth is >= stored depth (or entry is empty)
        if depth as i8 >= self.tt[idx].depth || self.tt[idx].key == 0 {
            self.tt[idx] = TTEntry {
                key,
                depth: depth as i8,
                best_move: best_move as u8,
                score,
                flag,
            };
        }
    }

    /// Alpha-beta minimax with PVS, TT, killer moves, history heuristic, and LMR.
    fn minimax(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool, ply: usize) -> (Option<usize>, i32) {
        if self.timed_out {
            return (None, 0);
        }

        self.nodes += 1;
        if self.nodes & 2047 == 0 && Instant::now() >= self.deadline {
            self.timed_out = true;
            return (None, 0);
        }

        let orig_alpha = alpha;
        let orig_beta = beta;

        // ── TT Probe ──
        let key = self.hash_key();
        let tt_idx = (key as usize) & TT_MASK;
        let tt_entry = self.tt[tt_idx];
        let mut tt_move: Option<usize> = None;

        if tt_entry.key == key {
            tt_move = Some(tt_entry.best_move as usize);
            if (tt_entry.depth as i32) >= depth {
                match tt_entry.flag {
                    EXACT => return (tt_move, tt_entry.score),
                    LOWERBOUND => alpha = alpha.max(tt_entry.score),
                    UPPERBOUND => beta = beta.min(tt_entry.score),
                    _ => {}
                }
                if alpha >= beta {
                    return (tt_move, tt_entry.score);
                }
            }
        }

        let valid = self.valid_cols_vec();

        // ── Terminal Checks ──
        if self.check_win(self.my_bb) {
            return (None, WIN_SCORE + depth);
        }
        if self.check_win(self.opp_bb) {
            return (None, -(WIN_SCORE + depth));
        }
        if valid.is_empty() {
            return (None, 0);
        }
        if depth == 0 {
            return (None, self.evaluate());
        }

        // ── Immediate One-Move Win Check ──
        let mine = maximizing;
        for &col in &valid {
            self.drop(col, mine);
            let won = if mine { self.check_win(self.my_bb) } else { self.check_win(self.opp_bb) };
            self.undo(col, mine);
            if won {
                let sc = if maximizing { WIN_SCORE + depth - 1 } else { -(WIN_SCORE + depth - 1) };
                self.tt_store(key, depth, col, sc, EXACT);
                return (Some(col), sc);
            }
        }

        let side = if maximizing { 0 } else { 1 };
        let ordered = self.order_moves(&valid, tt_move, ply, side);

        // ── PVS + LMR Search ──
        if maximizing {
            let mut value = -INF;
            let mut best = ordered[0];

            for (move_idx, &col) in ordered.iter().enumerate() {
                self.drop(col, true);

                let is_important = tt_move == Some(col)
                    || (ply < MAX_PLY && (self.killers[ply][0] == Some(col)
                                       || self.killers[ply][1] == Some(col)));
                let use_lmr = depth >= 3 && move_idx >= 2 && !is_important;

                let ns = if move_idx == 0 {
                    // First move: full-window search
                    self.minimax(depth - 1, alpha, beta, false, ply + 1).1
                } else {
                    // LMR + PVS: reduced-depth null-window scout
                    let rd = if use_lmr { depth - 2 } else { depth - 1 };
                    let mut s = self.minimax(rd, alpha, alpha + 1, false, ply + 1).1;
                    // Re-search at full depth if LMR scout failed high
                    if use_lmr && s > alpha && !self.timed_out {
                        s = self.minimax(depth - 1, alpha, alpha + 1, false, ply + 1).1;
                    }
                    // PVS re-search with full window if scout is in-window
                    if s > alpha && s < beta && !self.timed_out {
                        s = self.minimax(depth - 1, alpha, beta, false, ply + 1).1;
                    }
                    s
                };

                self.undo(col, true);
                if self.timed_out {
                    return (Some(best), if value > -INF { value } else { 0 });
                }
                if ns > value {
                    value = ns;
                    best = col;
                }
                alpha = alpha.max(value);
                if alpha >= beta {
                    // Killer move heuristic: store cutoff move
                    if ply < MAX_PLY && self.killers[ply][0] != Some(col) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(col);
                    }
                    // History heuristic: reward cutoff move
                    self.history[side][col] += depth * depth;
                    break;
                }
            }

            let flag = if value <= orig_alpha { UPPERBOUND }
                       else if value >= orig_beta { LOWERBOUND }
                       else { EXACT };
            self.tt_store(key, depth, best, value, flag);
            (Some(best), value)
        } else {
            let mut value = INF;
            let mut best = ordered[0];

            for (move_idx, &col) in ordered.iter().enumerate() {
                self.drop(col, false);

                let is_important = tt_move == Some(col)
                    || (ply < MAX_PLY && (self.killers[ply][0] == Some(col)
                                       || self.killers[ply][1] == Some(col)));
                let use_lmr = depth >= 3 && move_idx >= 2 && !is_important;

                let ns = if move_idx == 0 {
                    self.minimax(depth - 1, alpha, beta, true, ply + 1).1
                } else {
                    let rd = if use_lmr { depth - 2 } else { depth - 1 };
                    let mut s = self.minimax(rd, beta - 1, beta, true, ply + 1).1;
                    if use_lmr && s < beta && !self.timed_out {
                        s = self.minimax(depth - 1, beta - 1, beta, true, ply + 1).1;
                    }
                    if s < beta && s > alpha && !self.timed_out {
                        s = self.minimax(depth - 1, alpha, beta, true, ply + 1).1;
                    }
                    s
                };

                self.undo(col, false);
                if self.timed_out {
                    return (Some(best), if value < INF { value } else { 0 });
                }
                if ns < value {
                    value = ns;
                    best = col;
                }
                beta = beta.min(value);
                if alpha >= beta {
                    if ply < MAX_PLY && self.killers[ply][0] != Some(col) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(col);
                    }
                    self.history[side][col] += depth * depth;
                    break;
                }
            }

            let flag = if value <= orig_alpha { UPPERBOUND }
                       else if value >= orig_beta { LOWERBOUND }
                       else { EXACT };
            self.tt_store(key, depth, best, value, flag);
            (Some(best), value)
        }
    }
}

// ── Python-exposed methods ───────────────────────────────────────────────────
#[pymethods]
impl Engine {
    #[new]
    fn new(rows: usize, cols: usize, inarow: usize) -> Self {
        let col_height = rows + 1;
        let center = cols / 2;

        // ── Precompute window masks ──
        let mut masks = Vec::new();
        // Horizontal
        for row in 0..rows {
            for c in 0..=(cols - inarow) {
                let mut mask = 0u64;
                for i in 0..inarow {
                    mask |= 1u64 << ((c + i) * col_height + row);
                }
                masks.push(mask);
            }
        }
        // Vertical
        for c in 0..cols {
            for row in 0..=(rows - inarow) {
                let mut mask = 0u64;
                for i in 0..inarow {
                    mask |= 1u64 << (c * col_height + row + i);
                }
                masks.push(mask);
            }
        }
        // Diagonal /
        for row in 0..=(rows - inarow) {
            for c in 0..=(cols - inarow) {
                let mut mask = 0u64;
                for i in 0..inarow {
                    mask |= 1u64 << ((c + i) * col_height + row + i);
                }
                masks.push(mask);
            }
        }
        // Diagonal \
        for row in (inarow - 1)..rows {
            for c in 0..=(cols - inarow) {
                let mut mask = 0u64;
                for i in 0..inarow {
                    mask |= 1u64 << ((c + i) * col_height + row - i);
                }
                masks.push(mask);
            }
        }

        // Center column mask
        let mut cmask = 0u64;
        for row in 0..rows {
            cmask |= 1u64 << (center * col_height + row);
        }

        // Column order: center-first
        let mut col_order: Vec<usize> = (0..cols).collect();
        col_order.sort_by_key(|&c| if c >= center { c - center } else { center - c });

        Engine {
            rows, cols, inarow, col_height,
            window_masks: masks,
            center_mask: cmask,
            col_order,
            tt: vec![TTEntry::default(); TT_SIZE],
            my_bb: 0,
            opp_bb: 0,
            heights: [0; MAX_COLS],
            nodes: 0,
            deadline: Instant::now(),
            timed_out: false,
            killers: [[None; 2]; MAX_PLY],
            history: [[0; MAX_COLS]; 2],
        }
    }

    /// Find the best move given a flat board, player mark, and time budget (ms).
    fn find_best_move(&mut self, board: Vec<u8>, mark: u8, time_budget_ms: u64) -> u8 {
        // ── Convert flat board → bitboards ──
        self.my_bb = 0;
        self.opp_bb = 0;
        self.heights = [0; MAX_COLS];

        for col in 0..self.cols {
            for kaggle_row in 0..self.rows {
                let cell = board[col + kaggle_row * self.cols];
                if cell != 0 {
                    let bb_row = self.rows - 1 - kaggle_row;
                    let pos = col * self.col_height + bb_row;
                    if cell == mark {
                        self.my_bb |= 1u64 << pos;
                    } else {
                        self.opp_bb |= 1u64 << pos;
                    }
                }
            }
        }

        // Compute heights
        for col in 0..self.cols {
            let mut h = 0;
            for row in 0..self.rows {
                if (self.my_bb | self.opp_bb) & (1u64 << (col * self.col_height + row)) != 0 {
                    h = row + 1;
                } else {
                    break;
                }
            }
            self.heights[col] = h;
        }

        let valid = self.valid_cols_vec();
        if valid.is_empty() { return 0; }
        if valid.len() == 1 { return valid[0] as u8; }

        // ── Instant win check ──
        for &col in &valid {
            self.drop(col, true);
            if self.check_win(self.my_bb) {
                self.undo(col, true);
                return col as u8;
            }
            self.undo(col, true);
        }

        // ── Forced block check ──
        let mut threats = Vec::new();
        for &col in &valid {
            self.drop(col, false);
            if self.check_win(self.opp_bb) {
                threats.push(col);
            }
            self.undo(col, false);
        }
        if threats.len() == 1 {
            return threats[0] as u8;
        }

        // ── Clear per-search heuristics ──
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[0; MAX_COLS]; 2];

        // ── Iterative Deepening with Aspiration Windows ──
        self.deadline = Instant::now() + Duration::from_millis(time_budget_ms);
        let mut best_col = *self.col_order.iter().find(|&&c| valid.contains(&c)).unwrap_or(&valid[0]);
        let mut best_depth = 0u32;
        let mut prev_score = 0i32;

        for d in 1..=42i32 {
            self.nodes = 0;

            // Aspiration window: narrow band around previous score for d > 1
            let (mut a, mut b) = if d > 1 {
                (prev_score - 50, prev_score + 50)
            } else {
                (-INF, INF)
            };

            let mut search_complete = false;
            let mut result = (None, 0i32);

            loop {
                self.timed_out = false;
                result = self.minimax(d, a, b, true, 0);
                if self.timed_out { break; }

                if result.1 <= a && a > -INF {
                    a = -INF; // fail-low: widen alpha
                } else if result.1 >= b && b < INF {
                    b = INF;  // fail-high: widen beta
                } else {
                    search_complete = true;
                    break;
                }
            }

            if search_complete {
                if let Some(c) = result.0 {
                    best_col = c;
                    best_depth = d as u32;
                }
                prev_score = result.1;
            }

            if search_complete && (result.1 >= WIN_SCORE || result.1 <= -WIN_SCORE) {
                break;
            }

            // Don't start next depth if close to deadline
            if Instant::now() >= self.deadline - Duration::from_millis(50) {
                break;
            }
        }

        let _ = best_depth; // used for debugging if needed
        best_col as u8
    }

    /// Find the best move given bitboards directly (no flat-board conversion).
    ///
    /// Useful when the caller already stores bitboards with the same encoding
    /// (column-major, LSB = bottom row, col_height = rows + 1).
    fn find_best_move_bb(&mut self, my_bb: u64, opp_bb: u64, heights: Vec<usize>, time_budget_ms: u64) -> u8 {
        self.my_bb = my_bb;
        self.opp_bb = opp_bb;
        self.heights = [0; MAX_COLS];
        for (i, &h) in heights.iter().enumerate().take(self.cols) {
            self.heights[i] = h;
        }

        let valid = self.valid_cols_vec();
        if valid.is_empty() { return 0; }
        if valid.len() == 1 { return valid[0] as u8; }

        // Instant win check
        for &col in &valid {
            self.drop(col, true);
            if self.check_win(self.my_bb) {
                self.undo(col, true);
                return col as u8;
            }
            self.undo(col, true);
        }

        // Forced block check
        let mut threats = Vec::new();
        for &col in &valid {
            self.drop(col, false);
            if self.check_win(self.opp_bb) {
                threats.push(col);
            }
            self.undo(col, false);
        }
        if threats.len() == 1 {
            return threats[0] as u8;
        }

        // Clear per-search heuristics
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[0; MAX_COLS]; 2];

        // Iterative deepening with aspiration windows
        self.deadline = Instant::now() + Duration::from_millis(time_budget_ms);
        let mut best_col = *self.col_order.iter().find(|&&c| valid.contains(&c)).unwrap_or(&valid[0]);
        let mut prev_score = 0i32;

        for d in 1..=42i32 {
            self.nodes = 0;

            let (mut a, mut b) = if d > 1 {
                (prev_score - 50, prev_score + 50)
            } else {
                (-INF, INF)
            };

            let mut search_complete = false;
            let mut result = (None, 0i32);

            loop {
                self.timed_out = false;
                result = self.minimax(d, a, b, true, 0);
                if self.timed_out { break; }

                if result.1 <= a && a > -INF {
                    a = -INF;
                } else if result.1 >= b && b < INF {
                    b = INF;
                } else {
                    search_complete = true;
                    break;
                }
            }

            if search_complete {
                if let Some(c) = result.0 {
                    best_col = c;
                }
                prev_score = result.1;
            }

            if search_complete && (result.1 >= WIN_SCORE || result.1 <= -WIN_SCORE) {
                break;
            }

            if Instant::now() >= self.deadline - Duration::from_millis(50) {
                break;
            }
        }

        best_col as u8
    }

    /// Clear the transposition table (call between games).
    fn clear_tt(&mut self) {
        for entry in self.tt.iter_mut() {
            *entry = TTEntry::default();
        }
    }

    /// Return the number of nodes searched in the last find_best_move call.
    fn last_nodes(&self) -> u64 {
        self.nodes
    }
}

// ── Python Module ────────────────────────────────────────────────────────────
#[pymodule]
fn connect4_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
