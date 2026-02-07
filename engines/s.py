#!/usr/bin/env python3

import sys
import math
import time
import random
from collections import defaultdict

"""
Simple UCI-compatible chess engine in a single file.
Features:
- Bitboard-based board representation
- Legal move generation
- Negamax alpha-beta with iterative deepening
- Quiescence search (captures only)
- Transposition table (Zobrist hashing)
- MVV-LVA move ordering + history heuristic
- Null-move pruning (with check/depth/zugzwang/consecutive safeguards)
- Late Move Reductions (LMR) for quiet late moves with re-search on improvement
- Evaluation: material, PSTs, bishop pair, passed pawns,
  king safety and separate king endgame PST
- Anti-threefold repetition in clearly winning positions
- UCI protocol with info (depth, nodes, nps, score cp/mate)
- Blunder avoidance and king-safety heuristics (see below)
- EG PST for all pieces; passed/doubled/weak pawn bonuses; rook open/semi-open file; piece-in-center bonuses
- King safety: pawn shield, attack weight; TT replacement prefers EXACT/same depth
- Move ordering: killer moves, counter-move heuristic; extensions: check, pawn to 7th; dynamic LMR
- Quiescence: captures + promotions + quiet checks; depth limit (QSEARCH_MAX_PLY) to avoid check cycles
- PVS (null-window for non-PV), soft futility at depth 1 only, strict LMR (first 2 moves full depth)
- Eval: rook on 7th, knight outpost; passed-pawn rank bonus scales up in endgame; all new bonuses capped

This is designed to be reasonably strong but concise, not a top-tier engine.

Blunder avoidance and king safety (heuristic-based, conservative):
---------------------------------------------------------------
EVALUATION:
- Hanging pieces: Penalise positions where a piece is attacked and undefended.
  Only clear hangs are penalised; defended pieces (trades, sacrifices with
  compensation) are not. Applied to both sides; scale ~piece_value/2.
- King safety (MG): Penalise king on open file, or in centre (files 2–5,
  ranks 2–5), when phase > 8. Encourages castling and avoiding early king
  walks or centralisation without endgame justification.
- Winning positions: When |eval| > ~220 cp and phase > 6, extra penalty for
  king exposure (open file, centre). Reduces speculative king exposure or
  counterplay (e.g. perpetual checks) when clearly ahead.

MOVE ORDERING (root only, when not in check):
- En prise: Moves that leave our pieces attacked and undefended get a
  penalty added to the ordering score so we try "safe" moves first. We only
  deprioritise; we never filter. Defensive alternatives are tried earlier.
- SEE-lite (bad captures): Capture moves only. After simulating the capture,
  if the opponent can recapture on the destination with a cheaper piece, apply
  a small ordering penalty (tens of cp). Not applied when in check, when the
  capture gives check, or to castling. Cheap heuristic (min attacker only);
  no full SEE, no pruning. Demotes fake-good captures (e.g. Nxe4 … Bxe4) while
  search still finds sound sacrifices.
- King moves: Quiet king moves (excluding castling) in middlegame (phase > 8)
  are deprioritised to avoid early king walks or unnecessary centralisation.

Why correct sacrifices are not suppressed:
- We never filter or forbid moves; we only penalise in eval and reorder.
- Hang penalty applies only to attacked-and-undefended pieces. Defended
  pieces (winning trades, sacrifices with recapture) are not penalised.
- Sacrifices that lead to mate, decisive attack, or equal/better material
  are found by search; the modest heuristic penalties are outweighed by
  the score. Quiescence and main search still expand all moves.
"""

WHITE, BLACK = 0, 1

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)

# --- Feature flags: set False to disable (safe fallback, no regression) ---
USE_PVS = True           # Principal Variation Search: null-window for non-PV moves, re-search on fail-high
USE_FUTILITY = True      # Soft futility at depth 1 only; only when not in check and eval >= -150; margin 80 cp
USE_LMR_STRICT = True    # LMR: only quiet, depth>=3, skip first 2 moves, no check/promo, max reduction 2
QSEARCH_MAX_PLY = 8      # Max quiescence depth (promo/checks) to avoid check cycles; 0 = no limit on checks
COUNTER_BONUS = 400_000  # Counter-move ordering bonus (weak; must stay below TT/captures)
DISABLE_FORCING_FILTER = False  # If True: skip anti-blunder ordering (opponent has simple check/capture reply)
USE_STRONG_PIECE_PROTECTION = True  # If True: avoid quiet moves that leave R/B/Q hanging (root ordering + quiescence)

PIECE_SYMBOLS = {
    (WHITE, PAWN): "P",
    (WHITE, KNIGHT): "N",
    (WHITE, BISHOP): "B",
    (WHITE, ROOK): "R",
    (WHITE, QUEEN): "Q",
    (WHITE, KING): "K",
    (BLACK, PAWN): "p",
    (BLACK, KNIGHT): "n",
    (BLACK, BISHOP): "b",
    (BLACK, ROOK): "r",
    (BLACK, QUEEN): "q",
    (BLACK, KING): "k",
}

SYMBOL_TO_PIECE = {v: k for k, v in PIECE_SYMBOLS.items()}

FILES = "abcdefgh"
RANKS = "12345678"


def sq_index(file, rank):
    return rank * 8 + file


def fr_to_sq(f, r):
    return sq_index(ord(f) - 97, int(r) - 1)


def sq_to_coord(sq):
    f = sq & 7
    r = sq >> 3
    return FILES[f] + RANKS[r]


def bit(sq):
    return 1 << sq


def lsb(bb):
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


def pop_lsb(bb):
    l = (bb & -bb).bit_length() - 1
    return l, bb & (bb - 1)


def popcount(x):
    return x.bit_count()


class Move:
    __slots__ = ("from_sq", "to_sq", "piece", "capture", "promo", "flags")

    QUIET = 0
    CAPTURE = 1
    EN_PASSANT = 2
    CASTLE = 4

    def __init__(self, from_sq, to_sq, piece, capture=None, promo=None, flags=0):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.piece = piece
        self.capture = capture
        self.promo = promo
        self.flags = flags

    def is_capture(self):
        return self.capture is not None or (self.flags & Move.EN_PASSANT)

    def is_quiet(self):
        return not self.is_capture() and not (self.flags & Move.CASTLE)

    def uci(self):
        s = sq_to_coord(self.from_sq) + sq_to_coord(self.to_sq)
        if self.promo is not None:
            promo_map = {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}
            s += promo_map.get(self.promo, "q")
        return s


class Board:
    def __init__(self):
        self.bb = [[0] * 6 for _ in range(2)]
        self.occ = [0, 0]
        self.all_occ = 0
        self.side_to_move = WHITE
        self.castling = 0  # bit 0: white K, 1: white Q, 2: black K, 3: black Q
        self.ep_square = -1
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.history = []  # for undo
        self.hash_history = []  # for repetition detection
        self.hash_count = {}  # O(1) repetition: count of each key in history
        self.zobrist_key = 0
        # Incremental evaluation (piece sum only; eval_board adds bishop/passed/hang/king)
        self.eval_mg = 0
        self.eval_eg = 0
        self.phase = 0

    def clone(self):
        b = Board()
        b.bb = [self.bb[0][:], self.bb[1][:]]
        b.occ = self.occ[:]
        b.all_occ = self.all_occ
        b.side_to_move = self.side_to_move
        b.castling = self.castling
        b.ep_square = self.ep_square
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.history = list(self.history)
        b.hash_history = list(self.hash_history)
        b.hash_count = dict(self.hash_count)
        b.zobrist_key = self.zobrist_key
        b.eval_mg = self.eval_mg
        b.eval_eg = self.eval_eg
        b.phase = self.phase
        return b


ZOBRIST_PIECE = [[[0] * 64 for _ in range(6)] for _ in range(2)]
ZOBRIST_CASTLING = [0] * 16
ZOBRIST_EP = [0] * 64
ZOBRIST_SIDE = 0
# Precomputed Zobrist XOR for entire castle move (king + rook pieces only; castling rights done separately)
ZOBRIST_CASTLE_PIECE = [[0, 0], [0, 0]]  # [color][0=kingside, 1=queenside]


def init_zobrist():
    global ZOBRIST_SIDE, ZOBRIST_CASTLE_PIECE
    rnd = random.Random(20250101)
    for c in (WHITE, BLACK):
        for p in range(6):
            for sq in range(64):
                ZOBRIST_PIECE[c][p][sq] = rnd.getrandbits(64)
    for i in range(16):
        ZOBRIST_CASTLING[i] = rnd.getrandbits(64)
    for sq in range(64):
        ZOBRIST_EP[sq] = rnd.getrandbits(64)
    ZOBRIST_SIDE = rnd.getrandbits(64)
    # Precompute one XOR value per castle type (piece moves only)
    e1, g1, h1, f1 = fr_to_sq("e", "1"), fr_to_sq("g", "1"), fr_to_sq("h", "1"), fr_to_sq("f", "1")
    e8, g8, h8, f8 = fr_to_sq("e", "8"), fr_to_sq("g", "8"), fr_to_sq("h", "8"), fr_to_sq("f", "8")
    c1, a1, d1 = fr_to_sq("c", "1"), fr_to_sq("a", "1"), fr_to_sq("d", "1")
    c8, a8, d8 = fr_to_sq("c", "8"), fr_to_sq("a", "8"), fr_to_sq("d", "8")
    ZOBRIST_CASTLE_PIECE[WHITE][0] = (ZOBRIST_PIECE[WHITE][KING][e1] ^ ZOBRIST_PIECE[WHITE][KING][g1] ^
                                      ZOBRIST_PIECE[WHITE][ROOK][h1] ^ ZOBRIST_PIECE[WHITE][ROOK][f1])
    ZOBRIST_CASTLE_PIECE[WHITE][1] = (ZOBRIST_PIECE[WHITE][KING][e1] ^ ZOBRIST_PIECE[WHITE][KING][c1] ^
                                      ZOBRIST_PIECE[WHITE][ROOK][a1] ^ ZOBRIST_PIECE[WHITE][ROOK][d1])
    ZOBRIST_CASTLE_PIECE[BLACK][0] = (ZOBRIST_PIECE[BLACK][KING][e8] ^ ZOBRIST_PIECE[BLACK][KING][g8] ^
                                      ZOBRIST_PIECE[BLACK][ROOK][h8] ^ ZOBRIST_PIECE[BLACK][ROOK][f8])
    ZOBRIST_CASTLE_PIECE[BLACK][1] = (ZOBRIST_PIECE[BLACK][KING][e8] ^ ZOBRIST_PIECE[BLACK][KING][c8] ^
                                      ZOBRIST_PIECE[BLACK][ROOK][a8] ^ ZOBRIST_PIECE[BLACK][ROOK][d8])


def compute_hash(board: Board):
    h = 0
    for c in (WHITE, BLACK):
        for p in range(6):
            bb = board.bb[c][p]
            while bb:
                sq, bb = pop_lsb(bb)
                h ^= ZOBRIST_PIECE[c][p][sq]
    h ^= ZOBRIST_CASTLING[board.castling]
    if board.ep_square != -1:
        h ^= ZOBRIST_EP[board.ep_square]
    if board.side_to_move == BLACK:
        h ^= ZOBRIST_SIDE
    return h


def _recompute_incremental_eval(board: Board):
    """Full recompute of eval_mg, eval_eg, phase from current board (e.g. after set_fen)."""
    mg = 0
    eg = 0
    phase = 0
    for c in (WHITE, BLACK):
        for p in range(6):
            bb = board.bb[c][p]
            while bb:
                sq, bb = pop_lsb(bb)
                dm, de = _piece_eval_contribution(c, p, sq)
                mg += dm
                eg += de
                phase += _phase_delta(p)
    board.eval_mg = mg
    board.eval_eg = eg
    board.phase = max(0, min(24, phase))


def set_fen(board: Board, fen: str):
    parts = fen.strip().split()
    assert len(parts) >= 4
    for c in (WHITE, BLACK):
        for p in range(6):
            board.bb[c][p] = 0
    board.occ = [0, 0]
    board.all_occ = 0
    rows = parts[0].split("/")
    assert len(rows) == 8
    for rank in range(7, -1, -1):
        file = 0
        for ch in rows[7 - rank]:
            if ch.isdigit():
                file += int(ch)
            else:
                color, piece = SYMBOL_TO_PIECE[ch]
                sq = sq_index(file, rank)
                board.bb[color][piece] |= bit(sq)
                file += 1
    board.occ[WHITE] = 0
    board.occ[BLACK] = 0
    for p in range(6):
        board.occ[WHITE] |= board.bb[WHITE][p]
        board.occ[BLACK] |= board.bb[BLACK][p]
    board.all_occ = board.occ[WHITE] | board.occ[BLACK]
    board.side_to_move = WHITE if parts[1] == "w" else BLACK
    castling = 0
    if "K" in parts[2]:
        castling |= 1
    if "Q" in parts[2]:
        castling |= 2
    if "k" in parts[2]:
        castling |= 4
    if "q" in parts[2]:
        castling |= 8
    board.castling = castling
    board.ep_square = -1
    if parts[3] != "-":
        board.ep_square = fr_to_sq(parts[3][0], parts[3][1])
    board.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
    board.fullmove_number = int(parts[5]) if len(parts) > 5 else 1
    board.zobrist_key = compute_hash(board)
    board.history.clear()
    board.hash_history = [board.zobrist_key]
    board.hash_count = {board.zobrist_key: 1}
    _recompute_incremental_eval(board)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def in_bounds(sq):
    return 0 <= sq < 64


KNIGHT_DELTAS = [17, 15, 10, 6, -17, -15, -10, -6]
BISHOP_DELTAS = [9, 7, -9, -7]
ROOK_DELTAS = [8, -8, 1, -1]
KING_DELTAS = [8, -8, 1, -1, 9, 7, -9, -7]


def gen_slider_moves(board: Board, color, sq, deltas, moves, captures_only=False):
    """Generate rook/bishop/queen moves with proper wrap prevention."""
    from_sq = sq
    occ_own = board.occ[color]
    occ_opp = board.occ[1 - color]
    from_file = sq & 7
    from_rank = sq >> 3
    for d in deltas:
        to = sq + d
        while in_bounds(to):
            to_file = to & 7
            to_rank = to >> 3
            # horizontal: stay on same rank
            if d in (1, -1) and to_rank != from_rank:
                break
            # vertical: stay on same file
            if d in (8, -8) and to_file != from_file:
                break
            # diagonals: |df| == |dr|
            if d in (9, 7, -7, -9):
                df = to_file - from_file
                dr = to_rank - from_rank
                if abs(df) != abs(dr):
                    break
            to_bb = bit(to)
            if to_bb & occ_own:
                break
            if to_bb & occ_opp:
                moves.append(Move(from_sq, to, None, capture=None))
                break
            if not captures_only:
                moves.append(Move(from_sq, to, None))
            to += d


def gen_moves(board: Board, captures_only=False):
    color = board.side_to_move
    occ_own = board.occ[color]
    occ_opp = board.occ[1 - color]
    all_occ = board.all_occ
    moves = []

    # Pawn moves
    pawns = board.bb[color][PAWN]
    forward = 8 if color == WHITE else -8
    start_rank = 1 if color == WHITE else 6
    promo_rank = 6 if color == WHITE else 1
    ep_sq = board.ep_square
    while pawns:
        sq, pawns = pop_lsb(pawns)
        rank = sq >> 3
        # captures
        for df in (-1, 1):
            to = sq + forward + df
            if not in_bounds(to):
                continue
            # prevent wrap across board edges
            if abs((to & 7) - (sq & 7)) != 1:
                continue
            to_bb = bit(to)
            if (to_bb & occ_opp) or (to == ep_sq):
                if rank == promo_rank:
                    for promo_piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append(
                            Move(
                                sq,
                                to,
                                PAWN,
                                capture=None,
                                promo=promo_piece,
                                flags=Move.CAPTURE
                                      | (Move.EN_PASSANT if to == ep_sq else 0),
                            )
                        )
                else:
                    moves.append(
                        Move(
                            sq,
                            to,
                            PAWN,
                            capture=None,
                            flags=Move.CAPTURE
                                  | (Move.EN_PASSANT if to == ep_sq else 0),
                        )
                    )
        if captures_only:
            continue
        # single push
        to = sq + forward
        if in_bounds(to) and not (bit(to) & all_occ):
            if rank == promo_rank:
                for promo_piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append(Move(sq, to, PAWN, promo=promo_piece))
            else:
                moves.append(Move(sq, to, PAWN))
                # double push
                if rank == start_rank:
                    to2 = sq + 2 * forward
                    if in_bounds(to2) and not (bit(to2) & all_occ):
                        moves.append(Move(sq, to2, PAWN))

    # Knights
    knights = board.bb[color][KNIGHT]
    while knights:
        sq, knights = pop_lsb(knights)
        for d in KNIGHT_DELTAS:
            to = sq + d
            if not in_bounds(to):
                continue
            if abs((to & 7) - (sq & 7)) > 2:
                continue
            to_bb = bit(to)
            if to_bb & occ_own:
                continue
            if captures_only and not (to_bb & occ_opp):
                continue
            moves.append(Move(sq, to, KNIGHT, capture=None))

    # Bishops
    bishops = board.bb[color][BISHOP]
    while bishops:
        sq, bishops = pop_lsb(bishops)
        gen_slider_moves(board, color, sq, BISHOP_DELTAS, moves, captures_only)

    # Rooks
    rooks = board.bb[color][ROOK]
    while rooks:
        sq, rooks = pop_lsb(rooks)
        gen_slider_moves(board, color, sq, ROOK_DELTAS, moves, captures_only)

    # Queens
    queens = board.bb[color][QUEEN]
    while queens:
        sq, queens = pop_lsb(queens)
        gen_slider_moves(
            board, color, sq, BISHOP_DELTAS + ROOK_DELTAS, moves, captures_only
        )

    # King
    king_bb = board.bb[color][KING]
    if king_bb:
        sq = lsb(king_bb)
        for d in KING_DELTAS:
            to = sq + d
            if not in_bounds(to):
                continue
            if abs((to & 7) - (sq & 7)) > 1:
                continue
            to_bb = bit(to)
            if to_bb & occ_own:
                continue
            if captures_only and not (to_bb & occ_opp):
                continue
            moves.append(Move(sq, to, KING, capture=None))

        if not captures_only:
            # Castling: check if king is in check and if squares are attacked
            opp = 1 - color
            king_in_check = is_square_attacked(board, sq, opp)

            if color == WHITE:
                if board.castling & 1:  # Kingside
                    f1 = fr_to_sq("f", "1")
                    g1 = fr_to_sq("g", "1")
                    if not (all_occ & (bit(f1) | bit(g1))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, f1, opp) and not is_square_attacked(
                                board, g1, opp):
                            moves.append(
                                Move(sq, g1, KING, flags=Move.CASTLE)
                            )
                if board.castling & 2:  # Queenside
                    d1 = fr_to_sq("d", "1")
                    c1 = fr_to_sq("c", "1")
                    if not (all_occ & (bit(d1) | bit(c1))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, d1, opp) and not is_square_attacked(
                                board, c1, opp):
                            moves.append(
                                Move(sq, c1, KING, flags=Move.CASTLE)
                            )
            else:
                if board.castling & 4:  # Kingside
                    f8 = fr_to_sq("f", "8")
                    g8 = fr_to_sq("g", "8")
                    if not (all_occ & (bit(f8) | bit(g8))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, f8, opp) and not is_square_attacked(
                                board, g8, opp):
                            moves.append(
                                Move(sq, g8, KING, flags=Move.CASTLE)
                            )
                if board.castling & 8:  # Queenside
                    d8 = fr_to_sq("d", "8")
                    c8 = fr_to_sq("c", "8")
                    if not (all_occ & (bit(d8) | bit(c8))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, d8, opp) and not is_square_attacked(
                                board, c8, opp):
                            moves.append(
                                Move(sq, c8, KING, flags=Move.CASTLE)
                            )

    # Assign capture piece types (for MVV-LVA) lazily here
    for m in moves:
        to_bb = bit(m.to_sq)
        if m.flags & Move.EN_PASSANT:
            m.capture = PAWN
        else:
            for p in range(6):
                if to_bb & board.bb[1 - color][p]:
                    m.capture = p
                    break
    # Filter illegal (king in check) moves
    legal = []
    for m in moves:
        if make_move(board, m, legal_check_only=True):
            undo_move(board)
            legal.append(m)
    return legal


# Light cache for is_square_attacked (one slot; often hits for king squares in search)
_attacked_cache = [None, -1, -1, None]  # [key, sq, by_color, result]; list to allow in-place update


def _set_attacked_cache(key, sq, by_color, result):
    _attacked_cache[0], _attacked_cache[1], _attacked_cache[2], _attacked_cache[3] = key, sq, by_color, result


def is_square_attacked(board: Board, sq, by_color):
    if not in_bounds(sq):
        return False
    if _attacked_cache[0] is not None and _attacked_cache[0] == board.zobrist_key and _attacked_cache[1] == sq and _attacked_cache[2] == by_color:
        return _attacked_cache[3]
    all_occ = board.all_occ

    # Pawns
    if by_color == WHITE:
        for df in (-1, 1):
            from_sq = sq - 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[WHITE][PAWN]:
                if (from_sq >> 3) + 1 == (sq >> 3):
                    _set_attacked_cache(board.zobrist_key, sq, by_color, True)
                    return True
    else:
        for df in (-1, 1):
            from_sq = sq + 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[BLACK][PAWN]:
                if (from_sq >> 3) - 1 == (sq >> 3):
                    _set_attacked_cache(board.zobrist_key, sq, by_color, True)
                    return True

    # Knights
    for d in KNIGHT_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq):
            continue
        if abs((from_sq & 7) - (sq & 7)) > 2:
            continue
        if bit(from_sq) & board.bb[by_color][KNIGHT]:
            _set_attacked_cache(board.zobrist_key, sq, by_color, True)
            return True

    # Bishops / Queens
    from_file = sq & 7
    from_rank = sq >> 3
    for d in BISHOP_DELTAS:
        to = sq + d
        while in_bounds(to):
            to_file = to & 7
            to_rank = to >> 3
            df = to_file - from_file
            dr = to_rank - from_rank
            if abs(df) != abs(dr):
                break
            to_bb = bit(to)
            if to_bb & all_occ:
                if to_bb & (board.bb[by_color][BISHOP] | board.bb[by_color][QUEEN]):
                    _set_attacked_cache(board.zobrist_key, sq, by_color, True)
                    return True
                break
            to += d

    # Rooks / Queens
    for d in ROOK_DELTAS:
        to = sq + d
        while in_bounds(to):
            to_file = to & 7
            to_rank = to >> 3
            if d in (1, -1) and to_rank != (sq >> 3):
                break
            if d in (8, -8) and to_file != (sq & 7):
                break
            to_bb = bit(to)
            if to_bb & all_occ:
                if to_bb & (board.bb[by_color][ROOK] | board.bb[by_color][QUEEN]):
                    _set_attacked_cache(board.zobrist_key, sq, by_color, True)
                    return True
                break
            to += d

    # King
    for d in KING_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq):
            continue
        if abs((from_sq & 7) - (sq & 7)) > 1:
            continue
        if bit(from_sq) & board.bb[by_color][KING]:
            _set_attacked_cache(board.zobrist_key, sq, by_color, True)
            return True
    _set_attacked_cache(board.zobrist_key, sq, by_color, False)
    return False


def in_check(board: Board):
    """True if the side to move's king is attacked."""
    color = board.side_to_move
    king_sq = lsb(board.bb[color][KING])
    if king_sq == -1:
        return False
    return is_square_attacked(board, king_sq, 1 - color)


def _slider_attackers_to_sq(board: Board, sq, by_color, deltas, piece_types):
    """Return piece type of first attacker along any delta, or None. piece_types e.g. (BISHOP, QUEEN)."""
    from_f = sq & 7
    from_r = sq >> 3
    all_occ = board.all_occ
    for d in deltas:
        to = sq + d
        while in_bounds(to):
            to_f = to & 7
            to_r = to >> 3
            if deltas == BISHOP_DELTAS and abs(to_f - from_f) != abs(to_r - from_r):
                break
            if deltas == ROOK_DELTAS:
                if d in (1, -1) and to_r != from_r:
                    break
                if d in (8, -8) and to_f != from_f:
                    break
            to_bb = bit(to)
            if to_bb & all_occ:
                for pt in piece_types:
                    if to_bb & board.bb[by_color][pt]:
                        return pt
                break
            to += d
    return None


def get_attackers(board: Board, sq, by_color):
    """Return list of piece types (PAWN..QUEEN, not KING) that attack sq. Heuristic; KING excluded."""
    out = []
    if not in_bounds(sq):
        return out
    if by_color == WHITE:
        for df in (-1, 1):
            from_sq = sq - 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[WHITE][PAWN]:
                if (from_sq >> 3) + 1 == (sq >> 3):
                    out.append(PAWN)
                    break
    else:
        for df in (-1, 1):
            from_sq = sq + 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[BLACK][PAWN]:
                if (from_sq >> 3) - 1 == (sq >> 3):
                    out.append(PAWN)
                    break
    for d in KNIGHT_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq) or abs((from_sq & 7) - (sq & 7)) > 2:
            continue
        if bit(from_sq) & board.bb[by_color][KNIGHT]:
            out.append(KNIGHT)
            break
    pt = _slider_attackers_to_sq(board, sq, by_color, BISHOP_DELTAS, (BISHOP, QUEEN))
    if pt is not None:
        out.append(pt)
    pt = _slider_attackers_to_sq(board, sq, by_color, ROOK_DELTAS, (ROOK, QUEEN))
    if pt is not None:
        out.append(pt)
    return out


def is_open_file(board: Board, file):
    """True if there are no pawns on the given file (0..7). Uses cached FILE_BB."""
    return not (FILE_BB[file] & (board.bb[WHITE][PAWN] | board.bb[BLACK][PAWN]))


def is_zugzwang_prone(board: Board):
    """
    True if null move should be disabled: pawn-only endgame or very low
    non-pawn material (zugzwang-prone).
    """
    for color in (WHITE, BLACK):
        non_pawn = 0
        for p in (KNIGHT, BISHOP, ROOK, QUEEN):
            non_pawn += popcount(board.bb[color][p])
        if non_pawn == 0:
            # Pawn-only (or king-only) for at least one side
            return True
    total_non_pawn = 0
    for color in (WHITE, BLACK):
        for p in (KNIGHT, BISHOP, ROOK, QUEEN):
            total_non_pawn += popcount(board.bb[color][p])
    if total_non_pawn <= 2:
        return True
    return False


def make_null_move(board: Board):
    """Apply null move (switch side, clear ep). Returns saved state for undo."""
    saved = (board.side_to_move, board.ep_square, board.zobrist_key)
    if board.ep_square != -1:
        board.zobrist_key ^= ZOBRIST_EP[board.ep_square]
        board.ep_square = -1
    board.side_to_move = 1 - board.side_to_move
    board.zobrist_key ^= ZOBRIST_SIDE
    return saved


def undo_null_move(board: Board, saved):
    """Restore board after null move. Does not touch history/hash_history."""
    board.side_to_move, board.ep_square, board.zobrist_key = saved


class Undo:
    __slots__ = (
        "move",
        "castling",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
        "zobrist_key",
        "eval_mg",
        "eval_eg",
        "phase",
        "pushed_hash",
    )

    def __init__(self, move, board: Board):
        self.move = move
        self.castling = board.castling
        self.ep_square = board.ep_square
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number
        self.zobrist_key = board.zobrist_key
        self.eval_mg = board.eval_mg
        self.eval_eg = board.eval_eg
        self.phase = board.phase
        self.pushed_hash = False


def make_move(board: Board, move: Move, legal_check_only=False):
    color = board.side_to_move
    opp = 1 - color

    undo = Undo(move, board)
    board.history.append(undo)

    # update hash
    key = board.zobrist_key

    # Remove ep
    if board.ep_square != -1:
        key ^= ZOBRIST_EP[board.ep_square]
    board.ep_square = -1

    # Side
    key ^= ZOBRIST_SIDE

    from_bb = bit(move.from_sq)
    to_bb = bit(move.to_sq)

    # Identify moving piece type if not set
    piece = move.piece
    if piece is None:
        for p in range(6):
            if from_bb & board.bb[color][p]:
                piece = p
                break
        move.piece = piece

    # Remove moving piece from from_sq (for castle, key update done in castle block)
    board.bb[color][piece] ^= from_bb
    board.occ[color] ^= from_bb
    board.all_occ ^= from_bb
    is_castle = bool(move.flags & Move.CASTLE)
    if not is_castle:
        key ^= ZOBRIST_PIECE[color][piece][move.from_sq]

    captured_piece_type = move.capture

    if move.flags & Move.EN_PASSANT:
        cap_sq = move.to_sq + (-8 if color == WHITE else 8)
        cap_bb = bit(cap_sq)
        board.bb[opp][PAWN] ^= cap_bb
        board.occ[opp] ^= cap_bb
        board.all_occ ^= cap_bb
        key ^= ZOBRIST_PIECE[opp][PAWN][cap_sq]
    elif captured_piece_type is not None:
        # remove captured on to_sq
        for p in range(6):
            if to_bb & board.bb[opp][p]:
                board.bb[opp][p] ^= to_bb
                board.occ[opp] ^= to_bb
                board.all_occ ^= to_bb
                key ^= ZOBRIST_PIECE[opp][p][move.to_sq]
                break

    # Castling rights
    old_castling = board.castling
    # moving king
    if piece == KING:
        if color == WHITE:
            board.castling &= ~3
        else:
            board.castling &= ~12
    # moving rook from initial squares
    if from_bb & bit(fr_to_sq("a", "1")) or to_bb & bit(fr_to_sq("a", "1")):
        board.castling &= ~2
    if from_bb & bit(fr_to_sq("h", "1")) or to_bb & bit(fr_to_sq("h", "1")):
        board.castling &= ~1
    if from_bb & bit(fr_to_sq("a", "8")) or to_bb & bit(fr_to_sq("a", "8")):
        board.castling &= ~8
    if from_bb & bit(fr_to_sq("h", "8")) or to_bb & bit(fr_to_sq("h", "8")):
        board.castling &= ~4
    if old_castling != board.castling:
        key ^= ZOBRIST_CASTLING[old_castling]
        key ^= ZOBRIST_CASTLING[board.castling]

    # Castling move rook (Zobrist: one precomputed XOR for king+rook)
    if move.flags & Move.CASTLE:
        if color == WHITE:
            if move.to_sq == fr_to_sq("g", "1"):
                rook_from, rook_to = fr_to_sq("h", "1"), fr_to_sq("f", "1")
                key ^= ZOBRIST_CASTLE_PIECE[WHITE][0]
            else:
                rook_from, rook_to = fr_to_sq("a", "1"), fr_to_sq("d", "1")
                key ^= ZOBRIST_CASTLE_PIECE[WHITE][1]
        else:
            if move.to_sq == fr_to_sq("g", "8"):
                rook_from, rook_to = fr_to_sq("h", "8"), fr_to_sq("f", "8")
                key ^= ZOBRIST_CASTLE_PIECE[BLACK][0]
            else:
                rook_from, rook_to = fr_to_sq("a", "8"), fr_to_sq("d", "8")
                key ^= ZOBRIST_CASTLE_PIECE[BLACK][1]
        rf_bb = bit(rook_from)
        rt_bb = bit(rook_to)
        board.bb[color][ROOK] ^= rf_bb
        board.occ[color] ^= rf_bb
        board.all_occ ^= rf_bb
        board.bb[color][ROOK] ^= rt_bb
        board.occ[color] ^= rt_bb
        board.all_occ ^= rt_bb

    # Promotion (for castle, king already accounted in ZOBRIST_CASTLE_PIECE)
    if piece == PAWN and move.promo is not None:
        promo = move.promo
        board.bb[color][promo] ^= to_bb
        key ^= ZOBRIST_PIECE[color][promo][move.to_sq]
    else:
        board.bb[color][piece] ^= to_bb
        if not is_castle:
            key ^= ZOBRIST_PIECE[color][piece][move.to_sq]
    board.occ[color] ^= to_bb
    board.all_occ ^= to_bb

    # Set EP
    if piece == PAWN and abs(move.to_sq - move.from_sq) == 16:
        ep_sq = (move.to_sq + move.from_sq) // 2
        board.ep_square = ep_sq
        key ^= ZOBRIST_EP[ep_sq]

    # Halfmove clock
    if piece == PAWN or captured_piece_type is not None or (move.flags & Move.EN_PASSANT):
        board.halfmove_clock = 0
    else:
        board.halfmove_clock += 1

    if color == BLACK:
        board.fullmove_number += 1

    board.side_to_move = opp
    board.zobrist_key = key

    # Incremental eval update: only the piece sum (eval_mg, eval_eg, phase)
    dm_from_mg, dm_from_eg = _piece_eval_contribution(color, piece, move.from_sq)
    board.eval_mg -= dm_from_mg
    board.eval_eg -= dm_from_eg
    board.phase -= _phase_delta(piece)
    if move.flags & Move.EN_PASSANT:
        cap_sq = move.to_sq + (-8 if color == WHITE else 8)
        dc_mg, dc_eg = _piece_eval_contribution(opp, PAWN, cap_sq)
        board.eval_mg -= dc_mg
        board.eval_eg -= dc_eg
        board.phase -= _phase_delta(PAWN)
    elif captured_piece_type is not None:
        dc_mg, dc_eg = _piece_eval_contribution(opp, captured_piece_type, move.to_sq)
        board.eval_mg -= dc_mg
        board.eval_eg -= dc_eg
        board.phase -= _phase_delta(captured_piece_type)
    if move.flags & Move.CASTLE:
        if color == WHITE:
            rook_from = fr_to_sq("h", "1") if move.to_sq == fr_to_sq("g", "1") else fr_to_sq("a", "1")
            rook_to = fr_to_sq("f", "1") if move.to_sq == fr_to_sq("g", "1") else fr_to_sq("d", "1")
        else:
            rook_from = fr_to_sq("h", "8") if move.to_sq == fr_to_sq("g", "8") else fr_to_sq("a", "8")
            rook_to = fr_to_sq("f", "8") if move.to_sq == fr_to_sq("g", "8") else fr_to_sq("d", "8")
        drf_mg, drf_eg = _piece_eval_contribution(color, ROOK, rook_from)
        drt_mg, drt_eg = _piece_eval_contribution(color, ROOK, rook_to)
        board.eval_mg = board.eval_mg - drf_mg + drt_mg
        board.eval_eg = board.eval_eg - drf_eg + drt_eg
    if piece == PAWN and move.promo is not None:
        promo = move.promo
        dp_mg, dp_eg = _piece_eval_contribution(color, promo, move.to_sq)
        board.eval_mg += dp_mg
        board.eval_eg += dp_eg
        board.phase += _phase_delta(promo)
    else:
        dp_mg, dp_eg = _piece_eval_contribution(color, piece, move.to_sq)
        board.eval_mg += dp_mg
        board.eval_eg += dp_eg
        board.phase += _phase_delta(piece)
    board.phase = max(0, min(24, board.phase))

    # Check legality: own king (the side that just moved = `color`) not in check by opponent (`opp`)
    if piece == KING:
        king_sq = move.to_sq
    else:
        king_sq = lsb(board.bb[color][KING])
    if king_sq == -1 or is_square_attacked(board, king_sq, opp):
        # illegal move, undo immediately
        undo_move(board)
        return False

    if not legal_check_only:
        board.hash_history.append(board.zobrist_key)
        k = board.zobrist_key
        board.hash_count[k] = board.hash_count.get(k, 0) + 1
        board.history[-1].pushed_hash = True
    return True


def undo_move(board: Board):
    undo = board.history.pop()
    move = undo.move
    board.side_to_move = 1 - board.side_to_move
    color = board.side_to_move
    opp = 1 - color

    board.castling = undo.castling
    board.ep_square = undo.ep_square
    board.halfmove_clock = undo.halfmove_clock
    board.fullmove_number = undo.fullmove_number
    board.zobrist_key = undo.zobrist_key
    board.eval_mg = undo.eval_mg
    board.eval_eg = undo.eval_eg
    board.phase = undo.phase
    if undo.pushed_hash and board.hash_history:
        k = board.hash_history.pop()
        board.hash_count[k] = board.hash_count.get(k, 1) - 1
        if board.hash_count[k] <= 0:
            del board.hash_count[k]

    from_bb = bit(move.from_sq)
    to_bb = bit(move.to_sq)

    piece = move.piece
    # Remove piece from to_sq
    if piece == PAWN and move.promo is not None:
        # remove promo piece
        board.bb[color][move.promo] ^= to_bb
    else:
        board.bb[color][piece] ^= to_bb
    board.occ[color] ^= to_bb
    board.all_occ ^= to_bb

    # Restore moving piece to from_sq
    board.bb[color][piece] ^= from_bb
    board.occ[color] ^= from_bb
    board.all_occ ^= from_bb

    # Restore captured
    if move.flags & Move.EN_PASSANT:
        cap_sq = move.to_sq + (-8 if color == WHITE else 8)
        cap_bb = bit(cap_sq)
        board.bb[opp][PAWN] ^= cap_bb
        board.occ[opp] ^= cap_bb
        board.all_occ ^= cap_bb
    elif move.capture is not None:
        board.bb[opp][move.capture] ^= to_bb
        board.occ[opp] ^= to_bb
        board.all_occ ^= to_bb

    # Undo castling rook move
    if move.flags & Move.CASTLE:
        if color == WHITE:
            if move.to_sq == fr_to_sq("g", "1"):
                rook_from = fr_to_sq("h", "1")
                rook_to = fr_to_sq("f", "1")
            else:
                rook_from = fr_to_sq("a", "1")
                rook_to = fr_to_sq("d", "1")
        else:
            if move.to_sq == fr_to_sq("g", "8"):
                rook_from = fr_to_sq("h", "8")
                rook_to = fr_to_sq("f", "8")
            else:
                rook_from = fr_to_sq("a", "8")
                rook_to = fr_to_sq("d", "8")
        rf_bb = bit(rook_from)
        rt_bb = bit(rook_to)
        board.bb[color][ROOK] ^= rt_bb
        board.occ[color] ^= rt_bb
        board.all_occ ^= rt_bb
        board.bb[color][ROOK] ^= rf_bb
        board.occ[color] ^= rf_bb
        board.all_occ ^= rf_bb


# Evaluation

MG_PST = {
    PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    ROOK: [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ],
}

EG_KING_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

# Endgame PST for all pieces (king uses EG_KING_PST). Improves eval in endgame.
EG_PST = {
    PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        3, 6, 6, 6, 6, 6, 6, 3,
        4, 8, 12, 14, 14, 12, 8, 4,
        6, 12, 18, 22, 22, 18, 12, 6,
        10, 18, 26, 32, 32, 26, 18, 10,
        18, 28, 38, 44, 44, 38, 28, 18,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    ROOK: [
        0, 0, 0, 2, 2, 0, 0, 0,
        -2, 0, 0, 0, 0, 0, 0, -2,
        -2, 0, 0, 0, 0, 0, 0, -2,
        -2, 0, 0, 0, 0, 0, 0, -2,
        -2, 0, 0, 0, 0, 0, 0, -2,
        -2, 0, 0, 0, 0, 0, 0, -2,
        2, 4, 4, 4, 4, 4, 4, 2,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
}

# Cached file bitboards for eval and open-file checks (avoid recomputing every eval).
FILE_BB = [0] * 8
for _f in range(8):
    for _r in range(8):
        FILE_BB[_f] |= bit(sq_index(_f, _r))

# Center squares (d4,d5,e4,e5) for piece interaction bonuses
CENTER_SQS = {sq_index(3, 3), sq_index(3, 4), sq_index(4, 3), sq_index(4, 4)}

# Max positional bonus (cp) for new terms; scale by phase
EVAL_BONUS_CAP = 50

PIECE_VALUES_MG = [100, 320, 330, 500, 900, 0]
PIECE_VALUES_EG = [120, 300, 320, 500, 900, 0]

# Phase weights for incremental update (KNIGHT, BISHOP, ROOK, QUEEN)
PHASE_WEIGHTS = (1, 1, 2, 4)


def _piece_eval_contribution(color, piece, sq):
    """Return (mg, eg) contribution of one piece on sq. Used for incremental eval. EG uses EG_PST for all pieces."""
    sign = 1 if color == WHITE else -1
    mirror_sq = sq ^ 56 if color == BLACK else sq
    if piece == KING:
        mg = sign * (PIECE_VALUES_MG[KING] + MG_PST[KING][mirror_sq])
        eg = sign * (PIECE_VALUES_EG[KING] + EG_KING_PST[mirror_sq])
    else:
        mg = sign * (PIECE_VALUES_MG[piece] + MG_PST[piece][mirror_sq])
        eg = sign * (PIECE_VALUES_EG[piece] + EG_PST[piece][mirror_sq])
    return mg, eg


def _phase_delta(piece):
    """Return phase weight for piece (0 for PAWN/KING)."""
    if piece in (PAWN, KING):
        return 0
    return PHASE_WEIGHTS[piece - 2]  # KNIGHT=0, BISHOP=1, ROOK=2, QUEEN=3


def game_phase(board: Board):
    # simple: sum piece values excluding pawns and kings
    phase = 0
    for c in (WHITE, BLACK):
        for p, w in zip(
                (KNIGHT, BISHOP, ROOK, QUEEN),
                (1, 1, 2, 4),
        ):
            phase += popcount(board.bb[c][p]) * w
    return max(0, min(24, phase))


def min_attacker_value(board: Board, sq, by_color):
    """Minimum PIECE_VALUES_MG among attackers (excluding king). Large constant if none."""
    a = get_attackers(board, sq, by_color)
    if not a:
        return 10_000_000
    return min(PIECE_VALUES_MG[p] for p in a)


def hang_penalty_for_color(board: Board, color):
    """Heuristic penalty for pieces of color that are attacked and undefended. Used in move ordering."""
    opp = 1 - color
    total = 0
    for p in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
        bb = board.bb[color][p]
        while bb:
            sq, bb = pop_lsb(bb)
            if min_attacker_value(board, sq, opp) < 10_000_000 and not get_attackers(board, sq, color):
                total += max(50, PIECE_VALUES_MG[p] // 2)
    return total


def get_hanging_squares(board: Board, color):
    """Set of squares where color has a piece (N,B,R,Q) that is attacked and not defended. For blunder ordering."""
    opp = 1 - color
    out = set()
    for p in (KNIGHT, BISHOP, ROOK, QUEEN):
        bb = board.bb[color][p]
        while bb:
            sq, bb = pop_lsb(bb)
            if min_attacker_value(board, sq, opp) < 10_000_000 and not get_attackers(board, sq, color):
                out.add(sq)
    return out


def bad_capture_ordering_penalty(board: Board, to_sq, our_piece_value, opp_color):
    """
    SEE-lite for root capture ordering only. Board is *after* our capture (our piece on to_sq, opp to move).
    If the opponent can recapture with a cheaper piece, return a small ordering penalty (tens of cp).
    No full SEE, no recursion. Used only to demote fake-good captures; never prunes.
    """
    m_opp = min_attacker_value(board, to_sq, opp_color)
    if m_opp >= our_piece_value or m_opp >= 10_000_000:
        return 0
    loss = our_piece_value - m_opp
    return min(100, max(20, loss // 5))


def eval_board(board: Board):
    # positive is good for side to move (we return from POV of side_to_move later)
    # Use incremental piece sum (eval_mg, eval_eg, phase); rest computed here
    score = board.eval_mg
    eg = board.eval_eg
    phase = board.phase
    bishop_count = [0, 0]
    passed_pawns = [0, 0]
    passed_rank_bonus = [0, 0]  # extra by rank (closer to promotion)
    doubled_pawns = [0, 0]
    weak_pawns = [0, 0]
    rook_open_bonus = [0, 0]
    rook_semi_open_bonus = [0, 0]
    rook_7th_bonus = [0, 0]
    knight_center_bonus = [0, 0]
    bishop_center_bonus = [0, 0]
    knight_outpost_bonus = [0, 0]

    for color in (WHITE, BLACK):
        for p in range(6):
            bb = board.bb[color][p]
            while bb:
                sq, bb = pop_lsb(bb)
                if p == BISHOP:
                    bishop_count[color] += 1
                    if sq in CENTER_SQS:
                        bishop_center_bonus[color] += 12
                if p == KNIGHT:
                    if sq in CENTER_SQS:
                        knight_center_bonus[color] += 15
                    # Outpost: knight cannot be driven by enemy pawn (no enemy pawn can attack this sq)
                    f, r = sq & 7, sq >> 3
                    if color == WHITE:
                        if r >= 2 and f >= 1 and f <= 6:
                            attack_sq1 = sq_index(f - 1, r + 1) if r + 1 <= 7 else -1
                            attack_sq2 = sq_index(f + 1, r + 1) if r + 1 <= 7 else -1
                            if (attack_sq1 < 0 or not (bit(attack_sq1) & board.bb[BLACK][PAWN])) and (
                                    attack_sq2 < 0 or not (bit(attack_sq2) & board.bb[BLACK][PAWN])):
                                knight_outpost_bonus[color] += 12
                    else:
                        if r <= 5 and f >= 1 and f <= 6:
                            attack_sq1 = sq_index(f - 1, r - 1) if r - 1 >= 0 else -1
                            attack_sq2 = sq_index(f + 1, r - 1) if r - 1 >= 0 else -1
                            if (attack_sq1 < 0 or not (bit(attack_sq1) & board.bb[WHITE][PAWN])) and (
                                    attack_sq2 < 0 or not (bit(attack_sq2) & board.bb[WHITE][PAWN])):
                                knight_outpost_bonus[color] += 12
                if p == ROOK:
                    f = sq & 7
                    r = sq >> 3
                    if is_open_file(board, f):
                        rook_open_bonus[color] += 18
                    else:
                        our_pawns_on_f = FILE_BB[f] & board.bb[color][PAWN]
                        if not our_pawns_on_f:
                            rook_semi_open_bonus[color] += 10
                    if (color == WHITE and r == 6) or (color == BLACK and r == 1):
                        rook_7th_bonus[color] += min(EVAL_BONUS_CAP, 15)
                if p == PAWN:
                    f = sq & 7
                    pawn_rank = sq >> 3
                    ahead_mask = 0
                    if color == WHITE:
                        for r in range(pawn_rank + 1, 8):
                            ahead_mask |= bit(sq_index(f, r))
                    else:
                        for r in range(pawn_rank - 1, -1, -1):
                            ahead_mask |= bit(sq_index(f, r))
                    neighbor_mask = FILE_BB[f]
                    if f > 0:
                        neighbor_mask |= FILE_BB[f - 1]
                    if f < 7:
                        neighbor_mask |= FILE_BB[f + 1]
                    opp_pawns = board.bb[1 - color][PAWN]
                    if not (opp_pawns & neighbor_mask & ahead_mask):
                        passed_pawns[color] += 1
                        rank_for_bonus = (pawn_rank if color == WHITE else (7 - pawn_rank))
                        passed_rank_bonus[color] += rank_for_bonus * 8
                    same_file_ours = FILE_BB[f] & board.bb[color][PAWN]
                    if popcount(same_file_ours) > 1:
                        doubled_pawns[color] += 1
                    no_own_ahead = not (board.bb[color][PAWN] & ahead_mask)
                    opp_adjacent = bool(opp_pawns & neighbor_mask)
                    if no_own_ahead and opp_adjacent and (pawn_rank < 5 if color == WHITE else pawn_rank > 2):
                        weak_pawns[color] += 1

    # Bishop pair: full bonus in MG, slightly less in EG when phase is low
    bishop_pair_bonus = 30
    if bishop_count[WHITE] >= 2:
        score += bishop_pair_bonus
    if bishop_count[BLACK] >= 2:
        score -= bishop_pair_bonus

    # Passed pawns: base + rank-based (closer = more); rank bonus slightly stronger in endgame (no sharp jump)
    passed_scale = 24 + (24 - phase)
    score += 20 * passed_pawns[WHITE] + (passed_rank_bonus[WHITE] * passed_scale // 24)
    score -= 20 * passed_pawns[BLACK] + (passed_rank_bonus[BLACK] * passed_scale // 24)
    score -= 15 * doubled_pawns[WHITE]
    score += 15 * doubled_pawns[BLACK]
    score -= 12 * weak_pawns[WHITE]
    score += 12 * weak_pawns[BLACK]

    # Piece interaction: rooks open/semi-open/7th, center, knight outpost (new terms capped at EVAL_BONUS_CAP)
    score += (rook_open_bonus[WHITE] + rook_semi_open_bonus[WHITE] + min(EVAL_BONUS_CAP, rook_7th_bonus[WHITE]) +
              knight_center_bonus[WHITE] + bishop_center_bonus[WHITE] + min(EVAL_BONUS_CAP, knight_outpost_bonus[WHITE]))
    score -= (rook_open_bonus[BLACK] + rook_semi_open_bonus[BLACK] + min(EVAL_BONUS_CAP, rook_7th_bonus[BLACK]) +
              knight_center_bonus[BLACK] + bishop_center_bonus[BLACK] + min(EVAL_BONUS_CAP, knight_outpost_bonus[BLACK]))

    mg_weight = phase
    eg_weight = 24 - phase

    # --- Blunder avoidance: hanging pieces (eval) ---
    # Penalize pieces that are attacked and undefended. Conservative: only clear hangs.
    # Does not penalize defended pieces (winning trades, sacrifices with compensation).
    for color in (WHITE, BLACK):
        opp = 1 - color
        sign = 1 if color == WHITE else -1
        for p in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
            bb = board.bb[color][p]
            while bb:
                sq, bb = pop_lsb(bb)
                attacked = min_attacker_value(board, sq, opp) < 10_000_000
                defended = len(get_attackers(board, sq, color)) > 0
                if attacked and not defended:
                    val = PIECE_VALUES_MG[p]
                    score -= sign * (max(50, val // 2))

    # --- King safety (eval): MG-only ---
    # Open file, centre, pawn shield (pawns in front of king), attack weight.
    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        kbb = board.bb[color][KING]
        if not kbb:
            continue
        ksq = lsb(kbb)
        kf = ksq & 7
        kr = ksq >> 3
        if phase <= 8:
            continue
        if is_open_file(board, kf):
            score -= sign * 22
        if 2 <= kf <= 5 and 2 <= kr <= 5:
            score -= sign * 12
        # Micro-patch: extra penalty if king in center AND (queens on board OR open file near king); middlegame only
        if phase > 8 and 2 <= kf <= 5 and 2 <= kr <= 5:
            queens_out = bool(board.bb[WHITE][QUEEN] or board.bb[BLACK][QUEEN])
            open_near = is_open_file(board, kf) or (kf > 0 and is_open_file(board, kf - 1)) or (kf < 7 and is_open_file(board, kf + 1))
            if queens_out or open_near:
                score -= sign * min(EVAL_BONUS_CAP, 12)
        # Pawn shield: bonus for pawns on 2-3 squares in front of king (MG)
        shield_rank = kr + 1 if color == WHITE else kr - 1
        if 0 <= shield_rank <= 7:
            shield_bb = bit(sq_index(kf, shield_rank))
            if kf > 0:
                shield_bb |= bit(sq_index(kf - 1, shield_rank))
            if kf < 7:
                shield_bb |= bit(sq_index(kf + 1, shield_rank))
            shield_count = popcount(shield_bb & board.bb[color][PAWN])
            score += sign * shield_count * 14
        # Attack weight: penalty for squares around king attacked by opponent
        opp = 1 - color
        attack_weight = 0
        for d in KING_DELTAS:
            nsq = ksq + d
            if not in_bounds(nsq) or abs((nsq & 7) - kf) > 1:
                continue
            if is_square_attacked(board, nsq, opp):
                attack_weight += 1
        score -= sign * attack_weight * 8

    # --- Winning positions: extra king-exposure penalty ---
    # Avoid speculative king exposure / counterplay when clearly ahead.
    if score > 220 and phase > 6:
        kbb = board.bb[WHITE][KING]
        if kbb:
            ksq = lsb(kbb)
            kf, kr = ksq & 7, ksq >> 3
            if is_open_file(board, kf):
                score -= 28
            if 2 <= kf <= 5 and 2 <= kr <= 5:
                score -= 15
    elif score < -220 and phase > 6:
        kbb = board.bb[BLACK][KING]
        if kbb:
            ksq = lsb(kbb)
            kf, kr = ksq & 7, ksq >> 3
            if is_open_file(board, kf):
                score += 28
            if 2 <= kf <= 5 and 2 <= kr <= 5:
                score += 15

    total = (score * mg_weight + eg * eg_weight) // max(1, (mg_weight + eg_weight))

    # perspective
    return total if board.side_to_move == WHITE else -total


INF = 10_000_000
MATE_VALUE = 100_000


def probe_syzygy(board: Board):
    """
    Syzygy tablebase probe (placeholder for future integration).
    Returns (score_in_cp, success) or None if disabled/not in TB.
    Call from search at leaf or low depth when material is in TB; no multithreading.
    """
    return None


class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "move", "age")

    EXACT = 0
    LOWER = 1
    UPPER = 2

    def __init__(self, key=0, depth=0, score=0, flag=0, move=None, age=0):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move
        self.age = age


class Searcher:
    def __init__(self):
        self.tt_size = 1 << 20
        self.tt = [TTEntry() for _ in range(self.tt_size)]
        self.nodes = 0
        self.start_time = 0
        self.stop_time = 0
        self.stop = False
        self.best_move = None
        self.history_heur = defaultdict(int)
        self.killer1 = {}  # killer1[ply] = (from_sq, to_sq, promo) for quiet moves that caused cutoff
        self.killer2 = {}
        self.counter_move = {}  # counter_move[(color_played, from, to, promo)] = (from, to, promo) of reply
        self.age = 0
        self.max_depth = 0
        self.root_moves = []
        self.root_board = None
        self.root_eval = 0

    def clear(self):
        self.tt = [TTEntry() for _ in range(self.tt_size)]
        self.history_heur.clear()
        self.killer1 = {}
        self.killer2 = {}
        self.counter_move = {}
        self.age = 0

    def probe_tt(self, key):
        entry = self.tt[key & (self.tt_size - 1)]
        if entry.key == key:
            return entry
        return None

    def store_tt(self, key, depth, score, flag, move):
        idx = key & (self.tt_size - 1)
        entry = self.tt[idx]
        # Replace: different position, stale age, or strictly better (deeper / same depth but we have EXACT)
        replace = (
            entry.key != key
            or entry.age != self.age
            or depth > entry.depth
            or (depth == entry.depth and flag == TTEntry.EXACT and entry.flag != TTEntry.EXACT)
        )
        if replace:
            self.tt[idx] = TTEntry(key, depth, score, flag, move, self.age)

    def time_up(self):
        return self.stop or (self.stop_time and time.time() >= self.stop_time)

    def search(self, board: Board, max_depth, time_limit=None):
        self.nodes = 0
        self.start_time = time.time()
        self.stop_time = self.start_time + time_limit if time_limit else 0
        self.stop = False
        self.best_move = None
        self.root_board = board.clone()
        self.age += 1
        self.root_moves = gen_moves(board)
        self.max_depth = max_depth

        alpha = -INF
        beta = INF
        last_score = 0
        for depth in range(1, max_depth + 1):
            if self.time_up():
                break
            self.age += 1
            self.order_board = self.root_board.clone()
            score = self.negamax(board, depth, alpha, beta, 0, True, True, False, None)
            if self.time_up():
                break
            last_score = score
            if score <= alpha or score >= beta:
                alpha = -INF
                beta = INF
                score = self.negamax(board, depth, alpha, beta, 0, True, True, False, None)
                if self.time_up():
                    break
            alpha = score - 50
            beta = score + 50
            # send info
            elapsed = max(0.001, time.time() - self.start_time)
            nps = int(self.nodes / elapsed)
            score_cp, mate = self.score_to_uci(score)
            score_str = (
                f"mate {mate}" if mate is not None else f"cp {score_cp}"
            )
            pv = self.get_pv(board, depth)
            pv_str = " ".join(m.uci() for m in pv)
            print(
                f"info depth {depth} nodes {self.nodes} nps {nps} score {score_str} pv {pv_str}",
                flush=True,
            )
        self.root_eval = last_score

        # Final safety check: ensure reported best move is legal in the current root position.
        legal_root_moves = gen_moves(board)
        if not legal_root_moves:
            self.best_move = None
        elif self.best_move is None or not any(
                m.from_sq == self.best_move.from_sq
                and m.to_sq == self.best_move.to_sq
                and m.promo == self.best_move.promo
                for m in legal_root_moves
        ):
            # Fall back to the first legal move.
            self.best_move = legal_root_moves[0]

        return self.best_move

    def score_to_uci(self, score):
        if abs(score) > MATE_VALUE - 1000:
            # Convert internal mate score to "mate in N" (plies -> full moves, never 0).
            mate_plies = max(1, MATE_VALUE - abs(score))
            mate_in = (mate_plies + 1) // 2
            if score < 0:
                mate_in = -mate_in
            return None, mate_in
        return score, None

    def get_pv(self, board: Board, depth):
        # For robustness, only report a 1-move PV from the current best root move.
        # This avoids rare cases where stale TT entries could form an illegal multi-move PV.
        if self.best_move is None:
            return []
        return [self.best_move]

    def negamax(self, board: Board, depth, alpha, beta, ply, root=False, allow_null=True, extended=False, prev_move=None):
        if self.time_up():
            return 0
        self.nodes += 1

        # Extensions: in check (depth 1..4, once); pawn to 7th (rank 6 white / rank 1 black)
        do_extend = in_check(board) and 1 <= depth <= 4 and not extended
        next_depth = depth if do_extend else (depth - 1)
        next_extended = extended or do_extend

        # repetition and fifty-move (O(1) via hash_count)
        if board.halfmove_clock >= 100 or board.hash_count.get(board.zobrist_key, 0) >= 3:
            # anti-threefold in winning positions: if eval > threshold, avoid draw
            stand_pat = eval_board(board)
            if stand_pat > 80 and not root:
                # don't accept; treat as bad draw
                pass
            else:
                return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        # next_depth / next_extended set above (check extension)

        key = board.zobrist_key
        tt_entry = self.probe_tt(key)
        if tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score > MATE_VALUE - 1000:
                tt_score -= ply
            elif tt_score < -MATE_VALUE + 1000:
                tt_score += ply
            if tt_entry.flag == TTEntry.EXACT:
                return tt_score
            elif tt_entry.flag == TTEntry.LOWER and tt_score > alpha:
                alpha = tt_score
            elif tt_entry.flag == TTEntry.UPPER and tt_score < beta:
                beta = tt_score
            if alpha >= beta:
                return tt_score

        legal_moves = gen_moves(board)
        if not legal_moves:
            king_sq = lsb(board.bb[board.side_to_move][KING])
            if king_sq != -1 and is_square_attacked(board, king_sq, 1 - board.side_to_move):
                return -MATE_VALUE + ply
            else:
                return 0

        best_score = -INF
        best_move = None

        # Move ordering: TT move first, then MVV-LVA + history
        tt_move = tt_entry.move if tt_entry else None

        # Filter out illegal TT move if present
        if tt_move is not None:
            matched = False
            for m in legal_moves:
                if (
                        m.from_sq == tt_move.from_sq
                        and m.to_sq == tt_move.to_sq
                        and m.promo == tt_move.promo
                ):
                    matched = True
                    break
            if not matched:
                tt_move = None

        phase = board.phase
        order_board = getattr(self, "order_board", None)
        stm = board.side_to_move
        # For blunder ordering: squares where we have a hanging piece (N,B,R,Q) before any move
        hanging_before = (
            get_hanging_squares(board, stm)
            if (depth >= 2 and root and not in_check(board))
            else set()
        )

        def move_score(m):
            score = 0
            if tt_move and m.from_sq == tt_move.from_sq and m.to_sq == tt_move.to_sq and m.promo == tt_move.promo:
                score += 10_000_000
            # Counter-move: weak bonus only, must not override TT or MVV-LVA
            if prev_move is not None and COUNTER_BONUS:
                key = (1 - stm, prev_move.from_sq, prev_move.to_sq, prev_move.promo)
                c = self.counter_move.get(key)
                if c and c[0] == m.from_sq and c[1] == m.to_sq and c[2] == m.promo:
                    score += COUNTER_BONUS
            if m.is_quiet():
                k1 = self.killer1.get(ply)
                if k1 and k1[0] == m.from_sq and k1[1] == m.to_sq and k1[2] == m.promo:
                    score += 2_000_000
                else:
                    k2 = self.killer2.get(ply)
                    if k2 and k2[0] == m.from_sq and k2[1] == m.to_sq and k2[2] == m.promo:
                        score += 1_500_000
            if m.is_capture():
                victim = m.capture if m.capture is not None else PAWN
                attacker = m.piece
                if attacker is None:
                    for pt in range(6):
                        if bit(m.from_sq) & board.bb[stm][pt]:
                            attacker = pt
                            break
                score += 1000 * (
                            PIECE_VALUES_MG[victim] - (PIECE_VALUES_MG[attacker] if attacker is not None else 0) // 10)
            else:
                score += self.history_heur[(stm, m.from_sq, m.to_sq)]
            # Move filtering (ordering): avoid moves that leave pieces en prise when alternatives exist.
            # Only at root, when not in check; heuristic-based. Penalty tries "safe" moves first.
            # Blunder filter (depth>=2, quiet only): king attacked or newly hanging piece -> strong penalty (move to end).
            if root and order_board is not None and not in_check(board):
                if make_move(order_board, m):
                    us = 1 - order_board.side_to_move
                    if depth >= 2 and m.is_quiet() and not m.is_capture() and not (m.flags & Move.CASTLE):
                        king_sq = lsb(order_board.bb[us][KING])
                        if king_sq != -1 and is_square_attacked(order_board, king_sq, order_board.side_to_move):
                            score -= 5_000_000
                        else:
                            hanging_after = get_hanging_squares(order_board, us)
                            newly = set()
                            for sq in hanging_after:
                                if sq == m.to_sq:
                                    if m.from_sq not in hanging_before:
                                        newly.add(sq)
                                else:
                                    if sq not in hanging_before:
                                        newly.add(sq)
                            if newly:
                                score -= 3_000_000
                    # Strong piece protection: avoid unnecessary loss of R/B/Q (ordering only; no pruning)
                    if USE_STRONG_PIECE_PROTECTION and depth >= 2 and m.is_quiet() and not (m.flags & Move.CASTLE):
                        hanging_sp = get_hanging_squares(order_board, us)
                        strong_bb = order_board.bb[us][ROOK] | order_board.bb[us][BISHOP] | order_board.bb[us][QUEEN]
                        if any(bit(sq) & strong_bb for sq in hanging_sp):
                            score -= 5_000_000
                    # Anti-blunder ordering: if opponent has a simple forcing reply (check or good capture >= knight), penalise (no pruning)
                    if (
                            not DISABLE_FORCING_FILTER
                            and depth >= 2
                            and m.is_quiet()
                            and not m.is_capture()
                            and not (m.flags & Move.CASTLE)
                    ):
                        has_forcing = False
                        for opp_move in gen_moves(order_board):
                            if opp_move.is_capture():
                                victim_piece = None
                                for p in range(6):
                                    if bit(opp_move.to_sq) & order_board.bb[us][p]:
                                        victim_piece = p
                                        break
                                if victim_piece is not None and PIECE_VALUES_MG[victim_piece] >= PIECE_VALUES_MG[KNIGHT]:
                                    att = opp_move.piece
                                    if att is None:
                                        for p in range(6):
                                            if bit(opp_move.from_sq) & order_board.bb[order_board.side_to_move][p]:
                                                att = p
                                                break
                                    if att is not None and PIECE_VALUES_MG[att] <= PIECE_VALUES_MG[victim_piece] + 50:
                                        has_forcing = True
                                        break
                            else:
                                if make_move(order_board, opp_move):
                                    if in_check(order_board):
                                        has_forcing = True
                                    undo_move(order_board)
                                    if has_forcing:
                                        break
                        if has_forcing:
                            score -= 2_000_000
                    penalty = hang_penalty_for_color(order_board, us)
                    if m.is_capture() and not (m.flags & Move.CASTLE) and not in_check(order_board):
                        if m.promo is not None:
                            v = PIECE_VALUES_MG[m.promo]
                        else:
                            pt = m.piece
                            if pt is None:
                                for p in range(6):
                                    if bit(m.from_sq) & board.bb[board.side_to_move][p]:
                                        pt = p
                                        break
                            v = PIECE_VALUES_MG[pt] if pt is not None else 0
                        if v:
                            penalty += bad_capture_ordering_penalty(
                                order_board, m.to_sq, v, order_board.side_to_move
                            )
                    undo_move(order_board)
                    score -= penalty
            # King safety: deprioritize quiet king moves in MG (early walks, centralization).
            # Castling excluded. Does not suppress forced or clearly good king moves.
            is_king = bool(bit(m.from_sq) & board.bb[board.side_to_move][KING])
            if is_king and m.is_quiet() and not (m.flags & Move.CASTLE) and phase > 8:
                score -= 80
            return -score

        legal_moves.sort(key=move_score)

        original_alpha = alpha

        # Null-move pruning: try null move if allowed and constraints pass.
        # Disabled: in check, depth < 3, zugzwang-prone, or consecutive null (allow_null=False).
        NULL_R = 3
        if (
                depth >= 3
                and allow_null
                and not in_check(board)
                and not is_zugzwang_prone(board)
        ):
            saved = make_null_move(board)
            null_score = -self.negamax(
                board, depth - 1 - NULL_R, -beta, -beta + 1, ply + 1, root=False, allow_null=False, extended=next_extended, prev_move=None
            )
            undo_null_move(board, saved)
            if self.time_up():
                return 0
            if null_score >= beta:
                return beta

        # LMR: only quiet, depth>=3, not first 2 moves, not check/promo; max reduction 2 (disableable)
        LMR_FULL_MOVES = 2 if USE_LMR_STRICT else 4

        # Precompute for futility and LMR (no extra eval when not needed)
        in_check_before = in_check(board)
        current_eval_for_futility = eval_board(board) if (USE_FUTILITY and depth == 1) else None
        current_eval_for_lmr = eval_board(board) if (USE_LMR_STRICT and depth >= 3) else None
        king_attackers = 0
        if USE_LMR_STRICT and depth >= 3:
            ksq = lsb(board.bb[stm][KING])
            if ksq != -1:
                king_attackers = len(get_attackers(board, ksq, 1 - stm))

        # Soft futility: only depth 1, not in check, eval >= -150; margin 80 cp (defensive positions: don't prune)
        FUTILITY_MARGIN = 80

        for i, m in enumerate(legal_moves):
            # Defensive move detection for LMR (before make_move; board is current node)
            was_attacked = (
                    USE_LMR_STRICT
                    and depth >= 3
                    and min_attacker_value(board, m.from_sq, 1 - stm) < 10_000_000
            )
            if not make_move(board, m):
                continue
            gives_check = in_check(board)
            to_rank = m.to_sq >> 3
            moved_color = 1 - board.side_to_move
            pawn_push_78 = (
                    m.piece == PAWN and not m.is_capture()
                    and ((moved_color == WHITE and to_rank in (5, 6)) or (moved_color == BLACK and to_rank in (1, 2)))
            )
            pawn_to_7th = (
                    m.piece == PAWN and not m.is_capture()
                    and ((moved_color == WHITE and to_rank == 6) or (moved_color == BLACK and to_rank == 1))
            )
            effective_depth = next_depth
            if pawn_to_7th and depth <= 4 and not next_extended:
                effective_depth = depth

            # Futility: only when not in check, eval >= -150; in bad/defensive positions do not prune
            if (
                    USE_FUTILITY
                    and depth == 1
                    and not in_check_before
                    and current_eval_for_futility is not None
                    and current_eval_for_futility >= -150
                    and m.is_quiet()
                    and m.promo is None
                    and not gives_check
            ):
                child_stand_pat = eval_board(board)
                if child_stand_pat < -alpha - FUTILITY_MARGIN:
                    undo_move(board)
                    continue

            # LMR: forbid reduction in dangerous positions (eval < -50, in check, king move, attacked piece, or king has >=2 attackers)
            no_lmr_defensive = (
                    USE_LMR_STRICT
                    and (
                            (current_eval_for_lmr is not None and current_eval_for_lmr < -50)
                            or in_check_before
                            or (m.piece == KING)
                            or was_attacked
                            or king_attackers >= 2
                    )
            )
            do_lmr = (
                    depth >= 3
                    and i >= LMR_FULL_MOVES
                    and m.is_quiet()
                    and not gives_check
                    and not pawn_push_78
                    and m.promo is None
                    and not no_lmr_defensive
            )
            if do_lmr:
                hist = self.history_heur.get((stm, m.from_sq, m.to_sq), 0)
                lmr_reduction = min(2, 1 if hist > 8000 else 2)
                score = -self.negamax(
                    board, effective_depth - lmr_reduction, -beta, -alpha, ply + 1, root=False, allow_null=True, extended=next_extended, prev_move=m
                )
                if score > alpha:
                    score = -self.negamax(
                        board, effective_depth, -beta, -alpha, ply + 1, root=False, allow_null=True, extended=next_extended, prev_move=m
                    )
            else:
                # PVS: when depth>=2, first move full window, rest null-window; re-search on fail-high
                if USE_PVS and depth >= 2 and i > 0:
                    score = -self.negamax(
                        board, effective_depth, -alpha - 1, -alpha, ply + 1, root=False, allow_null=True, extended=next_extended, prev_move=m
                    )
                    if score > alpha and score < beta:
                        score = -self.negamax(
                            board, effective_depth, -beta, -alpha, ply + 1, root=False, allow_null=True, extended=next_extended, prev_move=m
                        )
                else:
                    score = -self.negamax(
                        board, effective_depth, -beta, -alpha, ply + 1, root=False, allow_null=True, extended=next_extended, prev_move=m
                    )
            undo_move(board)
            if self.time_up():
                return 0
            if score > best_score:
                best_score = score
                best_move = m
            if score > alpha:
                alpha = score
                if not m.is_capture():
                    self.history_heur[(stm, m.from_sq, m.to_sq)] += depth * depth
            if alpha >= beta:
                if m.is_quiet():
                    if self.killer1.get(ply) != (m.from_sq, m.to_sq, m.promo):
                        self.killer2[ply] = self.killer1.get(ply)
                        self.killer1[ply] = (m.from_sq, m.to_sq, m.promo)
                break

        if best_move is None:
            return 0

        if prev_move is not None:
            self.counter_move[(1 - stm, prev_move.from_sq, prev_move.to_sq, prev_move.promo)] = (
                best_move.from_sq, best_move.to_sq, best_move.promo
            )

        flag = TTEntry.EXACT
        if best_score <= original_alpha:
            flag = TTEntry.UPPER
        elif best_score >= beta:
            flag = TTEntry.LOWER

        store_score = best_score
        if store_score > MATE_VALUE - 1000:
            store_score += ply
        elif store_score < -MATE_VALUE + 1000:
            store_score -= ply
        self.store_tt(key, depth, store_score, flag, best_move if not root else best_move)

        if root:
            self.best_move = best_move
        return best_score

    def quiescence(self, board: Board, alpha, beta, ply, qply=0):
        """Qsearch: captures, then (if qply < limit) promotions and checking moves. Depth limit avoids check cycles."""
        if self.time_up():
            return 0
        self.nodes += 1
        # Mate-in-1: if side is in check and has no legal move, return mate (no search extension)
        if in_check(board):
            legal = gen_moves(board)
            if not legal:
                return -MATE_VALUE + ply
        stand_pat = eval_board(board)
        # Strong piece protection: avoid unnecessary loss of R/B/Q — reduce stand_pat if we have hanging R/B/Q
        if USE_STRONG_PIECE_PROTECTION and not in_check(board):
            stm = board.side_to_move
            hanging = get_hanging_squares(board, stm)
            strong_bb = board.bb[stm][ROOK] | board.bb[stm][BISHOP] | board.bb[stm][QUEEN]
            for sq in hanging:
                if bit(sq) & strong_bb:
                    for p in (ROOK, BISHOP, QUEEN):
                        if bit(sq) & board.bb[stm][p]:
                            stand_pat -= PIECE_VALUES_MG[p]
                            break
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        captures = gen_moves(board, captures_only=True)
        for m in captures:
            if not m.is_capture() and not (m.flags & Move.EN_PASSANT):
                continue
            if not make_move(board, m):
                continue
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            undo_move(board)
            if self.time_up():
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        # Promotions and quiet checks only within depth limit (no aggressive pruning)
        if QSEARCH_MAX_PLY and qply >= QSEARCH_MAX_PLY:
            return alpha

        all_moves = gen_moves(board, captures_only=False)
        for m in all_moves:
            if m.is_capture() or (m.flags & Move.EN_PASSANT):
                continue
            to_search = m.promo is not None
            if not to_search:
                if make_move(board, m):
                    to_search = in_check(board)
                    undo_move(board)
            if not to_search:
                continue
            if not make_move(board, m):
                continue
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            undo_move(board)
            if self.time_up():
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


class UCI:
    def __init__(self):
        self.board = Board()
        set_fen(self.board, START_FEN)
        self.searcher = Searcher()
        self.uci_name = "StrawberryChess v4.1-preview-1"
        self.uci_author = "MK"

    def loop(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line == "uci":
                self.cmd_uci()
            elif line == "isready":
                print("readyok", flush=True)
            elif line.startswith("position"):
                self.cmd_position(line)
            elif line.startswith("go"):
                self.cmd_go(line)
            elif line == "ucinewgame":
                self.searcher.clear()
                set_fen(self.board, START_FEN)
            elif line == "quit":
                break
            elif line == "stop":
                self.searcher.stop = True

    def cmd_uci(self):
        print(f"id name {self.uci_name}")
        print(f"id author {self.uci_author}")
        print("uciok", flush=True)

    def cmd_position(self, line):
        parts = line.split()
        idx = 1
        if parts[idx] == "startpos":
            set_fen(self.board, START_FEN)
            idx += 1
        elif parts[idx] == "fen":
            fen = " ".join(parts[idx + 1: idx + 7])
            set_fen(self.board, fen)
            idx += 7
        if idx < len(parts) and parts[idx] == "moves":
            idx += 1
            while idx < len(parts):
                mv = parts[idx]
                moves = gen_moves(self.board)
                found = None
                for m in moves:
                    if m.uci() == mv:
                        found = m
                        break
                if found:
                    make_move(self.board, found)
                idx += 1

    def cmd_go(self, line):
        parts = line.split()
        wtime = btime = winc = binc = movestogo = None
        depth = 20
        movetime = None
        i = 1
        while i < len(parts):
            if parts[i] == "wtime":
                wtime = int(parts[i + 1])
                i += 2
            elif parts[i] == "btime":
                btime = int(parts[i + 1])
                i += 2
            elif parts[i] == "winc":
                winc = int(parts[i + 1])
                i += 2
            elif parts[i] == "binc":
                binc = int(parts[i + 1])
                i += 2
            elif parts[i] == "movestogo":
                movestogo = int(parts[i + 1])
                i += 2
            elif parts[i] == "movetime":
                movetime = int(parts[i + 1])
                i += 2
            elif parts[i] == "depth":
                depth = int(parts[i + 1])
                i += 2
            else:
                i += 1

        time_limit = None
        if movetime is not None:
            time_limit = movetime / 1000.0
        elif wtime is not None and btime is not None:
            my_time = wtime if self.board.side_to_move == WHITE else btime
            my_inc = winc if self.board.side_to_move == WHITE else binc
            if movestogo is None:
                movestogo = 30
            time_limit = max(0.05, (my_time / movestogo + (my_inc or 0)) / 1000.0 * 0.7)

        self.searcher.stop = False
        best = self.searcher.search(self.board, depth, time_limit)
        if best is None:
            print("bestmove 0000", flush=True)
        else:
            print(f"bestmove {best.uci()}", flush=True)


def main():
    init_zobrist()
    uci = UCI()
    uci.loop()


if __name__ == "__main__":
    main()
