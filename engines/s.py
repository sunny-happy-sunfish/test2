#!/usr/bin/env python3

import sys
import time
import random
import queue
import threading
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
        self.zobrist_key = 0

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
        b.zobrist_key = self.zobrist_key
        return b


ZOBRIST_PIECE = [[[0] * 64 for _ in range(6)] for _ in range(2)]
ZOBRIST_CASTLING = [0] * 16
ZOBRIST_EP = [0] * 64
ZOBRIST_SIDE = 0

FILE_MASKS = [0] * 8
PASSED_PAWN_MASK = [[0] * 64 for _ in range(2)]


def init_zobrist():
    global ZOBRIST_SIDE
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


def init_eval_tables():
    for f in range(8):
        mask = 0
        for r in range(8):
            mask |= bit(sq_index(f, r))
        FILE_MASKS[f] = mask
        if __debug__:
            assert popcount(FILE_MASKS[f]) == 8

    for color in (WHITE, BLACK):
        for sq in range(64):
            f = sq & 7
            rank = sq >> 3
            mask = 0
            for nf in (f - 1, f, f + 1):
                if nf < 0 or nf > 7:
                    continue
                if color == WHITE:
                    for r in range(rank + 1, 8):
                        mask |= bit(sq_index(nf, r))
                else:
                    for r in range(rank - 1, -1, -1):
                        mask |= bit(sq_index(nf, r))
            PASSED_PAWN_MASK[color][sq] = mask

            if __debug__:
                m = mask
                while m:
                    s, m = pop_lsb(m)
                    sr = s >> 3
                    if color == WHITE:
                        assert sr > rank
                    else:
                        assert sr < rank


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
                        if not king_in_check and not is_square_attacked(board, f1, opp) and not is_square_attacked(board, g1, opp):
                            moves.append(
                                Move(sq, g1, KING, flags=Move.CASTLE)
                            )
                if board.castling & 2:  # Queenside
                    d1 = fr_to_sq("d", "1")
                    c1 = fr_to_sq("c", "1")
                    if not (all_occ & (bit(d1) | bit(c1))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, d1, opp) and not is_square_attacked(board, c1, opp):
                            moves.append(
                                Move(sq, c1, KING, flags=Move.CASTLE)
                            )
            else:
                if board.castling & 4:  # Kingside
                    f8 = fr_to_sq("f", "8")
                    g8 = fr_to_sq("g", "8")
                    if not (all_occ & (bit(f8) | bit(g8))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, f8, opp) and not is_square_attacked(board, g8, opp):
                            moves.append(
                                Move(sq, g8, KING, flags=Move.CASTLE)
                            )
                if board.castling & 8:  # Queenside
                    d8 = fr_to_sq("d", "8")
                    c8 = fr_to_sq("c", "8")
                    if not (all_occ & (bit(d8) | bit(c8))):
                        # Cannot castle if in check or if squares are attacked
                        if not king_in_check and not is_square_attacked(board, d8, opp) and not is_square_attacked(board, c8, opp):
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


def is_square_attacked(board: Board, sq, by_color):
    if not in_bounds(sq):
        return False
    all_occ = board.all_occ

    # Pawns
    if by_color == WHITE:
        for df in (-1, 1):
            from_sq = sq - 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[WHITE][PAWN]:
                if (from_sq >> 3) + 1 == (sq >> 3):
                    return True
    else:
        for df in (-1, 1):
            from_sq = sq + 8 - df
            if in_bounds(from_sq) and bit(from_sq) & board.bb[BLACK][PAWN]:
                if (from_sq >> 3) - 1 == (sq >> 3):
                    return True

    # Knights
    for d in KNIGHT_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq):
            continue
        if abs((from_sq & 7) - (sq & 7)) > 2:
            continue
        if bit(from_sq) & board.bb[by_color][KNIGHT]:
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
            return True
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
    """True if there are no pawns on the given file (0..7)."""
    pawns = board.bb[WHITE][PAWN] | board.bb[BLACK][PAWN]
    return not (FILE_MASKS[file] & pawns)


def is_semi_open_file(board: Board, color, file):
    """True if file has no own pawns but has at least one enemy pawn."""
    own_pawns = board.bb[color][PAWN]
    enemy_pawns = board.bb[1 - color][PAWN]
    file_mask = FILE_MASKS[file]
    return not (file_mask & own_pawns) and bool(file_mask & enemy_pawns)


def is_isolated_pawn(pawns_bb, sq):
    """True if pawn on sq has no friendly pawns on adjacent files."""
    file = sq & 7
    adjacent_files = 0
    if file > 0:
        adjacent_files |= FILE_MASKS[file - 1]
    if file < 7:
        adjacent_files |= FILE_MASKS[file + 1]
    return not (pawns_bb & adjacent_files)


def doubled_pawn_extras_on_file(pawns_bb, file):
    """Number of extra pawns on a file (2 pawns => 1 extra, etc.)."""
    return max(0, popcount(pawns_bb & FILE_MASKS[file]) - 1)


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
    )

    def __init__(self, move, board: Board):
        self.move = move
        self.castling = board.castling
        self.ep_square = board.ep_square
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number
        self.zobrist_key = board.zobrist_key


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

    # Remove moving piece from from_sq
    board.bb[color][piece] ^= from_bb
    board.occ[color] ^= from_bb
    board.all_occ ^= from_bb
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

    # Castling move rook
    if move.flags & Move.CASTLE:
        if color == WHITE:
            if move.to_sq == fr_to_sq("g", "1"):
                # king side
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
        board.bb[color][ROOK] ^= rf_bb
        board.occ[color] ^= rf_bb
        board.all_occ ^= rf_bb
        key ^= ZOBRIST_PIECE[color][ROOK][rook_from]

        board.bb[color][ROOK] ^= rt_bb
        board.occ[color] ^= rt_bb
        board.all_occ ^= rt_bb
        key ^= ZOBRIST_PIECE[color][ROOK][rook_to]

    # Promotion
    if piece == PAWN and move.promo is not None:
        promo = move.promo
        board.bb[color][promo] ^= to_bb
        key ^= ZOBRIST_PIECE[color][promo][move.to_sq]
    else:
        board.bb[color][piece] ^= to_bb
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
    if board.hash_history:
        board.hash_history.pop()

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


def move_causes_repetition(board: Board, move: Move, threshold=3):
    """Check whether `move` reaches a position repeated `threshold` times."""
    if not make_move(board, move):
        return False
    repeated = board.hash_history.count(board.zobrist_key) >= threshold
    undo_move(board)
    return repeated


def move_allows_immediate_repetition(board: Board, move: Move):
    """Check whether opponent has an immediate legal reply that repeats a seen position."""
    if not make_move(board, move):
        return False

    allows_repetition = False
    for reply in gen_moves(board):
        if not make_move(board, reply):
            continue
        if board.hash_history.count(board.zobrist_key) >= 2:
            allows_repetition = True
            undo_move(board)
            break
        undo_move(board)

    undo_move(board)
    return allows_repetition


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

PIECE_VALUES_MG = [100, 320, 330, 500, 900, 0]
PIECE_VALUES_EG = [120, 300, 320, 500, 900, 0]


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


def _see_least_attacker(board_bb, occ, sq, by_color):
    # Pawns
    if by_color == WHITE:
        for from_sq in (sq - 9, sq - 7):
            if in_bounds(from_sq) and abs((from_sq & 7) - (sq & 7)) == 1:
                bb = bit(from_sq)
                if (occ & bb) and (board_bb[WHITE][PAWN] & bb):
                    return from_sq, PAWN
    else:
        for from_sq in (sq + 7, sq + 9):
            if in_bounds(from_sq) and abs((from_sq & 7) - (sq & 7)) == 1:
                bb = bit(from_sq)
                if (occ & bb) and (board_bb[BLACK][PAWN] & bb):
                    return from_sq, PAWN

    # Knights
    for d in KNIGHT_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq) or abs((from_sq & 7) - (sq & 7)) > 2:
            continue
        bb = bit(from_sq)
        if (occ & bb) and (board_bb[by_color][KNIGHT] & bb):
            return from_sq, KNIGHT

    # Bishops / Queens (diagonals)
    sq_f = sq & 7
    sq_r = sq >> 3
    for d in BISHOP_DELTAS:
        to = sq + d
        while in_bounds(to):
            to_f = to & 7
            to_r = to >> 3
            if abs(to_f - sq_f) != abs(to_r - sq_r):
                break
            bb = bit(to)
            if occ & bb:
                if board_bb[by_color][BISHOP] & bb:
                    return to, BISHOP
                if board_bb[by_color][QUEEN] & bb:
                    return to, QUEEN
                break
            to += d

    # Rooks / Queens (orthogonals)
    for d in ROOK_DELTAS:
        to = sq + d
        while in_bounds(to):
            to_f = to & 7
            to_r = to >> 3
            if d in (1, -1) and to_r != sq_r:
                break
            if d in (8, -8) and to_f != sq_f:
                break
            bb = bit(to)
            if occ & bb:
                if board_bb[by_color][ROOK] & bb:
                    return to, ROOK
                if board_bb[by_color][QUEEN] & bb:
                    return to, QUEEN
                break
            to += d

    # King (last)
    for d in KING_DELTAS:
        from_sq = sq + d
        if not in_bounds(from_sq) or abs((from_sq & 7) - (sq & 7)) > 1:
            continue
        bb = bit(from_sq)
        if (occ & bb) and (board_bb[by_color][KING] & bb):
            return from_sq, KING

    return None, None


def see(board: Board, move: Move):
    """Static Exchange Evaluation (cp) for captures on move.to_sq."""
    if not move.is_capture():
        return 0
    if move.promo is not None:
        return 0

    color = board.side_to_move
    opp = 1 - color
    to_sq = move.to_sq
    from_sq = move.from_sq

    mover = move.piece
    if mover is None:
        for p in range(6):
            if bit(from_sq) & board.bb[color][p]:
                mover = p
                break
    if mover is None:
        return 0

    captured = move.capture
    if move.flags & Move.EN_PASSANT:
        captured = PAWN
    if captured is None:
        return 0

    gains = [PIECE_VALUES_MG[captured]]

    bb = [board.bb[WHITE][:], board.bb[BLACK][:]]
    occ = board.all_occ

    from_bb = bit(from_sq)
    to_bb = bit(to_sq)

    bb[color][mover] &= ~from_bb
    occ &= ~from_bb

    if move.flags & Move.EN_PASSANT:
        cap_sq = to_sq - 8 if color == WHITE else to_sq + 8
        cap_bb = bit(cap_sq)
        bb[opp][PAWN] &= ~cap_bb
        occ &= ~cap_bb
    else:
        bb[opp][captured] &= ~to_bb
        occ &= ~to_bb

    bb[color][mover] |= to_bb
    occ |= to_bb

    attacker = mover
    side = opp
    d = 0

    while True:
        a_sq, a_pt = _see_least_attacker(bb, occ, to_sq, side)
        if a_sq is None:
            break
        d += 1
        gains.append(PIECE_VALUES_MG[attacker] - gains[d - 1])

        a_from_bb = bit(a_sq)
        bb[side][a_pt] &= ~a_from_bb
        occ &= ~a_from_bb

        bb[1 - side][attacker] &= ~to_bb
        bb[side][a_pt] |= to_bb
        occ |= to_bb

        attacker = a_pt
        side = 1 - side

    while d > 0:
        d -= 1
        gains[d] = -max(-gains[d], gains[d + 1])
    return gains[0]


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


def knight_mobility(board: Board, color, sq):
    """Count knight destination squares excluding own occupied squares."""
    own_occ = board.occ[color]
    moves = 0
    for d in KNIGHT_DELTAS:
        to = sq + d
        if not in_bounds(to):
            continue
        if abs((to & 7) - (sq & 7)) > 2:
            continue
        if bit(to) & own_occ:
            continue
        moves += 1
    return moves


def slider_mobility(board: Board, color, sq, deltas):
    """Count reachable ray squares until first blocker (enemy blocker square counts)."""
    own_occ = board.occ[color]
    all_occ = board.all_occ
    from_file = sq & 7
    from_rank = sq >> 3
    moves = 0

    for d in deltas:
        to = sq + d
        while in_bounds(to):
            to_file = to & 7
            to_rank = to >> 3
            if d in (1, -1) and to_rank != from_rank:
                break
            if d in (8, -8) and to_file != from_file:
                break
            if d in (9, 7, -7, -9):
                if abs(to_file - from_file) != abs(to_rank - from_rank):
                    break

            to_bb = bit(to)
            if to_bb & own_occ:
                break

            moves += 1
            if to_bb & all_occ:
                break
            to += d
    return moves


def king_shelter_penalty(board: Board, color, king_sq):
    """Penalty for missing own pawns on 3 shelter squares in front of king."""
    kf = king_sq & 7
    kr = king_sq >> 3
    front_rank = kr + 1 if color == WHITE else kr - 1
    if front_rank < 0 or front_rank > 7:
        return 0

    pawns = board.bb[color][PAWN]
    missing = 0
    for f in (kf - 1, kf, kf + 1):
        if f < 0 or f > 7:
            continue
        sq = sq_index(f, front_rank)
        if not (pawns & bit(sq)):
            missing += 1
    return 10 * missing


def eval_board(board: Board):
    # positive is good for side to move (we return from POV of side_to_move later)
    mg = 0
    eg = 0
    bishop_count = [0, 0]
    passed_pawns = [0, 0]

    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        for p in range(6):
            bb = board.bb[color][p]
            while bb:
                sq, bb = pop_lsb(bb)
                mirror_sq = sq ^ 56 if color == BLACK else sq
                if p == KING:
                    mg += sign * (PIECE_VALUES_MG[p] + MG_PST[KING][mirror_sq])
                    eg += sign * (PIECE_VALUES_EG[p] + EG_KING_PST[mirror_sq])
                else:
                    mg += sign * (PIECE_VALUES_MG[p] + MG_PST[p][mirror_sq])
                    eg += sign * (PIECE_VALUES_EG[p] + MG_PST[p][mirror_sq])
                if p == BISHOP:
                    bishop_count[color] += 1
                if p == PAWN:
                    opp_pawns = board.bb[1 - color][PAWN]
                    if not (opp_pawns & PASSED_PAWN_MASK[color][sq]):
                        passed_pawns[color] += 1

    # bishop pair
    score = mg
    if bishop_count[WHITE] >= 2:
        score += 30
    if bishop_count[BLACK] >= 2:
        score -= 30

    # mobility bonus (cheap, local): knight / bishop / rook / queen
    MOBILITY_WEIGHTS = {
        KNIGHT: 3,
        BISHOP: 4,
        ROOK: 2,
        QUEEN: 1,
    }
    ISOLATED_PAWN_PENALTY = 12
    DOUBLED_PAWN_PENALTY = 10
    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1

        bb = board.bb[color][KNIGHT]
        while bb:
            sq, bb = pop_lsb(bb)
            score += sign * MOBILITY_WEIGHTS[KNIGHT] * knight_mobility(board, color, sq)

        bb = board.bb[color][BISHOP]
        while bb:
            sq, bb = pop_lsb(bb)
            score += sign * MOBILITY_WEIGHTS[BISHOP] * slider_mobility(board, color, sq, BISHOP_DELTAS)

        bb = board.bb[color][ROOK]
        while bb:
            sq, bb = pop_lsb(bb)
            score += sign * MOBILITY_WEIGHTS[ROOK] * slider_mobility(board, color, sq, ROOK_DELTAS)
            file = sq & 7
            if is_open_file(board, file):
                score += sign * 20
            elif is_semi_open_file(board, color, file):
                score += sign * 10

        bb = board.bb[color][QUEEN]
        while bb:
            sq, bb = pop_lsb(bb)
            score += sign * MOBILITY_WEIGHTS[QUEEN] * slider_mobility(board, color, sq, BISHOP_DELTAS + ROOK_DELTAS)

        pawns_bb = board.bb[color][PAWN]
        pawns = pawns_bb
        while pawns:
            sq, pawns = pop_lsb(pawns)
            if is_isolated_pawn(pawns_bb, sq):
                score -= sign * ISOLATED_PAWN_PENALTY
        for file in range(8):
            extras = doubled_pawn_extras_on_file(pawns_bb, file)
            if extras:
                score -= sign * (DOUBLED_PAWN_PENALTY * extras)

    # passed pawns bonus
    score += 20 * passed_pawns[WHITE]
    score -= 20 * passed_pawns[BLACK]

    phase = game_phase(board)
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
    # Avoid unnecessary exposure: open file, centralization without endgame.
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
        score -= sign * king_shelter_penalty(board, color, ksq)

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
MAX_PLY = 128


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
        self.soft_stop_time = 0
        self.stop = False
        self.best_move = None
        self.history_heur = defaultdict(int)
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.countermove = {}
        self.age = 0
        self.max_depth = 0
        self.root_moves = []
        self.root_board = None
        self.root_eval = 0

    def clear(self):
        self.tt = [TTEntry() for _ in range(self.tt_size)]
        self.history_heur.clear()
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.countermove.clear()
        self.age = 0

    def probe_tt(self, key):
        entry = self.tt[key & (self.tt_size - 1)]
        if entry.key == key:
            return entry
        return None

    def store_tt(self, key, depth, score, flag, move):
        idx = key & (self.tt_size - 1)
        entry = self.tt[idx]
        if entry.key != key or depth >= entry.depth:
            self.tt[idx] = TTEntry(key, depth, score, flag, move, self.age)

    def time_up(self):
        return self.stop or (self.stop_time and time.time() >= self.stop_time)

    def search(self, board: Board, max_depth, time_limit=None, hard_time_limit=None):
        self.nodes = 0
        self.start_time = time.time()
        self.soft_stop_time = self.start_time + time_limit if time_limit else 0
        self.stop_time = self.start_time + hard_time_limit if hard_time_limit else self.soft_stop_time
        self.stop = False
        self.best_move = None
        self.root_board = board.clone()
        self.age += 1
        self.root_moves = gen_moves(board)
        self.max_depth = max_depth

        alpha = -INF
        beta = INF
        last_score = 0
        prev_score = 0
        prev_iter_best_move = None
        last_iter_time = 0.0
        avg_iter_time = 0.0
        iter_count = 0
        for depth in range(1, max_depth + 1):
            now = time.time()
            if last_iter_time > 0 and self.stop_time:
                predicted_next_iter = last_iter_time * 2.2
                if now + predicted_next_iter > self.stop_time:
                    break
            if self.time_up():
                break
            iter_start = time.time()
            self.age += 1
            # Adaptive aspiration window: wider at small depth, narrower at larger depth,
            # and widened when score is volatile between iterations.
            base_window = max(25, 90 - 4 * depth)
            volatility = abs(last_score - prev_score) if depth > 1 else 0
            window = max(base_window, volatility + 20)
            center = last_score if depth > 1 else 0
            alpha = max(-INF, center - window)
            beta = min(INF, center + window)

            max_expansions = 5
            expansions = 0
            while True:
                self.order_board = self.root_board.clone()
                score = self.negamax(board, depth, alpha, beta, 0, True)
                if self.time_up():
                    break
                if score <= alpha:
                    if expansions >= max_expansions:
                        alpha = -INF
                        beta = INF
                        self.order_board = self.root_board.clone()
                        score = self.negamax(board, depth, alpha, beta, 0, True)
                        break
                    alpha = max(-INF, alpha - window)
                    window = min(INF // 2, window * 2)
                    expansions += 1
                    continue
                if score >= beta:
                    if expansions >= max_expansions:
                        alpha = -INF
                        beta = INF
                        self.order_board = self.root_board.clone()
                        score = self.negamax(board, depth, alpha, beta, 0, True)
                        break
                    beta = min(INF, beta + window)
                    window = min(INF // 2, window * 2)
                    expansions += 1
                    continue
                break
            if self.time_up():
                break
            prev_score, last_score = last_score, score
            prev_iter_best_move = self.best_move

            iter_elapsed = max(0.0001, time.time() - iter_start)
            last_iter_time = iter_elapsed
            iter_count += 1
            avg_iter_time += (iter_elapsed - avg_iter_time) / iter_count

            if depth > 1 and self.stop_time:
                remaining = self.stop_time - time.time()
                if (prev_score - last_score) > 120 and remaining < 2.0 * avg_iter_time:
                    break
                if remaining < 1.5 * avg_iter_time:
                    break

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
        def is_legal_root_move(candidate):
            if candidate is None:
                return False
            return any(
                m.from_sq == candidate.from_sq
                and m.to_sq == candidate.to_sq
                and m.promo == candidate.promo
                for m in legal_root_moves
            )

        if not legal_root_moves:
            self.best_move = None
        elif not is_legal_root_move(self.best_move):
            # Fallback order: root TT move, previous iteration best, then first legal.
            root_tt_entry = self.probe_tt(board.zobrist_key)
            tt_root_move = root_tt_entry.move if root_tt_entry else None
            if is_legal_root_move(tt_root_move):
                self.best_move = tt_root_move
            elif is_legal_root_move(prev_iter_best_move):
                self.best_move = prev_iter_best_move
            else:
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

    def negamax(self, board: Board, depth, alpha, beta, ply, root=False, allow_null=True):
        if self.time_up():
            return 0
        self.nodes += 1

        # repetition and fifty-move are immediate draws
        if board.halfmove_clock >= 100:
            return 0
        if board.hash_history.count(board.zobrist_key) >= 3:
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        key = board.zobrist_key
        tt_entry = self.probe_tt(key)
        if tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score > MATE_VALUE - 1000:
                tt_score -= ply
            elif tt_score < -MATE_VALUE + 1000:
                tt_score += ply
            if tt_entry.flag == TTEntry.EXACT and not root:
                return tt_score
            elif tt_entry.flag == TTEntry.LOWER and tt_score > alpha:
                alpha = tt_score
            elif tt_entry.flag == TTEntry.UPPER and tt_score < beta:
                beta = tt_score
            if alpha >= beta and not root:
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
        prev = board.history[-1].move if board.history else None
        prev_key = None
        if prev is not None:
            prev_key = (board.side_to_move, prev.from_sq, prev.to_sq, prev.promo)
        cm = self.countermove.get(prev_key) if prev_key is not None else None
        ply_idx = ply if ply < MAX_PLY else (MAX_PLY - 1)
        killer0, killer1 = self.killers[ply_idx]

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

        phase = game_phase(board)
        order_board = getattr(self, "order_board", None)
        ROOT_WINNING_EVAL_THRESHOLD = 150
        ROOT_REPETITION_ORDERING_PENALTY = 500_000
        root_winning_now = root and eval_board(board) > ROOT_WINNING_EVAL_THRESHOLD

        def move_score(m):
            score = 0
            if tt_move and m.from_sq == tt_move.from_sq and m.to_sq == tt_move.to_sq and m.promo == tt_move.promo:
                score += 10_000_000
            if m.is_capture():
                victim = m.capture if m.capture is not None else PAWN
                attacker = m.piece
                if attacker is None:
                    for pt in range(6):
                        if bit(m.from_sq) & board.bb[board.side_to_move][pt]:
                            attacker = pt
                            break
                score += 1000 * (PIECE_VALUES_MG[victim] - (PIECE_VALUES_MG[attacker] if attacker is not None else 0) // 10)
                score += 16 * see(board, m)
            else:
                mt = (m.from_sq, m.to_sq, m.promo)
                if killer0 is not None and mt == killer0:
                    score += 900_000
                elif killer1 is not None and mt == killer1:
                    score += 800_000
                elif cm is not None and mt == cm:
                    score += 700_000
                score += self.history_heur[(board.side_to_move, m.from_sq, m.to_sq)]
            # Move filtering (ordering): avoid moves that leave pieces en prise when alternatives exist.
            # Only at root, when not in check; heuristic-based. Penalty tries "safe" moves first.
            # SEE-lite: demote captures that can be recaptured by a cheaper piece (bad captures).
            if root and order_board is not None and not in_check(board):
                if root_winning_now and move_causes_repetition(order_board, m, threshold=3):
                    score -= ROOT_REPETITION_ORDERING_PENALTY
                if make_move(order_board, m):
                    us = 1 - order_board.side_to_move
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
                board, depth - 1 - NULL_R, -beta, -beta + 1, ply + 1, root=False, allow_null=False
            )
            undo_null_move(board, saved)
            if self.time_up():
                return 0
            if null_score >= beta:
                return beta

        LMR_FULL_MOVES = 4   # Don't reduce first N moves (late moves only)
        LMR_REDUCTION = 2    # Reduce depth by this for LMR try
        first_move = True

        for i, m in enumerate(legal_moves):
            if not make_move(board, m):
                continue
            repetition_after_move = root and (board.hash_history.count(board.zobrist_key) >= 2)
            # Late Move Reduction: only for late quiet non-checking moves at sufficient depth.
            do_lmr = (
                depth >= 3
                and i >= LMR_FULL_MOVES
                and m.is_quiet()
                and not in_check(board)   # do not reduce checking moves
            )
            if do_lmr:
                score = -self.negamax(
                    board, depth - 1 - LMR_REDUCTION, -beta, -alpha, ply + 1, root=False, allow_null=True
                )
                if score > alpha:
                    # Re-search at full depth if reduced search improved alpha
                    if first_move:
                        score = -self.negamax(
                            board, depth - 1, -beta, -alpha, ply + 1, root=False, allow_null=True
                        )
                    else:
                        score = -self.negamax(
                            board, depth - 1, -(alpha + 1), -alpha, ply + 1, root=False, allow_null=True
                        )
                        if score > alpha:
                            score = -self.negamax(
                                board, depth - 1, -beta, -alpha, ply + 1, root=False, allow_null=True
                            )
            else:
                if first_move:
                    score = -self.negamax(
                        board, depth - 1, -beta, -alpha, ply + 1, root=False, allow_null=True
                    )
                else:
                    score = -self.negamax(
                        board, depth - 1, -(alpha + 1), -alpha, ply + 1, root=False, allow_null=True
                    )
                    if score > alpha:
                        score = -self.negamax(
                            board, depth - 1, -beta, -alpha, ply + 1, root=False, allow_null=True
                        )
            undo_move(board)
            first_move = False
            if self.time_up():
                return 0
            if repetition_after_move and score > 150:
                score = 0
            if root and score > 150 and move_allows_immediate_repetition(board, m):
                score = 0
            if score > best_score:
                best_score = score
                best_move = m
            if score > alpha:
                alpha = score
                if not m.is_capture():
                    self.history_heur[(board.side_to_move, m.from_sq, m.to_sq)] += depth * depth
            if alpha >= beta:
                if not m.is_capture():
                    mt = (m.from_sq, m.to_sq, m.promo)
                    if self.killers[ply_idx][0] != mt:
                        self.killers[ply_idx][1] = self.killers[ply_idx][0]
                        self.killers[ply_idx][0] = mt
                    if prev_key is not None:
                        self.countermove[prev_key] = mt
                break

        if best_move is None:
            return 0

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

    def quiescence(self, board: Board, alpha, beta, ply):
        SEE_QS_DELTA = 100
        if self.time_up():
            return 0
        self.nodes += 1
        stand_pat = eval_board(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = gen_moves(board, captures_only=True)
        if not moves:
            return stand_pat

        for m in moves:
            if not m.is_capture() and not (m.flags & Move.EN_PASSANT):
                continue
            if m.promo is None:
                victim = m.capture if m.capture is not None else PAWN
                if PIECE_VALUES_MG[victim] < PIECE_VALUES_MG[ROOK]:
                    if see(board, m) < -SEE_QS_DELTA:
                        continue
            if not make_move(board, m):
                continue
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
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
        self.cmd_queue = queue.Queue()
        self.quit_event = threading.Event()

    def _stdin_reader(self):
        while not self.quit_event.is_set():
            line = sys.stdin.readline()
            if not line:
                self.quit_event.set()
                self.searcher.stop = True
                break

            line = line.strip()
            if not line:
                continue

            if line == "stop":
                self.searcher.stop = True
                continue

            if line == "quit":
                self.quit_event.set()
                self.searcher.stop = True

            self.cmd_queue.put(line)

    def loop(self):
        input_thread = threading.Thread(target=self._stdin_reader, daemon=True)
        input_thread.start()

        while not self.quit_event.is_set():
            try:
                line = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
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
                self.quit_event.set()
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
            fen = " ".join(parts[idx + 1 : idx + 7])
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

    def dynamic_moves_to_go(self):
        pieces = popcount(self.board.all_occ)
        if pieces <= 6:
            return 12
        if pieces <= 10:
            return 18
        return 28

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
        hard_time_limit = None
        if movetime is not None:
            time_limit = movetime / 1000.0
        elif wtime is not None and btime is not None:
            my_time = wtime if self.board.side_to_move == WHITE else btime
            my_inc = (winc if self.board.side_to_move == WHITE else binc) or 0

            reserve = max(50, int(0.02 * my_time)) + 20
            remaining_safe = max(0, my_time - reserve)

            est_moves = movestogo if movestogo is not None else self.dynamic_moves_to_go()
            base = remaining_safe / max(1, est_moves)

            if my_time <= 15000:
                inc_factor = 0.9
            elif my_time >= 120000:
                inc_factor = 0.6
            else:
                inc_factor = 0.7

            soft_budget = base + inc_factor * my_inc
            soft_budget = max(20.0, min(soft_budget, 0.25 * remaining_safe if remaining_safe > 0 else 20.0))

            hard_budget = min(remaining_safe, soft_budget * 3.0)
            time_limit = soft_budget / 1000.0
            hard_time_limit = hard_budget / 1000.0

        self.searcher.stop = False
        best = self.searcher.search(self.board, depth, time_limit, hard_time_limit)
        if best is None:
            print("bestmove 0000", flush=True)
        else:
            print(f"bestmove {best.uci()}", flush=True)


def main():
    init_zobrist()
    init_eval_tables()
    uci = UCI()
    uci.loop()


if __name__ == "__main__":
    main()
