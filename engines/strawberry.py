#!/usr/bin/env python3
import chess
import chess.polyglot
import sys
import time

INF = 100000
TIME_MARGIN = 0.03
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# -------------------- PST --------------------
PAWN_PST = [
     0,0,0,0,0,0,0,0,
     5,10,10,-20,-20,10,10,5,
     5,-5,-10,0,0,-10,-5,5,
     0,0,0,20,20,0,0,0,
     5,5,10,25,25,10,5,5,
    10,10,20,30,30,20,10,10,
    50,50,50,50,50,50,50,50,
     0,0,0,0,0,0,0,0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,0,0,0,0,-20,-40,
    -30,0,10,15,15,10,0,-30,
    -30,5,15,20,20,15,5,-30,
    -30,0,15,20,20,15,0,-30,
    -30,5,10,15,15,10,5,-30,
    -40,-20,0,5,5,0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,0,0,0,0,0,0,-10,
    -10,0,5,10,10,5,0,-10,
    -10,5,5,10,10,5,5,-10,
    -10,0,10,10,10,10,0,-10,
    -10,10,10,10,10,10,10,-10,
    -10,5,0,0,0,0,5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
     0,0,0,5,5,0,0,0,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
     5,10,10,10,10,10,10,5,
     0,0,0,0,0,0,0,0
]

QUEEN_PST = [
    -20,-10,-10,-5,-5,-10,-10,-20,
    -10,0,0,0,0,0,0,-10,
    -10,0,5,5,5,5,0,-10,
     -5,0,5,5,5,5,0,-5,
      0,0,5,5,5,5,0,-5,
    -10,5,5,5,5,5,0,-10,
    -10,0,5,0,0,0,0,-10,
    -20,-10,-10,-5,-5,-10,-10,-20
]

KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20,20,0,0,0,0,20,20,
     20,30,10,0,0,10,30,20
]

KING_ENDGAME_PST = [
     0,5,10,15,15,10,5,0,
     5,10,15,20,20,15,10,5,
    10,15,20,25,25,20,15,10,
    15,20,25,30,30,25,20,15,
    15,20,25,30,30,25,20,15,
    10,15,20,25,25,20,15,10,
     5,10,15,20,20,15,10,5,
     0,5,10,15,15,10,5,0
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
DEVELOPMENT_BONUS = {chess.KNIGHT: 20, chess.BISHOP: 15}

board = chess.Board()
TT = {}
start_time = 0
time_limit = 0
REPEAT_COUNTS = {}

# ---------------- Evaluation ----------------
def material_score(board):
    return sum((PIECE_VALUES[p.piece_type] if p.color else -PIECE_VALUES[p.piece_type]) for p in board.piece_map().values())

def evaluate(board):
    if board.is_checkmate():
        return -INF + 1
    score = 0
    pieces = board.piece_map()
    endgame = len(pieces) <= 6

    fen = board.board_fen() + (" w" if board.turn else " b")
    mat = material_score(board)
    rep = REPEAT_COUNTS.get(fen, 0)
    if mat > 150 and rep > 0:
        rep_penalty = -120 * rep
    else:
        rep_penalty = 0


    for sq, piece in pieces.items():
        idx = sq if piece.color else 63 - sq
        if piece.piece_type == chess.KING:
            pst = KING_ENDGAME_PST[idx] if endgame else PST[chess.KING][idx]
        else:
            pst = PST[piece.piece_type][idx]

        val = PIECE_VALUES[piece.piece_type] + pst

        # Анти-зевки
        attackers = board.attackers(not piece.color, sq)
        defenders = board.attackers(piece.color, sq)
        if attackers and not defenders and PIECE_VALUES[piece.piece_type]>=320:
            val -= 50

        # Центр и развитие для не-короля
        if piece.piece_type != chess.KING:
            if sq in CENTER_SQUARES:
                val += 20
            if piece.piece_type in DEVELOPMENT_BONUS and not board.has_castling_rights(piece.color):
                val += DEVELOPMENT_BONUS[piece.piece_type]

        # Король раннего дебюта: не лезем на 2й ряд
        if piece.piece_type==chess.KING and not board.has_castling_rights(piece.color):
            rank = chess.square_rank(sq)
            if (piece.color and rank<2) or (not piece.color and rank>5):
                val -= 200  # штраф за раннее движение короля

        score += val if piece.color else -val

    if mat > 200:
        score += 5 * (
            len(board.pieces(chess.QUEEN, chess.BLACK)) +
            len(board.pieces(chess.ROOK, chess.BLACK))
        )

    if mat < -200:
        score -= 5 * (
            len(board.pieces(chess.QUEEN, chess.WHITE)) +
            len(board.pieces(chess.ROOK, chess.WHITE))
        )

    if abs(mat) < 300:
        score += rep_penalty


    return score if board.turn else -score

# ---------------- MVV-LVA ----------------
def mvv_lva(board, move):
    if not board.is_capture(move): return 0
    v = board.piece_at(move.to_square)
    a = board.piece_at(move.from_square)
    if v and a: return 10*PIECE_VALUES[v.piece_type] - PIECE_VALUES[a.piece_type]
    return 0

# ---------------- Quiescence ----------------
def quiescence(board, alpha, beta):
    stand = evaluate(board)
    if stand >= beta: return beta
    if stand > alpha: alpha = stand
    for move in board.legal_moves:
        gives_check = board.gives_check(move)
        if board.is_capture(move) or (gives_check and evaluate(board) < 120):
            board.push(move)
            score = -quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
    return alpha

# ---------------- Negamax ----------------
def negamax(board, depth, alpha, beta):
    if time.time()-start_time > time_limit - TIME_MARGIN: raise TimeoutError
    key=(chess.polyglot.zobrist_hash(board), depth)
    if key in TT: return TT[key]

    if depth==0 or board.is_game_over(): return quiescence(board, alpha, beta), None

    fen=board.board_fen() + (" w" if board.turn else " b")
    REPEAT_COUNTS[fen] = REPEAT_COUNTS.get(fen,0)+1

    moves=sorted(board.legal_moves, key=lambda m: mvv_lva(board,m)+ (10 if m.to_square in CENTER_SQUARES and board.piece_at(m.from_square).piece_type!=chess.KING else 0), reverse=True)
    best_move=None

    for move in moves:
        board.push(move)
        score,_ = negamax(board, depth-1, -beta, -alpha)
        score = -score
        board.pop()
        if score>alpha:
            alpha = score
            best_move = move
            if alpha>=beta: break

    REPEAT_COUNTS[fen]-=1
    TT[key]=(alpha,best_move)
    return alpha,best_move

def fallback_move(board):
    for m in board.legal_moves:
        if board.is_castling(m): return m
    return next(iter(board.legal_moves))

# ---------------- UCI Loop ----------------
def uci_loop():
    global board,start_time,time_limit,REPEAT_COUNTS
    while True:
        line = sys.stdin.readline()
        if not line: return
        line=line.strip()
        if not line: continue

        if line=="uci":
            print("id name StrawberryChess v3.1")
            print("id author MK")
            print("uciok")
            sys.stdout.flush()
        elif line=="isready":
            print("readyok")
            sys.stdout.flush()
        elif line.startswith("position"):
            parts=line.split()
            if "startpos" in parts:
                board=chess.Board()
                idx=parts.index("startpos")+1
            else:
                board=chess.Board(" ".join(parts[1:7]))
                idx=7
            if idx<len(parts) and parts[idx]=="moves":
                for m in parts[idx+1:]: board.push(chess.Move.from_uci(m))
        elif line.startswith("go"):
            parts=line.split()
            wtime=btime=winc=binc=None
            movetime=None
            for i,p in enumerate(parts):
                if p=="movetime": movetime=int(parts[i+1])/1000
                elif p=="wtime": wtime=int(parts[i+1])/1000
                elif p=="btime": btime=int(parts[i+1])/1000
                elif p=="winc": winc=int(parts[i+1])/1000
                elif p=="binc": binc=int(parts[i+1])/1000
            if movetime is None:
                remaining = wtime if board.turn else btime
                inc = winc if board.turn else binc
                movetime=max(0.05,min(remaining*0.03+(inc or 0)*0.8,1.0))

            if movetime<0.15: max_depth=2
            elif movetime<0.3: max_depth=3
            elif movetime<0.6: max_depth=4
            else: max_depth=5

            start_time=time.time()
            time_limit=movetime
            TT.clear()
            REPEAT_COUNTS.clear()
            best=fallback_move(board)
            for d in range(1,max_depth+1):
                try:
                    _,move=negamax(board,d,-INF,INF)
                    if move: best=move
                except TimeoutError: break
            eval_score=evaluate(board)
            print(f"info score cp {eval_score}")
            print(f"bestmove {best.uci()}")
            sys.stdout.flush()
        elif line=="quit": return

if __name__=="__main__":
    uci_loop()
