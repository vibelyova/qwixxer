export interface GameView {
    player: StateView;
    bot: StateView;
    dice: number[];
    white_sum: number;
    phase: string;
    available_moves: MoveView[];
    game_over: boolean;
    message: string;
}

export interface StateView {
    rows: RowView[];
    strikes: number;
    score: number;
}

export interface RowView {
    numbers: number[];
    free: number | null;
    total: number;
    locked: boolean;
    ascending: boolean;
    color: string;
    marks: boolean[];
}

export interface MoveView {
    index: number;
    description: string;
    is_strike: boolean;
    is_pass: boolean;
}

/** A parsed target from a move description, e.g. { row: 0, number: 7 } */
export interface MoveTarget {
    row: number;
    number: number;
}

/** Parsed move: single target, double target, strike, or pass */
export interface ParsedMove {
    moveIndex: number;
    targets: MoveTarget[];
    isStrike: boolean;
    isPass: boolean;
}

/** Current selection state for the click-to-select interaction */
export interface SelectionState {
    /** Squares selected so far: [{ row, number }] */
    selected: MoveTarget[];
    /** Move indices that are still compatible with the current selection */
    compatibleMoves: number[];
    /** Phase: player_active or player_passive */
    phase: string;
}
