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
}

export interface MoveView {
    index: number;
    description: string;
    is_strike: boolean;
    is_pass: boolean;
}
