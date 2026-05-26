use crate::state::{Mark, State};
use crate::strategy::Strategy;
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn row_name_to_index(name: &str) -> usize {
    match name {
        "red" => 0,
        "yellow" => 1,
        "green" => 2,
        "blue" => 3,
        other => panic!("unknown row name from external bot: {other:?}"),
    }
}

#[derive(Serialize)]
struct RowView {
    marks: u8,
    next: Option<u8>,
}

#[derive(Serialize)]
struct PlayerView {
    red: RowView,
    yellow: RowView,
    green: RowView,
    blue: RowView,
    strikes: u8,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    phase: u8,
    active: usize,
    marked_phase1: bool,
    dice: [Option<u8>; 6],
    players: Vec<PlayerView>,
}

fn state_to_view(state: &State) -> PlayerView {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    PlayerView {
        red: RowView { marks: totals[0], next: frees[0] },
        yellow: RowView { marks: totals[1], next: frees[1] },
        green: RowView { marks: totals[2], next: frees[2] },
        blue: RowView { marks: totals[3], next: frees[3] },
        strikes: state.strikes,
    }
}

fn build_players(state: &State, opp_states: &[State]) -> Vec<PlayerView> {
    std::iter::once(state).chain(opp_states).map(state_to_view).collect()
}

fn dice_with_locks(dice: [u8; 6], state: &State) -> [Option<u8>; 6] {
    let locked = state.locked();
    core::array::from_fn(|i| {
        if i < 2 {
            Some(dice[i])
        } else if locked[i - 2] {
            None
        } else {
            Some(dice[i])
        }
    })
}

pub struct ExternalBot {
    name: String,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    _child: Child,
}

impl std::fmt::Debug for ExternalBot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExternalBot({})", self.name)
    }
}

impl Drop for ExternalBot {
    fn drop(&mut self) {
        let _ = self._child.kill();
        let _ = self._child.wait();
    }
}

impl ExternalBot {
    pub fn new(command: &str) -> Self {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let (program, args) = parts.split_first().expect("empty external bot command");

        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn external bot '{command}': {e}"));

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        Self { name: command.to_string(), stdin, stdout, _child: child }
    }

    fn query(&mut self, request: &Request, legal: &[Mark]) -> Option<Mark> {
        let json = serde_json::to_string(request).unwrap();
        writeln!(self.stdin, "{json}").unwrap();
        self.stdin.flush().unwrap();

        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)
            .unwrap_or_else(|e| panic!("failed to read from external bot '{}': {e}", self.name));
        if n == 0 {
            panic!("external bot '{}' closed stdout unexpectedly", self.name);
        }

        let value: serde_json::Value = serde_json::from_str(line.trim())
            .unwrap_or_else(|e| panic!("invalid JSON from '{}': {e}\nline: {line}", self.name));

        match value {
            serde_json::Value::Null => None,
            serde_json::Value::Object(ref obj) => {
                let color = obj.get("color")
                    .and_then(|v| v.as_str())
                    .expect("move must have a 'color' string");
                let number = obj.get("mark")
                    .and_then(|v| v.as_u64())
                    .expect("move must have a 'mark' integer") as u8;
                let mark = Mark { row: row_name_to_index(color), number };
                assert!(
                    legal.iter().any(|m| m.row == mark.row && m.number == mark.number),
                    "illegal move from '{}': {} {} (legal: {:?})",
                    self.name, color, number, legal,
                );
                Some(mark)
            }
            _ => panic!(
                "expected null or {{\"color\": ..., \"mark\": ...}} from '{}', got: {}",
                self.name, line.trim()
            ),
        }
    }
}

impl Strategy for ExternalBot {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let legal = state.generate_white_moves(dice[0] + dice[1]);
        let request = Request {
            phase: 1,
            active: 0,
            marked_phase1: false,
            dice: dice_with_locks(dice, state),
            players: build_players(state, opp_states),
        };
        self.query(&request, &legal)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let legal = state.generate_color_moves(dice);
        let request = Request {
            phase: 2,
            active: 0,
            marked_phase1: has_marked,
            dice: dice_with_locks(dice, state),
            players: build_players(state, opp_states),
        };
        self.query(&request, &legal)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        let legal = state.generate_white_moves(dice[0] + dice[1]);
        let request = Request {
            phase: 1,
            active: active_player + 1,
            marked_phase1: false,
            dice: dice_with_locks(dice, state),
            players: build_players(state, opp_states),
        };
        self.query(&request, &legal)
    }
}
