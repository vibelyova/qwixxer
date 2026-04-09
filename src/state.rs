use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub strikes: u8,
    rows: [Row; 4],
}

#[derive(Debug, Clone, Copy, Default)]
struct Row {
    ascending: bool,
    total: u8,
    // None if locked, otherwise next free number
    free: Option<u8>,
}

impl Default for State {
    fn default() -> Self {
        let ascending = Row {
            ascending: true,
            total: 0,
            free: Some(2),
        };
        let descending = Row {
            ascending: false,
            total: 0,
            free: Some(12),
        };
        Self {
            strikes: 0,
            rows: [ascending, ascending, descending, descending],
        }
    }
}

impl State {
    pub fn apply_move(&mut self, mov: impl Into<Move>) {
        match mov.into() {
            Move::Strike => {
                assert!(self.strikes < 4);
                self.strikes += 1;
            }
            Move::Single(mark) => {
                self.rows[mark.row].mark(mark.number);
            }
            Move::Double(mark1, mark2) => {
                self.rows[mark1.row].mark(mark1.number);
                self.rows[mark2.row].mark(mark2.number);
            }
        }
    }

    pub fn count_locked(&self) -> u8 {
        self.rows.iter().map(|row| row.free.is_none() as u8).sum()
    }

    pub fn locked(&self) -> [bool; 4] {
        self.rows
            .iter()
            .map(|row| row.free.is_none())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn lock(&mut self, locked: [bool; 4]) {
        self.rows
            .iter_mut()
            .zip(locked)
            .filter(|(_, lock)| *lock)
            .for_each(|(row, _)| row.free = None);
    }

    pub fn count_points(&self) -> isize {
        self.rows
            .iter()
            .map(|row| (row.total * (row.total + 1) / 2) as isize)
            .sum::<isize>()
            - self.strikes as isize * 5
    }

    // For opponent's move you can use [on_white, 0, 0, 0, 0, 0].
    // This method ignores the `Strike` move.
    pub fn generate_moves(&self, dice: [u8; 6]) -> Vec<Move> {
        let (white, color) = dice.split_at(2);
        let on_white: u8 = white.iter().sum();
        let mut white_moves = Vec::new();
        let mut color_moves = Vec::new();
        let mut double_moves = Vec::new();

        // Generates possible moves using color die for `row` with index `i`.
        let on_color = |i: usize, row: Row| {
            white
                .iter()
                .filter(move |&w| row.can_mark(w + color[i]))
                .map(move |w| (i, w + color[i]))
        };

        for (i, &row) in self.rows.iter().enumerate() {
            if row.can_mark(on_white) {
                let white_mov = (i, on_white);
                white_moves.push(white_mov);

                let mut state_after = self.clone();
                state_after.apply_move(white_mov);

                on_color(i, state_after.rows[i])
                    .for_each(|color_mov| double_moves.push((white_mov, color_mov)));
            }
            on_color(i, row).for_each(|color_mov| color_moves.push(color_mov));
        }

        let double_moves = white_moves
            .clone()
            .into_iter()
            .cartesian_product(color_moves.clone())
            // here we only care about double moves on different rows,
            // since we already collected ones on the same row
            .filter(|(white, color)| white.0 != color.0)
            .chain(double_moves)
            .unique()
            .map(Move::from);

        white_moves
            .into_iter()
            .chain(color_moves)
            .unique()
            .map(Move::from)
            .chain(double_moves)
            .collect()
    }

    //////////////////////////////////////
    // Metrics ///////////////////////////
    //////////////////////////////////////

    pub fn blanks(&self) -> u8 {
        self.rows
            .iter()
            .map(|row| {
                let Some(free) = row.free else {
                    // if the row is locked, we don't care
                    return 0;
                };
                if row.ascending {
                    (free - 2) - row.total
                } else {
                    (12 - free) - row.total
                }
            })
            .sum()
    }

    // Probability that opponent's dice will have a number I can mark with no blanks.
    pub fn probability(&self) -> f32 {
        self.rows
            .iter()
            .flat_map(|row| row.free)
            .unique()
            .map(|x| 6 - 7usize.abs_diff(x as usize))
            .sum::<usize>() as f32
            / 36.0
    }

    // Probability that at least one of N opponents will roll a number I can mark with no blanks.
    pub fn probability_n(&self, n: u8) -> f32 {
        let p = self.probability();
        1.0 - (1.0 - p).powi(n as i32)
    }
}

impl Row {
    fn can_mark(&self, number: u8) -> bool {
        if !(2..=12).contains(&number) {
            return false;
        }
        let Some(free) = self.free else {
            return false;
        };
        if self.ascending && number == 12 && self.total < 5 {
            return false;
        }
        if !self.ascending && number == 2 && self.total < 5 {
            return false;
        }
        self.ascending && free <= number || !self.ascending && free >= number
    }

    fn mark(&mut self, number: u8) {
        assert!(self.can_mark(number));
        self.total += 1;
        self.free = if self.ascending && number == 12 || !self.ascending && number == 2 {
            self.total += 1;
            None
        } else if self.ascending {
            Some(number + 1)
        } else {
            Some(number - 1)
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mark {
    pub row: usize,
    pub number: u8,
}

impl std::str::FromStr for Mark {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (row, number) = s.split_at(1);
        anyhow::ensure!(row.len() == 1, "Invalid row format");
        let row = match row.chars().next().unwrap() {
            'r' | 'R' => 0,
            'y' | 'Y' => 1,
            'g' | 'G' => 2,
            'b' | 'B' => 3,
            _ => anyhow::bail!("Invalid row format"),
        };
        let number = number.parse()?;
        Ok(Mark { row, number })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Move {
    Strike,
    Single(Mark),
    Double(Mark, Mark),
}

impl From<(usize, u8)> for Move {
    fn from((row, number): (usize, u8)) -> Self {
        Move::Single(Mark { row, number })
    }
}

impl From<((usize, u8), (usize, u8))> for Move {
    fn from((first, second): ((usize, u8), (usize, u8))) -> Self {
        Move::Double(
            Mark {
                row: first.0,
                number: first.1,
            },
            Mark {
                row: second.0,
                number: second.1,
            },
        )
    }
}

impl std::str::FromStr for Move {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split_whitespace().collect();
        anyhow::ensure!(parts.len() == 1 || parts.len() == 2, "Invalid move format");

        match parts[0] {
            "strike" => Ok(Move::Strike),
            _ => {
                let mark: Mark = parts[0].parse()?;
                if parts.len() == 1 {
                    Ok(Move::Single(mark))
                } else {
                    let mark2: Mark = parts[1].parse()?;
                    Ok(Move::Double(mark, mark2))
                }
            }
        }
    }
}
