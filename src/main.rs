mod bot;
mod game;
mod state;
mod strategy;

use game::Player;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let mut game = game::Game::new(vec![
        Player::new(
            Box::<strategy::Random>::default(),
            Box::new(SmallRng::from_entropy()),
        ),
        Player::new(
            Box::<strategy::Interactive>::default(),
            //Box::new(game::ManualDice),
            Box::new(SmallRng::from_entropy()),
        ),
    ]);

    game.play();
    game.print_game_over();
}
