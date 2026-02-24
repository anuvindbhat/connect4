use connect4::ai::find_best_move_detailed;
use connect4::config::HeuristicWeights;
use connect4::game::{BoardGeometry, BoardState};
use connect4::tt::TranspositionTable;
use connect4::types::Player;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::cell::RefCell;
use std::hint::black_box;
use std::time::Duration;

fn bench_search(c: &mut Criterion) {
    let geo = BoardGeometry::<u64>::new(7, 6);
    let weights = HeuristicWeights::default();
    let player = Player::Red;
    let depth = 7;
    let tt = RefCell::new(TranspositionTable::<u64>::new(64));

    // 1. Empty Board
    let state_empty = BoardState::<u64>::default();

    // 2. Mid-Game Board (No win)
    let mut state_mid = BoardState::<u64>::default();
    // Alternating columns to avoid wins
    let moves: [u32; 14] = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6];
    let mut curr_p = Player::Red;
    for &m in &moves {
        state_mid = state_mid.drop_piece(m, curr_p, &geo).unwrap();
        curr_p = curr_p.other();
    }

    let mut no_tt_group = c.benchmark_group("find_best_move_no_tt");
    no_tt_group.warm_up_time(Duration::from_secs(5));
    no_tt_group.measurement_time(Duration::from_secs(30));

    no_tt_group.bench_function("empty_board", |b| {
        b.iter(|| {
            black_box(find_best_move_detailed(
                black_box(&state_empty),
                black_box(&geo),
                black_box(player),
                black_box(depth),
                black_box(false),
                black_box(&weights),
                black_box(None),
            ))
        });
    });

    no_tt_group.bench_function("mid_game_board", |b| {
        b.iter(|| {
            black_box(find_best_move_detailed(
                black_box(&state_mid),
                black_box(&geo),
                black_box(player),
                black_box(depth),
                black_box(false),
                black_box(&weights),
                black_box(None),
            ))
        });
    });

    no_tt_group.finish();

    let mut tt_group = c.benchmark_group("find_best_move_tt");
    tt_group.warm_up_time(Duration::from_secs(5));
    tt_group.measurement_time(Duration::from_secs(30));

    tt_group.bench_function("empty_board", |b| {
        b.iter_batched_ref(
            || {
                let mut t = tt.borrow_mut();
                t.reset();
                t
            },
            |tt| {
                black_box(find_best_move_detailed(
                    black_box(&state_empty),
                    black_box(&geo),
                    black_box(player),
                    black_box(depth),
                    black_box(false),
                    black_box(&weights),
                    black_box(Some(tt)),
                ))
            },
            BatchSize::PerIteration,
        );
    });
    tt_group.bench_function("mid_game_board", |b| {
        b.iter_batched_ref(
            || {
                let mut t = tt.borrow_mut();
                t.reset();
                t
            },
            |tt| {
                black_box(find_best_move_detailed(
                    black_box(&state_mid),
                    black_box(&geo),
                    black_box(player),
                    black_box(depth),
                    black_box(false),
                    black_box(&weights),
                    black_box(Some(tt)),
                ))
            },
            BatchSize::PerIteration,
        );
    });

    tt_group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
