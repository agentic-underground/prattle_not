//! `speedup` — `prattle_core::speedup`, which says what it measures.

fn main() {
    prattle_core::speedup::run(&std::env::args().collect::<Vec<_>>());
}
