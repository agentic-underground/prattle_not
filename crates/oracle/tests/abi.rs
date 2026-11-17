//! The oracle runtime loads behind Rust and answers its ABI. Needs `just oracle`; INTEGRATION,
//! because the runtime is built outside the repository from upstream's pinned source.

#[test]
#[ignore = "integration"]
fn the_oracle_runtime_loads_and_reports_its_abi() {
    let lib = oracle::Oracle::load_default().unwrap_or_else(|e| panic!("{e} — run `just oracle`"));
    let abi = lib.abi();
    assert!(abi > 0, "the oracle reported ABI {abi}");
    println!("oracle ABI {abi} from {}", lib.path().display());
}
