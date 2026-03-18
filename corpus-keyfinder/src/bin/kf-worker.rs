/// Subprocess worker for key detection.
/// Usage: kf-worker <path> <max_seconds>
/// Outputs: key_name\tcof_position  (or empty line on failure)
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        std::process::exit(1);
    }

    let path = &args[1];
    let max_seconds: f64 = args[2].parse().unwrap_or(60.0);

    match corpus_keyfinder::detect_key(path, max_seconds) {
        Ok(Some(kr)) => {
            println!("{}\t{}", kr.name, kr.cof_position);
        }
        Ok(None) => {
            println!();
        }
        Err(_) => {
            println!();
        }
    }
}
