//! prattle-server — serves the page and the model weights, and says where to click.
//!
//!   --port N          (default 8192; 0 picks a free one, which is what the story harness uses)
//!   PRATTLE_SITE      the built page        (default target/site)
//!   PRATTLE_MODELS    local model folders   (default models), served at /models/
//!   PRATTLE_ORACLE    a shell command that runs scripts/oracle_worker.py; when set, /chat serves
//!                     LatentPort's own pair through it (`just chat` sets it)
//!   PRATTLE_TLS       a folder holding server.chain.crt and server.key; when set, it serves HTTPS
//!   PRATTLE_COMPARE   a FAST_LLM file; when set, /compare runs it beside the three files it was packed
//!                     from (found under PRATTLE_MODELS), in Prattle's own Rust

mod compare;
mod oracle;

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::path::PathBuf;

use axum::Router;
use axum::http::{HeaderValue, header};
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

/// The address other machines on the network reach this one by. Connecting a UDP socket sends no
/// packet; it only asks the kernel which interface would route outward.
fn network_ip() -> IpAddr {
    UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| s.connect("192.0.2.1:9").map(|_| s))
        .and_then(|s| s.local_addr())
        .map(|a| a.ip())
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

fn port() -> u16 {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--port") {
        Some(i) => args.get(i + 1).and_then(|p| p.parse().ok()).expect("--port takes a number"),
        None => 8192,
    }
}

#[tokio::main]
async fn main() {
    // two TLS backends are in the build (the hub client's, which a THIN FAST_LLM uses while /compare
    // loads, and this server's), so rustls cannot pick one by itself: this process uses aws-lc-rs,
    // installed before anything opens a connection. A second install is harmless and reports Err.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let site = std::env::var("PRATTLE_SITE").unwrap_or_else(|_| "target/site".into());
    let models = std::env::var("PRATTLE_MODELS").unwrap_or_else(|_| "models".into());
    if !std::path::Path::new(&site).join("index.html").exists() {
        eprintln!("⛔ no page at {site}/index.html — run `just web` first");
        std::process::exit(1);
    }

    let mut app = Router::new();
    if let Ok(command) = std::env::var("PRATTLE_ORACLE") {
        app = app.merge(oracle::routes(oracle::Worker::spawn(&command)));
    }
    if let Ok(file) = std::env::var("PRATTLE_COMPARE") {
        app = app.merge(compare::routes(compare::Compare::spawn(file.into(), models.clone().into())));
    }
    let app = app
        .nest_service("/models", ServeDir::new(&models))
        .fallback_service(ServeDir::new(&site))
        .layer(SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static("no-store")));

    let listener = std::net::TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port()))).unwrap_or_else(|e| {
        eprintln!("⛔ cannot bind 0.0.0.0:{} — {e}", port());
        std::process::exit(1);
    });
    let bound = listener.local_addr().unwrap().port();
    // PRATTLE_TLS names a folder holding server.chain.crt and server.key: one server, over HTTPS — what a
    // browser elsewhere on the network needs for WebGPU (a secure context).
    let tls = std::env::var("PRATTLE_TLS").ok().map(|d| (PathBuf::from(&d).join("server.chain.crt"), PathBuf::from(&d).join("server.key")));
    let scheme = if tls.is_some() { "https" } else { "http" };
    println!("prattle-server listening on 0.0.0.0:{bound}");
    println!("  open  {scheme}://{}:{bound}/", network_ip());
    if std::env::var("PRATTLE_COMPARE").is_ok() {
        println!("  compare  {scheme}://{}:{bound}/compare   (a FAST_LLM beside its three files; loading — the page says when ready)", network_ip());
    }
    if std::env::var("PRATTLE_ORACLE").is_ok() {
        println!("  chat  {scheme}://{}:{bound}/chat   (LatentPort's 4B → 27B; loading — the page says when ready)", network_ip());
    }

    serve(app, listener, tls).await;
}

/// Serve `app` on `listener` until Ctrl-C — over HTTPS when `tls` names a certificate chain and its key
/// (PEM files), plain HTTP otherwise.
async fn serve(app: Router, listener: std::net::TcpListener, tls: Option<(PathBuf, PathBuf)>) {
    listener.set_nonblocking(true).unwrap();
    if let Some((chain, key)) = tls {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default(); // see main
        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&chain, &key).await.unwrap_or_else(|e| {
            eprintln!("⛔ cannot read the certificate {} and key {} — {e}", chain.display(), key.display());
            std::process::exit(1);
        });
        let handle = axum_server::Handle::new();
        let stop = handle.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            stop.graceful_shutdown(None);
        });
        axum_server::from_tcp_rustls(listener, config).handle(handle).serve(app.into_make_service()).await.unwrap();
        return;
    }
    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })
        .await
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A route served with a certificate answers over HTTPS to a client that trusts only that certificate.
    #[tokio::test]
    async fn the_server_speaks_https_with_the_certificate_it_is_given() {
        let dir = std::env::temp_dir().join(format!("prattle-tls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let (chain, key) = (dir.join("server.chain.crt"), dir.join("server.key"));
        std::fs::write(&chain, cert.cert.pem()).unwrap();
        std::fs::write(&key, cert.key_pair.serialize_pem()).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route("/", axum::routing::get(|| async { "prattle over tls" }));
        tokio::spawn(serve(app, listener, Some((chain.clone(), key))));
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let out = tokio::process::Command::new("curl")
            .args(["-s", "--max-time", "5", "--cacert"])
            .arg(&chain)
            .arg(format!("https://localhost:{port}/"))
            .output()
            .await
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), "prattle over tls", "no HTTPS answer (curl exit {:?})", out.status.code());
    }
}
