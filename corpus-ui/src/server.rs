use std::sync::{Arc, Mutex};

use axum::routing::{any, get, post};
use axum::Router;
use rusqlite::Connection;
use tower_http::cors::CorsLayer;

use crate::api;
use crate::ws;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Connection>>,
}

/// Build the Axum router with all API routes and static file serving.
pub fn build_router(conn: Connection) -> Router {
    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
    };

    Router::new()
        .route("/api/stats", get(api::stats))
        .route("/api/file", get(api::file_info))
        .route("/api/search", get(api::search))
        .route("/api/compose", post(api::compose))
        .route("/api/compose/live", any(ws::compose_live))
        .route("/api/file/preview", get(api::file_preview))
        .fallback(get(api::static_files))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the server on the given port, returning the bound address.
pub async fn run(conn: Connection, port: u16) -> anyhow::Result<()> {
    let app = build_router(conn);
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    tracing::info!("Corpus UI listening on http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
