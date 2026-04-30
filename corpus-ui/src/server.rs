use std::sync::{Arc, Mutex};

use axum::routing::{any, get, post, put};
use axum::Router;
use rusqlite::Connection;
use tower_http::cors::CorsLayer;

use crate::api;
use crate::compose_api::{self, CompositionStore};
use crate::ws;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Connection>>,
    pub compositions: CompositionStore,
}

/// Build the Axum router with all API routes and static file serving.
pub fn build_router(conn: Connection) -> Router {
    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
        compositions: CompositionStore::default(),
    };

    Router::new()
        .route("/api/stats", get(api::stats))
        .route("/api/file", get(api::file_info))
        .route("/api/search", get(api::search))
        .route("/api/compose", post(api::compose))
        .route("/api/concept", post(api::concept_search))
        .route("/api/compose/live", any(ws::compose_live))
        .route("/api/compose/visual", post(compose_api::compose_visual))
        .route("/api/compose/audio", post(compose_api::compose_audio))
        .route("/api/compose/{id}/preview", get(compose_api::compose_preview))
        .route("/api/compose/{id}/render", get(compose_api::compose_render))
        .route("/api/compose/{id}/adjust", put(compose_api::compose_adjust))
        .route("/api/file/preview", get(api::file_preview))
        .route("/graph", get(api::graph_page))
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
