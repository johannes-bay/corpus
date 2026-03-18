pub mod models;
pub mod queries;
pub mod schema;

use anyhow::Result;
use rusqlite::Connection;

/// Open an existing index database and run corpus migrations (additive only).
pub fn open_db(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    schema::migrate(&conn)?;
    Ok(conn)
}
