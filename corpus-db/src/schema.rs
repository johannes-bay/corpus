use anyhow::Result;
use rusqlite::Connection;

const MIGRATIONS: &[&str] = &[
    // Migration 0: properties table for enrichment data
    "CREATE TABLE IF NOT EXISTS properties (
        path      TEXT NOT NULL REFERENCES files(path),
        domain    TEXT NOT NULL,
        key       TEXT NOT NULL,
        value_num REAL,
        value_txt TEXT,
        PRIMARY KEY (path, domain, key)
    );
    CREATE INDEX IF NOT EXISTS idx_properties_domain_key ON properties(domain, key);
    CREATE INDEX IF NOT EXISTS idx_properties_domain_key_num ON properties(domain, key, value_num);",
    // Migration 1: embeddings table
    "CREATE TABLE IF NOT EXISTS embeddings (
        path      TEXT NOT NULL REFERENCES files(path),
        model     TEXT NOT NULL,
        vector    BLOB NOT NULL,
        dim       INTEGER NOT NULL,
        PRIMARY KEY (path, model)
    );",
    // Migration 2: compositions tables
    "CREATE TABLE IF NOT EXISTS compositions (
        id        TEXT PRIMARY KEY,
        name      TEXT NOT NULL,
        seed_path TEXT NOT NULL REFERENCES files(path),
        axes      TEXT NOT NULL,
        created   INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS composition_items (
        composition_id TEXT NOT NULL REFERENCES compositions(id),
        path           TEXT NOT NULL REFERENCES files(path),
        role           TEXT NOT NULL DEFAULT 'companion',
        score          REAL,
        axes_detail    TEXT,
        PRIMARY KEY (composition_id, path)
    );",
];

/// Tracks which migrations have been applied.
fn ensure_migration_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS corpus_migrations (
            id      INTEGER PRIMARY KEY,
            applied INTEGER NOT NULL
        );",
    )?;
    Ok(())
}

/// Run all pending migrations. Never touches the original files/hashes/meta tables.
pub fn migrate(conn: &Connection) -> Result<()> {
    ensure_migration_table(conn)?;

    for (i, sql) in MIGRATIONS.iter().enumerate() {
        let id = i as i64;
        let already_applied: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM corpus_migrations WHERE id = ?1",
            [id],
            |row| row.get(0),
        )?;

        if !already_applied {
            conn.execute_batch(sql)?;
            conn.execute(
                "INSERT INTO corpus_migrations (id, applied) VALUES (?1, unixepoch())",
                [id],
            )?;
            tracing::info!("Applied migration {id}");
        }
    }

    Ok(())
}
