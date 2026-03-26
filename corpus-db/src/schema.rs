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
    // Migration 3: segments, segment embeddings, and segment properties
    "CREATE TABLE IF NOT EXISTS segments (
        id           TEXT PRIMARY KEY,
        path         TEXT NOT NULL REFERENCES files(path),
        segment_type TEXT NOT NULL,
        segment_key  TEXT NOT NULL,
        label        TEXT,
        bbox_x       REAL,
        bbox_y       REAL,
        bbox_w       REAL,
        bbox_h       REAL,
        time_start   REAL,
        time_end     REAL,
        confidence   REAL,
        area_frac    REAL,
        model        TEXT,
        mask_rle     BLOB,
        UNIQUE(path, segment_type, segment_key)
    );
    CREATE INDEX IF NOT EXISTS idx_segments_path ON segments(path);
    CREATE INDEX IF NOT EXISTS idx_segments_type ON segments(segment_type);
    CREATE INDEX IF NOT EXISTS idx_segments_path_type ON segments(path, segment_type);
    CREATE INDEX IF NOT EXISTS idx_segments_label ON segments(label);

    CREATE TABLE IF NOT EXISTS segment_embeddings (
        segment_id TEXT NOT NULL REFERENCES segments(id),
        model      TEXT NOT NULL,
        vector     BLOB NOT NULL,
        dim        INTEGER NOT NULL,
        PRIMARY KEY (segment_id, model)
    );
    CREATE INDEX IF NOT EXISTS idx_segemb_model ON segment_embeddings(model);",
    // Migration 4: Full-text search index and embedding neighbor graph
    "CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
        path UNINDEXED,
        source_type UNINDEXED,
        source_key UNINDEXED,
        content,
        tokenize='porter unicode61'
    );

    CREATE TABLE IF NOT EXISTS neighbors (
        path_a     TEXT NOT NULL,
        path_b     TEXT NOT NULL,
        model      TEXT NOT NULL,
        similarity REAL NOT NULL,
        PRIMARY KEY (path_a, model, path_b)
    );
    CREATE INDEX IF NOT EXISTS idx_neighbors_b ON neighbors(path_b, model);

    CREATE TABLE IF NOT EXISTS projects (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        project_root TEXT NOT NULL UNIQUE,
        file_count   INTEGER NOT NULL DEFAULT 0,
        date_range   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);

    CREATE TABLE IF NOT EXISTS file_projects (
        path       TEXT NOT NULL REFERENCES files(path),
        project_id TEXT NOT NULL REFERENCES projects(id),
        PRIMARY KEY (path)
    );
    CREATE INDEX IF NOT EXISTS idx_file_projects_proj ON file_projects(project_id);",
    // Migration 5: pipeline registry
    "CREATE TABLE IF NOT EXISTS pipelines (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT,
        script       TEXT NOT NULL,
        model        TEXT,
        model_version TEXT,
        input_query  TEXT,
        output_table TEXT NOT NULL,
        params       TEXT,
        files_total  INTEGER NOT NULL DEFAULT 0,
        files_done   INTEGER NOT NULL DEFAULT 0,
        files_failed INTEGER NOT NULL DEFAULT 0,
        started      INTEGER,
        completed    INTEGER,
        status       TEXT NOT NULL DEFAULT 'pending',
        UNIQUE(name, model)
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
