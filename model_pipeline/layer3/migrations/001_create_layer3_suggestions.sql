CREATE TABLE IF NOT EXISTS layer3_suggestions (
    id                      SERIAL PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    cluster_id              TEXT NOT NULL UNIQUE,
    suggested_category_name TEXT NOT NULL,
    payee_list              TEXT[] NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_layer3_user_status
    ON layer3_suggestions (user_id, status);
