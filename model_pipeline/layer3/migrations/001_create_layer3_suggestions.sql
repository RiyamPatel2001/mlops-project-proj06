-- STATUS: Already applied in production.
-- Table confirmed at: mlops.public.layer3_suggestions
-- Database: postgresql://mlops_user:mlops_pass@postgres.mlops.svc.cluster.local:5432/mlops
-- Inspected: 2026-04-23, table size 24 kB (rows already present)
-- DO NOT re-run this migration. Kept for documentation only.
--
-- To verify:
--   psql $POSTGRES_DSN -c "\d public.layer3_suggestions"

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
