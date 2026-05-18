# Local OncoTrack database

## Why it isn't in git
`medical_agent.db` is a SQLite file. Tracking it in git would:
- Bloat history every time a row changes (18 MB binary diffs)
- Leak whatever sample patient data the dev happened to be testing with
- Couple every dev's local state to whatever was committed last

It is listed in `.gitignore` and the previously-tracked copy has been
removed from the index.

## Create one from scratch
```bash
python scripts/bootstrap_db.py
```
Creates `./medical_agent.db` from the SQLAlchemy metadata and runs any
pending schema migrations. Idempotent — running it again does nothing if
the file already exists.

## Reset (wipe + recreate)
```bash
python scripts/bootstrap_db.py --reset
```
Refused on non-sqlite `DATABASE_URL` values so a Postgres prod DB cannot
be dropped by accident.

## Use a different location / driver
```bash
DATABASE_URL=sqlite:///./tmp/foo.db python scripts/bootstrap_db.py
DATABASE_URL=postgresql://... python scripts/bootstrap_db.py
```

## Where the schema lives
- SQLAlchemy models: `backend/models.py`
- Schema migrations: `backend/schema_migrations.py` + `backend/migrations/versions/`
- Bootstrap entry point: `scripts/bootstrap_db.py`
