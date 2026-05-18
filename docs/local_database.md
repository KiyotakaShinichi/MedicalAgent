# Local database hygiene

`medical_agent.db` is a local SQLite demo database. It should not be committed.
The file is ignored by `.gitignore`; use migrations and seed scripts to rebuild
it when needed.

## Reset the local demo DB

```bash
python scripts/reset_local_db.py
```

This removes `medical_agent.db`, applies Alembic migrations, then runs
`seed_db.py` to restore the demo patient.

PowerShell with an explicit URL:

```powershell
$env:DATABASE_URL='sqlite:///./medical_agent.db'
python scripts/reset_local_db.py
```

Schema only:

```bash
python scripts/reset_local_db.py --no-seed
```

The reset script refuses non-SQLite URLs so it cannot wipe a shared database by
accident. For Postgres or other shared environments, use normal Alembic
migrations instead:

```bash
alembic upgrade head
```
