# Feature Store

MedicalAgent includes a local, account-free feature-store manifest for training and serving consistency.

## What is materialized
- Offline feature CSV
- Feature schema and data types
- Missingness rates
- Source and artifact hashes

Evidence: [backend/services/feature_store.py](backend/services/feature_store.py)

## Command
```
python scripts/materialize_feature_store.py
```

## Non-diagnostic boundary
Feature-store artifacts support engineering practice and do not indicate clinical validation.
