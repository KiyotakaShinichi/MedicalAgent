# Model Registry

The project uses a local model registry to store metadata, artifacts, and lifecycle state for PoC monitoring models.

## Capabilities
- Registry metadata for model name, version, task, artifacts, and training data hashes
- Promotion to champion or rollback to prior versions
- Audit logging for model lifecycle changes

Evidence: [backend/services/model_artifacts.py](backend/services/model_artifacts.py), [backend/models.py](backend/models.py)

## Notes
- Registry usage is for engineering practice only.
- Promotion and rollback do not imply clinical validation.
