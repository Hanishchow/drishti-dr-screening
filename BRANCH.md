# `backend` branch

Server, models, training and simulation. **No dashboard.**

```
core/      quality gate, lesion segmentation, features, rule grader, triage
dr/        datasets, splits, transforms, CORAL model, training, metrics
server/    API: auth, records, review queue, batched inference, audit, sync
sim/       district telemedicine simulation
deploy/    Docker for district and PHC edge
```

## Running

```bash
uvicorn server.app:app --port 8000
```

`web/` is absent here, so `GET /` returns a JSON service descriptor instead of
the dashboard and no static files are mounted. `server/app.py` already guards
for this, so nothing breaks — the interactive docs at `/docs` are how you
exercise the API on this branch.

Allow the separately-hosted dashboard to call this API:

```bash
DRISHTI_CORS_ORIGINS=https://your-frontend-host uvicorn server.app:app --port 8000
```

## Branch layout

| Branch | Contains |
|---|---|
| `main` | Everything, integrated. The deployable trunk. |
| `backend` | This branch. API and ML only. |
| `frontend` | Dashboard only; points at an API via `<meta name="drishti-api">`. |

Merge work back into `main` rather than between `backend` and `frontend`
directly — `main` is the only branch where the two are tested together.
