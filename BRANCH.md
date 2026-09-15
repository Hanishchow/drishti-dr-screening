# `frontend` branch

The Drishti dashboard. **No server.** Static files only — HTML, CSS and ES
modules, with no build step and no dependencies to install.

```
web/index.html      markup only
web/css/styles.css  design tokens and layout
web/js/api.js       transport, session, shared helpers
web/js/screening.js patient, capture, result panels, history
web/js/queue.js     ophthalmologist review queue and sign-off
web/js/planning.js  district capacity simulation
web/js/app.js       entry point: session bootstrap, navigation
```

## Pointing it at an API

This branch has no backend, so it needs one running somewhere. Set the origin
in `web/index.html`:

```html
<meta name="drishti-api" content="http://localhost:8000">
```

Leave it empty when the backend serves these files itself, which is how `main`
deploys.

## Running it

```bash
python -m http.server 5500 --directory web
```

Then open <http://localhost:5500>.

The API must allow this origin, or the browser blocks every call:

```bash
DRISHTI_CORS_ORIGINS=http://localhost:5500 uvicorn server.app:app --port 8000
```

That is a real constraint, not a formality — the server sends an explicit
allow-list rather than a wildcard, precisely so an arbitrary page a clinician
has open cannot drive the API with their session.

## There is no sign-in screen

Deliberate. The dashboard asks the API for a session, which is granted only
where `DRISHTI_OPEN_ACCESS=1` is set — a demo or a PHC edge node. A district
server refuses to start with it enabled, because open access there would hand
clinical sign-off to anyone who loads the page.

Authorisation is unaffected either way: every endpoint checks capabilities, so
a session obtained this way still cannot sign off a grade unless its role is
ophthalmologist. If the API declines, the page says so and explains how to
enable it rather than showing a form that would not help.

## Branch layout

| Branch | Contains |
|---|---|
| `main` | Everything, integrated. The deployable trunk. |
| `backend` | API, ML and simulation. No dashboard. |
| `frontend` | This branch. Dashboard only. |

Merge work back into `main` rather than between `backend` and `frontend`
directly — `main` is the only branch where the two are tested together, and
the only one whose test suite covers the API these modules call.
