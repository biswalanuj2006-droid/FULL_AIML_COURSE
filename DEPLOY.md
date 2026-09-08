# 🚀 Deploy Guide — AI/ML Engineering Academy

One FastAPI service serves **everything**: the React SPA, `/api/*`, and all course
assets (`/assets/Images`, `/assets/Videos`, `/assets/diagrams`). Any platform that
runs Python can host it in one click.

---

## 0. Prepare (once)

```bash
# from the repo root
cd platform/frontend && npm install && npm run build && cd ../..
```

## 1. Railway (recommended — free tier, zero-config)

1. Push this repo to GitHub.
2. https://railway.app → **New Project → Deploy from GitHub repo** → pick this repo.
3. Railway reads `railway.toml` automatically:
   - builds the frontend (Nixpacks),
   - starts `uvicorn` after seeding the database,
   - health-checks `/api/health`.
4. **Variables** → add `JWT_SECRET` (any long random string). That's it.
5. **Settings → Networking → Generate Domain** → your app is live.

Optional Postgres: **New → Database → PostgreSQL**, then set
`DATABASE_URL=${{Postgres.DATABASE_URL}}` on the web service.

CLI alternative:

```bash
railway login
railway init
railway up
```

## 2. Render

`render.yaml` blueprint: **New → Blueprint** → select the repo → Apply.
Free tier + persistent disk included.

## 3. Fly.io

```bash
fly launch        # picks up fly.toml (Docker build + volume + healthcheck)
fly deploy
```

## 4. Docker / Docker Compose (any VPS, EC2, Azure, GCP)

```bash
docker compose up --build -d        # http://localhost:8000
```

The image bakes in the SPA + all diagrams/videos; SQLite lives on the `/data` volume.

## 5. Heroku

```bash
heroku create aiml-academy
heroku stack:set container      # or git push heroku main (Procfile buildpack)
git push heroku main
```

The `Procfile` seeds the DB on release and boots the web process.

## 6. Vercel / Netlify (frontend-only split deploy)

1. Deploy the backend to Railway first (section 1).
2. Replace `YOUR-RAILWAY-APP` in `vercel.json` / `netlify.toml` with your
   Railway domain, then import the repo there.
   Static hosting proxies `/api/*` and `/assets/*` to the backend.

## 7. Cloud-agnostic notes

- **Database**: SQLite works out of the box; set `DATABASE_URL` to Postgres for
  multi-instance deploys.
- **Secrets**: always set `JWT_SECRET` in production.
- **Seeding**: every start command runs `python seed_run.py`, which is idempotent
  (it skips when data exists).
- **Assets**: served from the repo checkout — no CDN needed; put one in front later.

## Default accounts (change in production!)

| role | email | password |
|---|---|---|
| admin | admin@aiml.dev | admin12345 |
| learner | demo@aiml.dev | demo12345 |
