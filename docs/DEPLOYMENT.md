# Deployment

Free-tier friendly setup:

| Piece | Host | Why |
|---|---|---|
| Postgres + pgvector | [Neon](https://neon.tech) | serverless Postgres, pgvector supported, free tier |
| API (Docker) | [Render](https://render.com) | deploys `api/Dockerfile` from GitHub via `render.yaml` |
| Web | [Vercel](https://vercel.com) | native Next.js hosting |

> Alternatives: Supabase for the DB, Fly.io / Railway / Google Cloud Run for the API.

## 1. Train the production model (locally)

```bash
pip install -e "./ml[sbert,tracking]"
cd ml && python -m alexandria_ml.pipeline      # Goodbooks-10k, MiniLM embeddings, BPR
```

Copy the metrics from `ml/artifacts/manifest.json` into the README results table.

## 2. Database - Neon

1. Create a project → copy the connection string.
2. Use the **direct** connection (turn off "Connection pooling" in the Connect dialog) and paste it
   as-is - the API converts `postgresql://` / `postgres://` to the `postgresql+psycopg://` driver URL.
3. Seed from your machine (creates tables, the `vector` extension and the HNSW index):

```bash
cd api
DATABASE_URL="postgresql://...?sslmode=require" python -m app.seed --artifacts ../ml/artifacts
```

## 3. API - Render

1. Push the repo to GitHub.
2. Render → **New → Blueprint** → select the repo (uses `render.yaml`).
3. Set environment variables:
   - `DATABASE_URL` - the Neon URL from step 2
   - `CORS_ORIGINS` - `https://<your-vercel-app>.vercel.app,http://localhost:3000` (comma-separated)
   - `ANTHROPIC_API_KEY` - optional, enables the chat librarian
   - `JWT_SECRET` is generated automatically
4. Verify `https://<service>.onrender.com/health` returns `{"status":"ok","books":10000}`.

Free Render instances sleep when idle; the first request after a sleep takes ~30-50 s.

## 4. Web - Vercel

1. Vercel → **Add New Project** → import the repo, set **Root Directory** to `web`.
2. Environment variable: `NEXT_PUBLIC_API_URL=https://<service>.onrender.com`
3. Deploy, then add the Vercel URL to the API's `CORS_ORIGINS` and redeploy the API.

## 5. Updating the model

Retrain → re-run the seeder against Neon (it upserts books, keeping user data) → restart the
Render service so the in-memory recommender reloads.

## Security checklist

- [ ] Strong `JWT_SECRET` (Render generates one).
- [ ] `CORS_ORIGINS` restricted to your frontend domain.
- [ ] Keep `ANTHROPIC_API_KEY` only in the host's secret store - never in the repo or `NEXT_PUBLIC_*` vars.
- [ ] Set a **monthly spend limit** in the Anthropic Console (Settings → Limits).
- [x] `/chat/onboarding` is rate limited per user (hour/day), per IP (hour) and globally (day);
      tune with the `CHAT_LIMIT_*` environment variables. Usage is stored in the `chat_usage` table.
