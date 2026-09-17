# Roadmap

Alexandria is built in small, shippable phases. Each phase ends with something you can run,
test, and talk about in an interview. ✅ = implemented in this repo, ⬜ = next steps.

---

## Phase 0 - Project setup ✅
- Monorepo: `core/` (shared algorithms), `ml/` (training), `api/` (serving), `web/` (UI).
- Python venv, Ruff, pytest; Next.js + TypeScript + Tailwind scaffold.

**Try it:** `pytest core/tests`

## Phase 1 - Data pipeline ✅
- Download Goodbooks-10k (10k books, ~6M ratings, Goodreads shelf tags).
- Clean titles, drop duplicate ratings, re-index users/items to dense ids.
- Turn noisy user shelves ("to-read", "ya-fantasy", "owned-books") into a curated genre vocabulary.
- Synthetic dataset with the same schema for fast tests and CI.

**Key files:** `ml/alexandria_ml/data/` · **Learn:** implicit vs explicit feedback, data leakage.

## Phase 2 - Baselines & content model ✅
- Popularity baseline (always build one first!).
- Book "documents" from title + author + genres + tags → MiniLM sentence embeddings
  (TF-IDF + SVD fallback).
- Evaluation harness: per-user holdout, Recall@K, NDCG@K, hit rate, catalog coverage.

**Key files:** `ml/alexandria_ml/features/embeddings.py`, `core/alexandria_core/metrics.py`

## Phase 3 - Collaborative filtering ✅
- BPR matrix factorization in PyTorch with negative sampling.
- MLflow tracking of params, per-epoch loss and all evaluation metrics.
- Refit on full data and export versioned artifacts + `manifest.json`.

**Key files:** `ml/alexandria_ml/models/bpr.py`, `ml/alexandria_ml/pipeline.py`

## Phase 4 - Hybrid recommender with online adaptation ✅
- Blend z-scored content, CF, genre, popularity and quality signals.
- CF weight grows with feedback volume (cold start → personalized).
- Fold-in: solve a user vector per request from feedback - "YouTube-style" instant adaptation.
- MMR diversity + exploration slots; per-item explanations.
- Evaluated with the *same code path* the API serves (`hybrid_foldin`, `hybrid_cold5`).

**Key files:** `core/alexandria_core/recommender.py`, `core/alexandria_core/foldin.py`

## Phase 5 - Backend API ✅
- FastAPI + SQLAlchemy 2.0 + Postgres/pgvector (HNSW index for similar-book search).
- JWT auth, onboarding, library/feedback, recommendations, event logging.
- Seeder that loads ML artifacts (idempotent upsert for model refreshes).
- API tests on SQLite; CI integration test on real Postgres + pgvector.

**Try it:** `uvicorn app.main:app --reload` → http://localhost:8000/docs

## Phase 6 - LLM onboarding chatbot ✅
- Claude interviews the reader; structured outputs return the reply *and* extracted
  genres/loved/disliked books every turn.
- Mentions are resolved to catalog ids; the user confirms before they become feedback.
- Degrades gracefully (quiz still works) when no API key is configured.

**Key files:** `api/app/llm/onboarding_agent.py`, `web/src/app/onboarding/ChatOnboarding.tsx`

## Phase 7 - Frontend ✅
- Landing, auth, onboarding (quick picks or chat), Discover grid with one-tap feedback and
  live refresh, My Shelf, book pages with "more like this".

## Phase 8 - Containers, CI & deployment ✅ (deploy steps in DEPLOYMENT.md)
- Dockerfiles for API and web, `docker compose` stack with pgvector.
- GitHub Actions: lint, unit tests, pipeline test, Postgres integration smoke test, image builds.
- Deploy: Neon (Postgres), Render (API), Vercel (web).

---

## Next steps (good follow-ups to extend the project)

### Phase 9 - Close the feedback loop ⬜
- [ ] Export `interactions` from production and merge with Goodbooks ratings for retraining.
- [ ] Scheduled GitHub Action (weekly) that retrains, evaluates, and only promotes a model if
      NDCG@20 doesn't regress (model registry in MLflow).
- [x] Hot-reload the recommender when a new `model_version` is seeded.

### Phase 10 - Scale the catalog: 10k → 200k books ⬜
Goodbooks-10k only covers popular books and has no descriptions, which is why the content model is
weak (Recall@20 0.03). The [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads)
has 2.36M books, 229M interactions and real blurbs (check its academic-use terms first).
- [ ] Ingest a filtered subset (~100-200k books with enough interactions) + descriptions.
- [ ] Re-embed with descriptions; measure the content model's improvement.
- [ ] Item cold start: content-only scoring for books with no interactions.
- [ ] Move candidate generation into pgvector (ANN over content + CF vectors, then blend ~500
      candidates) - the in-memory full scan won't fit a free tier past ~100k books.
- [ ] `halfvec` / smaller embeddings to fit Neon's free storage.

### Phase 11 - Better models ⬜
- [ ] Two-tower neural retrieval model (user tower over history + item tower over text features).
- [ ] Learning-to-rank re-ranker (LightGBM LambdaMART) on candidate features + logged impressions.
- [x] Richer book text: descriptions, subjects and covers from Open Library (improved similarity;
      ranking accuracy unchanged - see ARCHITECTURE.md).
- [ ] Hyperparameter search (Optuna) tracked in MLflow.

### Phase 12 - Measure online ⬜
- [ ] Dashboard of CTR / "loved" rate by recommendation reason and position from `events`.
- [ ] A/B test framework: assign users to ranking variants, compare engagement.
- [x] Rate limiting on the LLM chat endpoint.
- [ ] Alembic migrations, structured logging, Sentry/OpenTelemetry.

### Phase 13 - Product polish ⬜
- [ ] "Chat to refine" on the Discover page ("something shorter and funnier").
- [ ] Goodreads CSV import to bootstrap a profile.
- [ ] Reading-goal tracking and shareable shelves.
