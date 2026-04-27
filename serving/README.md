# Serving Team Contribution Overview

As a member of the Serving team, my work covered the full path from model selection and API serving to product integration inside the ActualBudget UI. The serving stack was not limited to exposing a prediction endpoint; it also included secure user-aware access, frontend changes for prediction review, a persistence layer for feedback and personalization, and multi-model inference routing based on request demand.

## High-level contribution summary

- Built and integrated the serving workflow used by the application to classify transactions.
- Extended the ActualBudget UI so model predictions are visible and actionable during import and review flows.
- Unified authentication and improved the login experience between the budgeting app and the ML serving layer.
- Fixed the default multi-user access issue in the open-source Actual codebase so users can only access files they created or own.
- Implemented the FastAPI serving service and connected it to a PostgreSQL-backed persistence layer.
- Added support for three different models and routed them during serving depending on traffic and usage context.

## Initial implementation details

The initial phase focused on benchmarking candidate models and selecting a deployment path that was practical for production-style serving. We compared four models: `distilbert`, `minilm`, `tfidf_logreg`, and `fasttext`, measuring latency, throughput, model size, and hardware compatibility. Based on those results, we chose FastAPI as the serving framework and used the benchmarking results to guide the final serving design.

## UI, authentication, and user isolation changes

One major part of my work was integrating the ML workflow into the ActualBudget product experience instead of keeping it as a standalone backend experiment. I updated the UI to support a more unified login and authentication flow, so the serving layer can reliably identify the currently logged-in user and process requests in a user-aware way.

I also implemented UI improvements around prediction review. This included a modal window that displays model predictions to the user, along with controls that let the user review or override the predicted category. I added support for tagging a transaction with a custom category and included a custom category button so users can assign a new category to a specific payee when the default prediction is not sufficient.

The open-source Actual repository we cloned was originally designed more for a single-user or single-organization setup. Because of that, a logged-in user could access files created by other users. I changed this behavior so file access is restricted per user, ensuring that a user can only access the files that they created or are allowed to own. This was important for making the system usable in a proper multi-user environment.

## FastAPI service and PostgreSQL schema

I implemented the serving backend using FastAPI and structured it as the main API layer for transaction classification and related serving operations. This service handles prediction requests, user-linked feedback, custom category tagging, and other interactions needed by the frontend.

To support persistence, I designed and implemented the PostgreSQL schema used by the serving system. This schema stores prediction feedback, user-specific examples, custom categories, and supporting serving metadata so the application can learn from user corrections and maintain personalized behavior across sessions.

## Multi-model serving

I implemented three models that are used during serving based on demand and request type:

- `MiniLM` for higher-quality interactive predictions.
- `FastText` for fast bulk-serving scenarios.
- `TF-IDF + Logistic Regression` as a lightweight fallback path.

Instead of using a single static model for every request, the serving system can choose among these models depending on the serving context. This made the deployment more flexible by balancing prediction quality, response time, and serving cost for different workloads.

## End-to-end serving impact

Overall, my contribution as a Serving team member was to turn the model-serving work into a usable application feature. That included model evaluation, API implementation, frontend integration, authentication flow updates, multi-user access control fixes, PostgreSQL-backed persistence, and demand-aware multi-model serving. In practice, this connected the ML models to the real user workflow rather than keeping them as isolated benchmark artifacts.
