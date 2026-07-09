# BookHaven: AI-Powered Bookstore Ecosystem

BookHaven is an enterprise-grade online bookstore built upon a **Unified Search and Recommendation (USR) Framework**. The system uses a compute-decoupled microservices architecture designed to isolate transactional I/O operations from heavy deep learning tensor computations.

---

## 🏗️ System Architecture

BookHaven divides its operations across three decoupled execution layers:

```
                  +-----------------------------------+
                  |        React Client (SPA)         |
                  |     Tailwind CSS / Axios / HMR     |
                  +-----------------------------------+
                                    |
                                    | HTTP / REST (JWT Cookies)
                                    v
                  +-----------------------------------+
                  |     Express.js API Gateway (BFF)  |
                  |   Node.js Session & Transactions  |
                  +-----------------------------------+
                    /                               \
    HTTP / REST    /                                 \  Mongoose
   (Low Latency)  /                                   \ (Vector Queries)
                 v                                     v
+-----------------------------+       +-----------------------------+
|    FastAPI AI Engine        |       |    MongoDB Atlas NoSQL      |
|  PyTorch / Transformers /   |       |    HNSW vector indexes      |
|  Matrix Math / RAM Cache    |       |   SearchLogs / Transactions |
+-----------------------------+       +-----------------------------+
```

1. **Client Tier (ReactJS)**: A responsive single-page client optimized for minimal cognitive load. Integrates UX search debouncing and in-flight request cancellation via native `AbortController` hooks.
2. **Transactional Domain BFF (Node.js/Express)**: Orchestrates business logic, authentication route guards, Stripe PCI-DSS token storage, and asynchronous telemetry logging.
3. **Computational AI Domain (FastAPI/Python)**: Houses PyTorch neural weights, fits lexical BM25 indices, calculates vector dot products, and executes ranking logic in memory.

---

## 🧠 Core AI Modules & Detailed Workflows

### 🔍 1. Multi-Stage Hybrid Search Pipeline Workflow
Retrieves and ranks relevant catalog books in under 150ms using a cascading funnel:

```
[User Input Query] ---> Debounce 500ms / Abort In-Flight ---> Express Gateway
                                                                   |
                                                            Audit Draft Log
                                                                   |
                                                         GET /api/v1/search (AI)
                                                                   |
                                                         [Flan-T5 Intent Parse]
                                                                   |
                                                       +-----------+-----------+
                                                       |                       |
                                                       v                       v
                                                 [BM25 Lexical]          [E5 Vector]
                                                       |                       |
                                                       +-----------+-----------+
                                                                   |
                                                        [Reciprocal Rank Fusion]
                                                                   | (Top 50)
                                                        [Cross-Encoder Rerank]
                                                                   | (Top 20 IDs)
                                                          Gateway Metadata
                                                        Assembly (Map Sort)
                                                                   |
                                                            Response & Audit
```

* **Step 1: Input Debounce & Query Dispatch (React Client)**: The search bar intercepts the user's keystrokes. To prevent database lookup overloading, the input triggers a 500ms debounce timer. It also instantiates a native `AbortController` signal, aborting any slow in-flight fetch requests if the user continues typing.
* **Step 2: Gateway Interception & Audit Staging (Express BFF)**: The Express server catches the route (`GET /api/search`), checks secure cookies to optionally extract `userId` metadata, and instantly writes a draft audit log in `search_logs` to ensure auditability even if downstream connections time out.
* **Step 3: Query Parsing (FastAPI AI Engine)**: zero-shot classification via `google/flan-t5-small` categorizes the query intent into `['author', 'genre', 'title', 'general']` labels to guide the semantic focus.
* **Step 4: Dual-Stream Retrieval (FastAPI AI Engine)**:
  * *Lexical Stream (BM25)*: Queries the tokenized metadata corpus using the `rank_bm25` index, capturing term overlaps (exact titles, authors, ISBNs).
  * *Semantic Stream (E5)*: Prefixes the query with `"query: "` and passes it through the `intfloat/multilingual-e5-base` Bi-Encoder, yielding a 768-dimensional query embedding. It computes Cosine Similarity against cached catalog vector matrices using fast vectorized NumPy matrix multiplications:
    $$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
* **Step 5: Rank Fusion & Cross-Attention Reranking (FastAPI AI Engine)**: Merges the top 100 candidate arrays from both streams using Reciprocal Rank Fusion ($k=60$) to neutralize incommensurable score bounds. The top 50 fused candidates are concatenated into token-level sequences (`[CLS] Query [SEP] Book Description [SEP]`) and scored by the `ms-marco-MiniLM-L-6-v2` Cross-Encoder self-attention model, returning the Top 20 best candidate IDs.
* **Step 6: Metadata Assembly & Exclusions (Express BFF)**: Retrieves the Top 20 full book metadata records from MongoDB using the `$in` query operator. It excludes the heavy `embedding_vector` payload (`select("-embedding_vector")`) to minimize network transit. Finally, it uses a JavaScript `Map` to restore the exact ranking order computed by the AI engine, returns the JSON payload to the client, and updates the search audit log to a completed status asynchronously.

---

### 🎯 2. Context-Aware Hybrid Recommendation Workflow
Calculates personalized carousels by fusing three recommendation metrics under a **Multi-Attribute Utility Theory (MAUT)** model:

```
[Page Context Load] ---> GET /api/recommendations ---> Fetch Profile (Check Cold-Start)
                                                                |
                                                     +----------+----------+
                                                     |                     |
                                            (Returning User)          (New User)
                                                     |                     |
                                            Load User Profile         Load Onboarding
                                           Interactions History      Preferences & B_max
                                                     |                     |
                                                     v                     v
                                                 +---+---------------------+---+
                                                 |   Tripartite AI Retrieval   |
                                                 |  - Collaborative (SVD/ALS)  |
                                                 |  - Semantic Proximity (KNN)  |
                                                 |  - Onboarding Tags Matching |
                                                 +---+---------------------+---+
                                                                 |
                                                          [MAUT Scoring]
                                                      (Context Weight Fusion)
                                                                 |
                                                      [Alibaba Price Penalty]
                                                                 |
                                                      [Slate Fill-Rate Check]
                                                            /         \
                                                       (>= 5)         (< 5)
                                                         /               \
                                                        v            Self-Healing
                                                Top 20 Book Slate    Budget Loop
```

* **Step 1: Context Trigger (React Client)**: When loading the Homepage or Book Details screen, the client calls `GET /api/recommendations` with page context query parameters (e.g. `pageType=Homepage` or `pageType=Book Detail`).
* **Step 2: User Profiling & Cold-Start Routing (FastAPI AI Engine)**: Checks the user's historical interaction logs. 
  * *New Users*: Initialize the profile vector by extracting onboarding categories and authors as a baseline user anchoring embedding, setting $B_{max}$ as the budget comfort index.
  * *Returning Users*: Extract SVD collaborative vectors and compute the active anchoring vector based on interaction weights decayed exponentially over time ($w_{decay} = w_{raw} \cdot e^{-\lambda_k \Delta t}$).
* **Step 3: Tripartite Retrieval (FastAPI AI Engine)**:
  * *Behavioral (SVD/ALS)*: Queries the collaborative filter predictions using low-rank user-item matrices derived from regularized matrix factorization.
  * *Semantic (KNN)*: Runs approximate nearest neighbor cosine similarity matches against the user's active anchoring embedding.
  * *Static Preferences*: Scores matching categorical genres and author labels.
* **Step 4: Contextual Score Fusion (MAUT)**: Fuses the three retrieval scores using dynamically assigned context weights. The weights vary by page context to maximize relevance:
  * *Homepage Context*: Weighted heavily toward Collaborative Filtering ($W_{SVD} = 0.7$, $W_{KNN} = 0.2$, $W_{Sem} = 0.1$) to show broad crowd-wisdom preferences.
  * *Book Detail Context*: Weighted heavily toward Content-based Semantic Similarity ($W_{SVD} = 0.3$, $W_{KNN} = 0.6$, $W_{Sem} = 0.1$) to display contextually relevant similar books.
* **Step 5: Price Attenuation & Self-Healing Budget Allocation (FastAPI AI Engine)**: Attenuates utility scores for books exceeding the user's budget Comfort ceiling ($B_{max}$) using the Alibaba Soft Price Penalty:
  $$S_{\text{final}} = S_{\text{AI}} \cdot \exp(-\lambda \cdot \max(0, P_{\text{item}} - B_{\text{max}}))$$
  If the resulting candidate count drops below $K=5$, the self-healing loop relaxes $B_{max}$ by 15% and recalculates scores in memory. If still starved after 2 relaxation cycles, the engine injects popular bestseller fallback slates.
* **Step 6: Gateway Resolution (Express BFF)**: Maps the Top 20 recommended book IDs to MongoDB documents, strips database vector weights, and mounts the carousel grid onto the React interface.

---

## 📂 Project Directory Structure

```text
bookhaven-root/
├── backend/                       # Node.js/Express Transactional Server
│   ├── controllers/               # Route Controllers (Auth, Search, Checkout)
│   ├── db/                        # MongoDB Database Connections
│   ├── middleware/                # Route Interceptors (verifyToken.js)
│   ├── models/                    # Mongoose Schemas (User, Book, Review, PaymentMethod)
│   ├── routes/                    # API Endpoints Routing
│   ├── services/                  # Business Logic Services (search_service.js)
│   ├── utils/                     # JWT Tokens and Cron Scheduling Utilities
│   └── index.js                   # Express App Entry Point
├── frontend/                      # React/Vite Presentation Client
│   ├── src/
│   │   ├── components/            # Reusable UI Elements (Navbar, Footer, Carousels)
│   │   ├── hooks/                 # Custom React Hooks (useMainPageLogic)
│   │   ├── pages/                 # Full Page Views (MainWebPage, Lookback, Details)
│   │   ├── App.jsx                # Router Routing and Layout
│   │   └── main.jsx               # React Dom Mounting Point
│   └── vite.config.js             # Vite Build Settings
└── ai_engine/                     # FastAPI Python Server
    ├── cache/                     # Pre-computed numpy vectors & metadata cache
    ├── jobs/                      # Cron jobs (offline recommendation updates)
    ├── requirements.txt           # Python library requirements
    ├── search_service.py          # AI Engine implementation class
    └── main.py                    # FastAPI ASGI application entry
```

---

## ⚡ Getting Started & Setup

### Prerequisites
* **Node.js**: v18.0.0 or higher
* **Python**: v3.10.0 or higher
* **MongoDB**: A MongoDB Atlas connection URI

### 1. Configuration (`.env`)
Create a `.env` file in the project root directory:
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/bookstore_db?retryWrites=true&w=majority
PORT=5001
JWT_SECRET=your_jwt_secret_key_here
NODE_ENV=development
CLIENT_URL=http://localhost:5173
AI_ENGINE_URL=http://localhost:8000
```

### 2. Startup Database Seeding
If you are running the project for the first time, seed the catalog:
```bash
cd backend
npm install
node seed_data.js
```

### 3. Running the Transactional BFF Gateway (Node.js)
```bash
cd backend
npm run dev
# Running on http://localhost:5001
```

### 4. Running the Client UI (React)
```bash
cd frontend
npm install
npm run dev
# Running on http://localhost:5173
```

### 5. Running the AI Inference Engine (FastAPI)
Create a Python virtual environment and launch uvicorn:
```bash
cd ai_engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
# Running on http://localhost:8000
# Initial startup downloads models and vectorizes the database (takes 5-15 mins on CPU, sub-second on subsequent reboots due to cache).
```

### 6. Synchronizing Vector Dimensions
To synchronize the pre-computed 768-D E5 vector embeddings into your remote MongoDB documents for inspection:
```bash
cd ai_engine
source venv/bin/activate
python sync_embeddings_to_db.py
```
