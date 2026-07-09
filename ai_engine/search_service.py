import os
import re
import json
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables from the parent directory if running locally
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

CACHE_VERSION = 2

class SearchService:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = "bookstore_db"
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings_cache_path = os.path.join(self.cache_dir, "embeddings.npy")
        self.metadata_cache_path = os.path.join(self.cache_dir, "metadata.json")

        print("[SearchService] Initializing ML Models...")
        # 1. Load SentenceTransformer for semantic embeddings
        self.bi_encoder = SentenceTransformer("intfloat/multilingual-e5-base")
        
        # 2. Load CrossEncoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 3. Load Flan-T5 for Intent Parsing (google/flan-t5-small is light, ~300MB)
        self.intent_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.intent_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # 4. Book Metadata & Retrieval Data
        self.books = []
        self.book_embeddings = None
        self.bm25 = None
        self.book_prices = {}
        
        # Load and index data
        self.load_and_index_books()

    def load_and_index_books(self):
        """Loads book data from MongoDB, generates embeddings or loads them from local cache, and fits BM25."""
        if not self.mongo_uri:
            print("[SearchService] Warning: MONGO_URI not found in environment variables.")
            return

        try:
            print("[SearchService] Connecting to MongoDB...")
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            books_col = db["books"]

            # Load all book prices dynamically for recommendations
            print("[SearchService] Loading book prices...")
            self.book_prices = {
                str(b["_id"]): float(b.get("price", 0)) if b.get("price") is not None else 0.0
                for b in books_col.find({}, {"_id": 1, "price": 1})
            }

            # Fetch basic fields needed for indexing
            db_books = list(books_col.find(
                {}, 
                {
                    "_id": 1,
                    "title": 1,
                    "author": 1,
                    "description": 1,
                    "genres": 1,
                    "inStock": 1,
                    "stockQuantity": 1,
                    "metrics": 1
                }
            ))
            db_count = len(db_books)
            db_book_by_id = {str(b["_id"]): b for b in db_books}
            print(f"[SearchService] Total books in database: {db_count}")

            # Check if we can load from local cache to avoid generating 17k embeddings (saves minutes of startup time)
            cache_valid = False
            if os.path.exists(self.embeddings_cache_path) and os.path.exists(self.metadata_cache_path):
                try:
                    with open(self.metadata_cache_path, "r", encoding="utf-8") as f:
                        cached_metadata = json.load(f)
                    
                    cache_books = cached_metadata
                    cache_version = 1
                    if isinstance(cached_metadata, dict):
                        cache_version = cached_metadata.get("version", 1)
                        cache_books = cached_metadata.get("books", [])

                    has_recommendation_fields = (
                        isinstance(cache_books, list) and
                        (not cache_books or ("inStock" in cache_books[0] and "stockQuantity" in cache_books[0]))
                    )

                    if len(cache_books) == db_count:
                        self.book_embeddings = np.load(self.embeddings_cache_path)
                        if cache_version == CACHE_VERSION and has_recommendation_fields:
                            self.books = cache_books
                        else:
                            print("[SearchService] Enriching legacy metadata cache with recommendation fields...")
                            enriched_books = []
                            for cached_book in cache_books:
                                source_book = db_book_by_id.get(str(cached_book.get("id")), {})
                                stock_quantity = int(source_book.get("stockQuantity", 0) or 0)
                                enriched = {
                                    **cached_book,
                                    "inStock": bool(source_book.get("inStock", stock_quantity > 0)),
                                    "stockQuantity": stock_quantity,
                                    "purchaseCount": int(source_book.get("metrics", {}).get("purchaseCount", 0) or 0)
                                }
                                enriched_books.append(enriched)
                            self.books = enriched_books
                            with open(self.metadata_cache_path, "w", encoding="utf-8") as f:
                                json.dump({"version": CACHE_VERSION, "books": self.books}, f, ensure_ascii=False, indent=2)
                        cache_valid = True
                        print("[SearchService] Loaded books and embeddings from local cache successfully.")
                except Exception as cache_err:
                    print(f"[SearchService] Cache load failed, regenerating: {cache_err}")

            if not cache_valid:
                print("[SearchService] Building indexes from scratch. Generating embeddings (this may take a couple of minutes)...")
                self.books = []
                texts_to_embed = []
                
                for b in db_books:
                    b_id = str(b["_id"])
                    title = b.get("title", "")
                    author = b.get("author", "")
                    description = b.get("description", "")
                    genres = b.get("genres", [])
                    genres_str = ", ".join(genres) if isinstance(genres, list) else ""
                    stock_quantity = int(b.get("stockQuantity", 0) or 0)
                    
                    metadata = {
                        "id": b_id,
                        "title": title,
                        "author": author,
                        "description": description,
                        "genres": genres,
                        "inStock": bool(b.get("inStock", stock_quantity > 0)),
                        "stockQuantity": stock_quantity,
                        "purchaseCount": int(b.get("metrics", {}).get("purchaseCount", 0) or 0)
                    }
                    self.books.append(metadata)
                    
                    # Text representation to encode semantically (E5-base requires 'passage: ' prefix)
                    semantic_text = f"passage: Title: {title} | Author: {author} | Description: {description} | Genres: {genres_str}"
                    texts_to_embed.append(semantic_text)
                
                # Generate embeddings using Bi-Encoder
                self.book_embeddings = self.bi_encoder.encode(
                    texts_to_embed, 
                    show_progress_bar=True, 
                    convert_to_numpy=True
                )
                
                # Save cache
                np.save(self.embeddings_cache_path, self.book_embeddings)
                with open(self.metadata_cache_path, "w", encoding="utf-8") as f:
                    json.dump({"version": CACHE_VERSION, "books": self.books}, f, ensure_ascii=False, indent=2)
                print("[SearchService] Local cache generated and saved.")

            # Fit BM25 Keyword Search Index
            print("[SearchService] Fitting BM25 keyword retrieval index...")
            tokenized_corpus = []
            for b in self.books:
                # Combine title, author, description, and genres for tokenization
                genres_str = " ".join(b.get("genres", []))
                corpus_text = f"{b['title']} {b['author']} {b['description']} {genres_str}".lower()
                # Clean punctuation and tokenize
                tokens = re.findall(r"\w+", corpus_text)
                tokenized_corpus.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("[SearchService] Initialization and Indexing completed successfully.")
        except Exception as e:
            print(f"[SearchService] Error during database loading/indexing: {e}")

    def parse_intent_flan_t5(self, query: str) -> str:
        """Parses user query intent using google/flan-t5-small model."""
        try:
            prompt = (
                "Classify the search query intent for an online bookstore into exactly one of these labels: "
                "['author', 'genre', 'title', 'general'].\n\n"
                f"Query: '{query}'\n"
                "Intent:"
            )
            inputs = self.intent_tokenizer(prompt, return_tensors="pt")
            outputs = self.intent_model.generate(**inputs, max_new_tokens=8)
            result = self.intent_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            for label in ["author", "genre", "title"]:
                if label in result:
                    return label
            return "general"
        except Exception as e:
            print(f"[SearchService] Flan-T5 Intent parsing failed: {e}")
            return "general"

    def perform_hybrid_search(self, query: str) -> dict:
        """Executes Hybrid Search: Intent Parsing + BM25 + Vector + RRF + Cross-Encoder Reranking."""
        if not self.books or self.bm25 is None or self.book_embeddings is None:
            return {"intent": "general", "books": []}

        # Stage 1: Intent Parsing using Flan-T5
        intent = self.parse_intent_flan_t5(query)
        print(f"[SearchService] Parsed Intent for query '{query}': {intent}")

        # Stage 2: Keyword BM25 Retrieval (Get top 100 matching books)
        query_tokens = re.findall(r"\w+", query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:100]
        
        bm25_results = []
        for rank, idx in enumerate(top_bm25_indices):
            if bm25_scores[idx] > 0: # Only retrieve positive matches
                bm25_results.append((self.books[idx]["id"], rank + 1))

        # Stage 3: Semantic Vector Search (Get top 100 matching books - E5-base requires 'query: ' prefix)
        query_embedding = self.bi_encoder.encode(f"query: {query}", convert_to_numpy=True)
        # Cosine similarity using dot product of normalized embeddings
        norm_embeddings = self.book_embeddings / np.linalg.norm(self.book_embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        vector_scores = np.dot(norm_embeddings, norm_query)
        
        top_vector_indices = np.argsort(vector_scores)[::-1][:100]
        vector_results = []
        for rank, idx in enumerate(top_vector_indices):
            vector_results.append((self.books[idx]["id"], rank + 1))

        # Stage 4: Reciprocal Rank Fusion (RRF)
        # Merge keyword search ranks and semantic search ranks
        k = 60 # Constant parameter for RRF
        rrf_scores = {}
        
        # Process BM25
        for b_id, rank in bm25_results:
            rrf_scores[b_id] = rrf_scores.get(b_id, 0.0) + (1.0 / (k + rank))
        
        # Process Vector Search
        for b_id, rank in vector_results:
            rrf_scores[b_id] = rrf_scores.get(b_id, 0.0) + (1.0 / (k + rank))
        
        # Sort candidates by RRF score descending
        fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:100]

        # Stage 5: MiniLM Cross-Encoder Reranking
        # Re-rank the top 50 fused candidates to get the absolute best top 20
        if not fused_candidates:
            return {"intent": intent, "books": []}

        candidate_ids = [c_id for c_id, _ in fused_candidates]
        candidate_books = [b for b in self.books if b["id"] in candidate_ids]
        
        # Build query-passage pairs for the Cross-Encoder
        pairs = []
        for cb in candidate_books:
            genres_str = ", ".join(cb["genres"]) if isinstance(cb["genres"], list) else ""
            passage = f"Title: {cb['title']} | Author: {cb['author']} | Description: {cb['description']} | Genres: {genres_str}"
            pairs.append((query, passage))

        # Score matching pairs
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Map scores back to books
        reranked_results = []
        for idx, score in enumerate(rerank_scores):
            reranked_results.append({
                "id": candidate_books[idx]["id"],
                "score": float(score)
            })

        # Sort based on Cross-Encoder score descending
        reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)[:50]

        return {
            "intent": intent,
            "books": reranked_results
        }

    def get_hybrid_recommendations(self, user_id_str: str, page_type: str = "Homepage", item_id_str: str = None) -> list:
        """
        Generates top 20 hybrid recommendations using dynamic context-based weights (BPMN Pipeline).
        - Phase 1: Context acquisition & Authentication. Evaluates Cold-Start.
        - Phase 2: Tripartite AI Retrieval (Branch A: KNN search, Branch B: pre-computed/cached SVD, Branch C: static semantic matching)
        - Phase 3: Dynamic fusion, Alibaba's soft exponential budget penalty & Hard Prune, and Self-Healing catalog starvation loop.
        """
        if not self.mongo_uri:
            return []

        try:
            import math
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            users_col = db["users"]
            interactions_col = db["interactions"]
            promotions_col = db["promotions"]

            user_obj_id = ObjectId(user_id_str)
            user = users_col.find_one({"_id": user_obj_id})
            if not user:
                print(f"[SearchService] User {user_id_str} not found.")
                return []

            # 1. Fetch user interactions
            user_interactions = list(interactions_col.find({"userId": user_obj_id}))
            
            # Map book_id -> index in self.books
            book_id_to_index = {b["id"]: idx for idx, b in enumerate(self.books)}
            num_books = len(self.books)

            # Build user interaction dictionary: book_id -> weight
            user_interacted_dict = {}
            for item in user_interactions:
                b_id = str(item.get("bookId"))
                weight = float(item.get("implicitWeight", 1.0))
                user_interacted_dict[b_id] = max(user_interacted_dict.get(b_id, 0.0), weight)

            # Extract user preferences from onboarding data
            pref = user.get("preferences", {})
            fav_categories = [cat.lower().strip() for cat in pref.get("favCategories", []) if cat]
            fav_authors = [auth.lower().strip() for auth in pref.get("favAuthors", []) if auth]
            user_budget = pref.get("userBudget")
            B_max = float(user_budget) if user_budget is not None else 50.0

            # ----------------------------------------------------
            # Phase 1: Context Acquisition & Anchoring Vector
            # ----------------------------------------------------
            is_cold_start = len(user_interactions) == 0
            anchoring_vector = None

            # If Book Detail page, anchoring vector is the target item embedding
            if page_type == "Book Detail" and item_id_str:
                if item_id_str in book_id_to_index:
                    idx = book_id_to_index[item_id_str]
                    anchoring_vector = self.book_embeddings[idx]
                    print(f"[SearchService] Using Target Book {item_id_str} embedding as Anchoring Vector.")
            
            # Otherwise, use User Profile Vector (weighted interactions or onboarding anchoring vector)
            if anchoring_vector is None:
                if not is_cold_start:
                    interacted_vectors = []
                    interacted_weights = []
                    for b_id, weight in user_interacted_dict.items():
                        if b_id in book_id_to_index:
                            idx = book_id_to_index[b_id]
                            interacted_vectors.append(self.book_embeddings[idx])
                            interacted_weights.append(weight)
                    
                    if interacted_vectors:
                        interacted_vectors = np.array(interacted_vectors)
                        interacted_weights = np.array(interacted_weights).reshape(-1, 1)
                        anchoring_vector = np.sum(interacted_vectors * interacted_weights, axis=0) / (np.sum(interacted_weights) + 1e-9)
                
                # If still none (e.g. cold start user), initialize from onboarding categories
                if anchoring_vector is None:
                    # Onboarding categories dynamic anchoring vector
                    matched_vectors = []
                    for idx, b in enumerate(self.books):
                        genres = [g.lower() for g in b.get("genres", [])]
                        if any(cat == genre or cat in genre or genre in cat for cat in fav_categories for genre in genres):
                            matched_vectors.append(self.book_embeddings[idx])
                    
                    if matched_vectors:
                        anchoring_vector = np.mean(matched_vectors, axis=0)
                        print(f"[SearchService] Initialized Cold-Start Anchoring Vector from {len(matched_vectors)} onboarding category books.")
                    else:
                        # Fallback to mean of all book vectors
                        anchoring_vector = np.mean(self.book_embeddings, axis=0)

            # ----------------------------------------------------
            # Phase 2: Tripartite AI Retrieval (Parallel Processing simulation)
            # ----------------------------------------------------
            
            # Branch A: HNSW KNN Vector Search (Cosine Similarity against Anchoring Vector)
            # Constraint: inStock == true (or price/stock check)
            content_scores = np.zeros(num_books)
            norms = np.linalg.norm(self.book_embeddings, axis=1)
            anchoring_norm = np.linalg.norm(anchoring_vector)
            
            cosine_sims = np.dot(self.book_embeddings, anchoring_vector) / (norms * anchoring_norm + 1e-9)
            
            for idx, b in enumerate(self.books):
                # Filter out of stock books if stockQuantity is 0 or inStock is false
                is_in_stock = b.get("inStock", True)
                stock_quantity = int(b.get("stockQuantity", 0) or 0)
                if not is_in_stock or stock_quantity <= 0:
                    content_scores[idx] = -9.0 # Very low cosine similarity score
                else:
                    content_scores[idx] = cosine_sims[idx]

            # Branch B: Pre-computed SVD / User-Based Collaborative Filtering candidates
            behavior_scores = np.zeros(num_books)
            if not is_cold_start:
                # Group co-interactions of books the user interacted with
                interacted_book_ids = [ObjectId(bid) for bid in user_interacted_dict.keys()]
                co_interactions = list(interactions_col.find({"bookId": {"$in": interacted_book_ids}}))
                
                other_user_ids = set(str(ci["userId"]) for ci in co_interactions if str(ci["userId"]) != user_id_str)
                
                if other_user_ids:
                    similar_users_interactions = list(interactions_col.find({"userId": {"$in": [ObjectId(uid) for uid in other_user_ids]}}))
                    
                    user_profiles = {}
                    for item in similar_users_interactions:
                        uid_str = str(item["userId"])
                        bid_str = str(item["bookId"])
                        weight = float(item.get("implicitWeight", 1.0))
                        
                        if uid_str not in user_profiles:
                            user_profiles[uid_str] = {}
                        user_profiles[uid_str][bid_str] = max(user_profiles[uid_str].get(bid_str, 0.0), weight)
                    
                    user_similarities = {}
                    u_norm = np.sqrt(sum(w**2 for w in user_interacted_dict.values()))
                    
                    for other_uid_str, other_profile in user_profiles.items():
                        shared_books = set(user_interacted_dict.keys()).intersection(set(other_profile.keys()))
                        if not shared_books:
                            continue
                        dot_product = sum(user_interacted_dict[bid] * other_profile[bid] for bid in shared_books)
                        v_norm = np.sqrt(sum(w**2 for w in other_profile.values()))
                        sim = dot_product / (u_norm * v_norm + 1e-9)
                        if sim > 0.05:
                            user_similarities[other_uid_str] = sim

                    predicted_weights = np.zeros(num_books)
                    for other_uid_str, sim in user_similarities.items():
                        other_profile = user_profiles[other_uid_str]
                        for bid_str, weight in other_profile.items():
                            if bid_str in book_id_to_index:
                                idx = book_id_to_index[bid_str]
                                predicted_weights[idx] += sim * weight
                    behavior_scores = predicted_weights

            # Branch C: Static Semantic Attributes Matching
            semantic_scores = np.zeros(num_books)
            # If Book Detail page, match candidates with target book's attributes
            if page_type == "Book Detail" and item_id_str and item_id_str in book_id_to_index:
                t_idx = book_id_to_index[item_id_str]
                target_book = self.books[t_idx]
                target_genres = [g.lower() for g in target_book.get("genres", [])]
                target_author = target_book.get("author", "").lower()
                
                for idx, b in enumerate(self.books):
                    b_genres = [g.lower() for g in b.get("genres", [])]
                    genre_matches = len(set(b_genres).intersection(target_genres))
                    b_author = b.get("author", "").lower()
                    author_match = 1.0 if target_author and target_author in b_author else 0.0
                    semantic_scores[idx] = genre_matches * 1.5 + author_match * 3.0
            else:
                # If Homepage, match candidates with user's onboarding attributes
                for idx, b in enumerate(self.books):
                    b_genres = [g.lower() for g in b.get("genres", [])]
                    genre_matches = sum(
                        1 for cat in fav_categories for genre in b_genres
                        if cat == genre or cat in genre or genre in cat
                    )
                    b_author = b.get("author", "").lower()
                    author_match = 1.0 if any(auth in b_author for auth in fav_authors) else 0.0
                    semantic_scores[idx] = genre_matches * 1.5 + author_match * 3.0

            # ----------------------------------------------------
            # Phase 3: Dynamic Fusion & Self-Healing Budget Loop
            # ----------------------------------------------------
            
            # Fetch active discount rules from Promotions collection
            discount_rules = []
            try:
                discount_rules = list(promotions_col.find({"$or": [{"active": True}, {"isActive": True}]}))
            except Exception as pe:
                print(f"[SearchService] Promotions collection fetch skipped or empty: {pe}")

            # Define Context-Based Weights
            if page_type == "Book Detail":
                W_SVD = 0.3
                W_KNN = 0.6
                W_Sem = 0.1
            else:
                W_SVD = 0.7
                W_KNN = 0.2
                W_Sem = 0.1

            # Normalize scores helper
            def normalize_vector(v):
                min_v, max_v = np.min(v), np.max(v)
                if max_v - min_v > 1e-9:
                    return (v - min_v) / (max_v - min_v)
                return v

            norm_content = normalize_vector(content_scores)
            norm_behavior = normalize_vector(behavior_scores)
            norm_semantic = normalize_vector(semantic_scores)

            # Perform Weighted Scoring & Fusion
            fusion_scores = W_SVD * norm_behavior + W_KNN * norm_content + W_Sem * norm_semantic

            # Self-Healing Starvation Loop configuration
            retries = 0
            max_retries = 2
            lambda_val = 0.05
            current_B_max = B_max
            final_slate_ids = []

            while retries <= max_retries:
                candidate_scores = np.copy(fusion_scores)
                
                # Apply Soft Exponential Price Penalty and Hard Prune
                for idx, b in enumerate(self.books):
                    b_id = b["id"]
                    price = self.book_prices.get(b_id, 0.0)
                    if not b.get("inStock", True) or int(b.get("stockQuantity", 0) or 0) <= 0:
                        candidate_scores[idx] = -9999.0
                        continue
                    
                    # Apply active promotions (discount rates) if matched
                    discount_rate = 0.0
                    for rule in discount_rules:
                        rule_genres = [g.lower() for g in rule.get("genres", [])]
                        b_genres = [g.lower() for g in b.get("genres", [])]
                        
                        if (rule.get("bookId") and str(rule["bookId"]) == b_id) or \
                           (rule.get("author") and rule["author"].lower() in b.get("author", "").lower()) or \
                           (rule_genres and any(g in b_genres for g in rule_genres)):
                            discount_rate = max(
                                discount_rate,
                                float(rule.get("discountRate", rule.get("discountPercentage", 0.0))) / (
                                    100.0 if rule.get("discountPercentage") is not None else 1.0
                                )
                            )
                    
                    discounted_price = price * (1.0 - discount_rate)
                    
                    # Alibaba Soft Exponential Budget Penalty
                    penalty = math.exp(-lambda_val * max(0.0, discounted_price - current_B_max))
                    candidate_scores[idx] *= penalty
                    
                    # Hard Prune: If book exceeds current budget, eliminate it
                    if discounted_price > current_B_max:
                        candidate_scores[idx] = -9999.0

                # Exclude books the user has already interacted with or target book itself
                exclude_ids = set(user_interacted_dict.keys())
                if item_id_str:
                    exclude_ids.add(item_id_str)
                    
                for bid in exclude_ids:
                    if bid in book_id_to_index:
                        idx = book_id_to_index[bid]
                        candidate_scores[idx] = -9999.0

                # Extract books with valid scores (not pruned)
                top_indices = np.argsort(candidate_scores)[::-1]
                temp_ids = []
                for idx in top_indices:
                    if candidate_scores[idx] < -9000.0:
                        continue
                    temp_ids.append(self.books[idx]["id"])
                    if len(temp_ids) == 20:
                        break

                fill_rate = len(temp_ids)
                print(f"[SearchService] Budget Check (B_max: {current_B_max:.2f}, Retry: {retries}): Slate Fill Rate = {fill_rate}/20")

                if fill_rate >= 5 or retries >= max_retries:
                    final_slate_ids = temp_ids
                    break
                else:
                    # Catalog Starvation: Enter Self-Healing Loop, Relax Budget Max +15%
                    retries += 1
                    current_B_max *= 1.15

            # If Fill Rate still < 5 after loop, inject Graceful Fallback Catalog (popular items ignoring budget)
            if len(final_slate_ids) < 5:
                print("[SearchService] Catalog Starvation unresolved. Injecting Graceful Fallback Catalog...")
                fallback_scores = np.copy(fusion_scores)
                for bid in exclude_ids:
                    if bid in book_id_to_index:
                        idx = book_id_to_index[bid]
                        fallback_scores[idx] = -9999.0
                
                top_fallback_indices = np.argsort(fallback_scores)[::-1]
                for idx in top_fallback_indices:
                    bid = self.books[idx]["id"]
                    if bid not in final_slate_ids:
                        final_slate_ids.append(bid)
                        if len(final_slate_ids) == 20:
                            break

            return final_slate_ids

        except Exception as e:
            print(f"[SearchService] Error during hybrid recommendation: {e}")
            import traceback
            traceback.print_exc()
            return []
