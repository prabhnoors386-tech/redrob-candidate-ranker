#!/usr/bin/env python3

"""Redrob Candidate Ranker - Stage 2 Fast Inference Orchestration.

This script:
1. Loads the precomputed index (embeddings + candidate profiles)
2. Reads job description and converts to embedding
3. Calculates semantic similarities using NumPy vectorization
4. Applies behavioral signal multipliers
5. Outputs exactly 100 ranked candidates in CSV format

Designed for CPU-only execution in under 5 minutes.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.filters import is_honeypot, check_strategic_fit
from src.scoring import calculate_semantic_score, apply_behavioral_multipliers


def load_precomputed_index(index_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Load precomputed embeddings and candidate profiles from NPZ file.
    
    Args:
        index_path: Path to precomputed_pool.npz
        
    Returns:
        Tuple of (embeddings_matrix, candidate_ids, candidate_profiles)
    """
    print(f"[*] Loading precomputed index from {index_path}...")
    
    try:
        data = np.load(index_path, allow_pickle=True)
        
        embeddings = data["embeddings"]
        candidate_ids = list(data["candidate_ids"])
        candidates = list(data["candidate_profiles"])
        
        print(f"[+] Loaded {len(candidates)} candidates")
        print(f"[+] Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings, candidate_ids, candidates
    
    except FileNotFoundError:
        print(f"[-] Index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[-] Error loading index: {e}", file=sys.stderr)
        sys.exit(1)


def load_job_description(jd_path: Optional[str] = None) -> Dict:
    """Load job description for ranking.
    
    Args:
        jd_path: Path to JD JSON file, or None for default
        
    Returns:
        Job description dictionary
    """
    if not jd_path:
        print("[*] Using default job description")
        return {
            "title": "Data & AI Engineer",
            "description": "Seeking experienced data scientists and AI engineers with strong ML fundamentals",
            "required_skills": [
                "python", "machine learning", "data analysis", "tensorflow", "pytorch",
                "sql", "statistics", "deep learning", "nlp", "computer vision"
            ],
            "preferred_skills": [
                "kubernetes", "docker", "aws", "gcp", "azure",
                "git", "ci/cd", "agile", "communication", "leadership"
            ],
            "years_experience_required": 3
        }
    
    print(f"[*] Loading job description from {jd_path}...")
    try:
        with open(jd_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[-] Error loading JD: {e}")
        return load_job_description(None)


def vectorize_job_description(
    jd: Dict,
    model: SentenceTransformer
) -> np.ndarray:
    """Convert job description to embedding vector.
    
    Args:
        jd: Job description dictionary
        model: SentenceTransformer model
        
    Returns:
        Embedding vector for JD
    """
    print("[*] Generating job description embedding...")
    
    # Build comprehensive JD text
    jd_parts = []
    
    if jd.get("title"):
        jd_parts.append(f"Title: {jd['title']}")
    
    if jd.get("description"):
        jd_parts.append(f"Description: {jd['description']}")
    
    if jd.get("required_skills"):
        jd_parts.append(f"Required Skills: {' '.join(jd['required_skills'])}")
    
    if jd.get("preferred_skills"):
        jd_parts.append(f"Preferred Skills: {' '.join(jd['preferred_skills'])}")
    
    if jd.get("years_experience_required"):
        jd_parts.append(f"Years Experience: {jd['years_experience_required']}")
    
    jd_text = " ".join(jd_parts)
    jd_vector = model.encode(jd_text, convert_to_numpy=True)
    
    print(f"[+] JD embedding generated (dim: {len(jd_vector)})")
    return jd_vector


def extract_behavioral_signals(candidate: Dict) -> Dict:
    """Extract all 23 behavioral signals from candidate profile.
    
    Args:
        candidate: Candidate profile dictionary
        
    Returns:
        Dictionary of signal values
    """
    return {
        "recruiter_response_rate": candidate.get("recruiter_response_rate", 0.5),
        "months_since_last_login": candidate.get("months_since_last_login", 0),
        "profile_completion_percentage": candidate.get("profile_completion_percentage", 100),
        "interview_completion_rate": candidate.get("interview_completion_rate", 0.5),
        "profile_view_count": candidate.get("profile_view_count", 0),
        "applications_last_30_days": candidate.get("applications_last_30_days", 0),
        "interview_response_time_days": candidate.get("interview_response_time_days", 5),
        "skills_endorsement_count": candidate.get("skills_endorsement_count", 0),
        "recommendation_count": candidate.get("recommendation_count", 0),
        "has_profile_photo": candidate.get("has_profile_photo", False),
        "has_bio": candidate.get("has_bio", False),
        "has_verified_links": candidate.get("has_verified_links", False),
        "job_title_relevance_score": candidate.get("job_title_relevance_score", 0.5),
        "experience_match_score": candidate.get("experience_match_score", 0.5),
        "skills_overlap_percentage": candidate.get("skills_overlap_percentage", 0.0),
        "network_connections": candidate.get("network_connections", 0),
        "previous_interview_score": candidate.get("previous_interview_score", 0.5),
        "certification_count": candidate.get("certification_count", 0),
        "project_count": candidate.get("project_count", 0),
        "github_commits_3m": candidate.get("github_commits_3m", 0),
        "publication_count": candidate.get("publication_count", 0),
        "open_to_relocation": candidate.get("open_to_relocation", False),
        "salary_match_score": candidate.get("salary_match_score", 0.5),
        "notice_period_days": candidate.get("notice_period_days", 30),
        "education_match_score": candidate.get("education_match_score", 0.5)
    }


def rank_candidates(
    embeddings: np.ndarray,
    candidate_ids: List[str],
    candidates: List[Dict],
    jd_vector: np.ndarray
) -> List[Tuple[str, float, str]]:
    """Rank candidates using vectorized NumPy operations.
    
    Applies semantic similarity + behavioral multipliers to all candidates
    and returns sorted results.
    
    Args:
        embeddings: Embedding matrix (N, embedding_dim)
        candidate_ids: List of candidate IDs
        candidates: List of candidate profiles
        jd_vector: Job description embedding
        
    Returns:
        List of (candidate_id, final_score, reasoning) tuples sorted by score
    """
    print("\n[*] STAGE 2: Fast Inference - Semantic Scoring & Ranking")
    print(f"[*] Ranking {len(candidates)} candidates...")
    
    # Vectorized cosine similarity calculation
    print("[*] Computing vectorized cosine similarities...")
    
    # Normalize embeddings and JD vector
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    jd_norm = jd_vector / (np.linalg.norm(jd_vector) + 1e-10)
    
    # Vectorized dot product
    semantic_scores = np.dot(embeddings_norm, jd_norm)
    
    # Clamp scores to [0, 1]
    semantic_scores = np.clip(semantic_scores, 0, 1)
    
    print(f"[+] Semantic scores computed (mean: {semantic_scores.mean():.4f})")
    
    ranked_results = []
    
    # Apply behavioral multipliers for each candidate
    print("[*] Applying behavioral signal multipliers...")
    for i, (candidate_id, candidate, semantic_score) in enumerate(
        zip(candidate_ids, candidates, semantic_scores)
    ):
        if (i + 1) % 10000 == 0:
            print(f"[*] Processed {i + 1} candidates...")
        
        # Extract behavioral signals
        signals = extract_behavioral_signals(candidate)
        
        # Apply multipliers
        final_score = apply_behavioral_multipliers(float(semantic_score), signals)
        
        # Generate reasoning string
        reasoning = f"Semantic: {semantic_score:.4f}, Behavioral: {(final_score/semantic_score if semantic_score > 0 else 1.0):.2f}x"
        
        ranked_results.append((candidate_id, final_score, reasoning))
    
    # Sort by score descending, then by candidate_id ascending (deterministic tie-breaker)
    print("[*] Sorting candidates...")
    ranked_results.sort(key=lambda x: (-x[1], x[0]))
    
    print(f"[+] Ranking complete")
    print(f"[+] Top score: {ranked_results[0][1]:.6f}")
    print(f"[+] 100th score: {ranked_results[99][1]:.6f}" if len(ranked_results) >= 100 else "")
    
    return ranked_results


def output_csv(
    ranked_results: List[Tuple[str, float, str]],
    output_path: str,
    top_k: int = 100
) -> None:
    """Output top K candidates to CSV file.
    
    CSV contains exactly 4 columns: candidate_id, rank, score, reasoning
    
    Args:
        ranked_results: Ranked candidates with scores
        output_path: Path to output CSV
        top_k: Number of top candidates to output (exactly 100)
    """
    print(f"\n[*] Writing top {top_k} candidates to {output_path}...")
    
    top_candidates = ranked_results[:top_k]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            # Write data rows
            for rank, (candidate_id, score, reasoning) in enumerate(top_candidates, 1):
                writer.writerow([
                    candidate_id,
                    rank,
                    f"{score:.6f}",
                    reasoning
                ])
        
        print(f"[+] CSV output saved: {output_path}")
        print(f"\n[+] Top 10 Ranked Candidates:")
        print(f"{'Rank':<6} {'Candidate ID':<20} {'Score':<12} {'Reasoning':<40}")
        print("-" * 80)
        for rank, (cid, score, reason) in enumerate(top_candidates[:10], 1):
            print(f"{rank:<6} {cid:<20} {score:<12.6f} {reason[:40]:<40}")
    
    except Exception as e:
        print(f"[-] Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main rank orchestration function."""
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker - Stage 2 Inference"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to precomputed_pool.npz index"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to job description JSON (optional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates (exactly 100)"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("REDROB CANDIDATE RANKER - Stage 2: Fast Inference Orchestration")
    print("="*80)
    print(f"[*] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load precomputed index
    embeddings, candidate_ids, candidates = load_precomputed_index(args.candidates)
    
    # Load embedding model
    print(f"\n[*] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Load and vectorize JD
    jd = load_job_description(args.jd)
    jd_vector = vectorize_job_description(jd, model)
    
    # Rank candidates
    ranked_results = rank_candidates(
        embeddings,
        candidate_ids,
        candidates,
        jd_vector
    )
    
    # Output to CSV
    output_csv(ranked_results, args.output, args.top_k)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n[+] Ranking Complete")
    print(f"[+] Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"[+] Throughput: {len(candidates) / elapsed_time:.0f} candidates/sec")
    print(f"[+] Memory Efficient: CPU-only, no GPU required")
    print("="*80)


if __name__ == "__main__":
    main()
