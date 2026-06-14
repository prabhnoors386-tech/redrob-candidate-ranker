#!/usr/bin/env python3

"""Precomputation Stage: Candidate Filtering and Embedding Generation.

This script:
1. Loads candidates from compressed JSONL.GZ or raw JSONL
2. Applies Stage 1 structural validation and honeypot detection
3. Generates embeddings using sentence-transformers
4. Outputs a lightweight NPZ index for fast inference
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from filters import is_honeypot, check_strategic_fit, validate_candidate_data


def load_candidates(
    input_path: str,
    sample_size: Optional[int] = None
) -> Tuple[List[Dict], int]:
    """Load candidates from compressed or raw JSONL.
    
    Args:
        input_path: Path to candidates file (jsonl or jsonl.gz)
        sample_size: Optional limit on number of candidates to load
        
    Returns:
        Tuple of (candidates_list, total_loaded_count)
    """
    print(f"[*] Loading candidates from {input_path}...")
    candidates = []
    total_count = 0
    
    try:
        # Handle gzip-compressed files
        if input_path.endswith('.gz'):
            with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if sample_size and line_num >= sample_size:
                        break
                    try:
                        candidate = json.loads(line.strip())
                        candidates.append(candidate)
                        total_count += 1
                        
                        if total_count % 10000 == 0:
                            print(f"[*] Loaded {total_count} candidates...")
                    except json.JSONDecodeError as e:
                        print(f"[-] Error parsing line {line_num}: {e}")
                        continue
        else:
            # Handle uncompressed JSONL
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if sample_size and line_num >= sample_size:
                        break
                    try:
                        candidate = json.loads(line.strip())
                        candidates.append(candidate)
                        total_count += 1
                        
                        if total_count % 10000 == 0:
                            print(f"[*] Loaded {total_count} candidates...")
                    except json.JSONDecodeError as e:
                        print(f"[-] Error parsing line {line_num}: {e}")
                        continue
        
        print(f"[+] Successfully loaded {total_count} candidates")
        return candidates, total_count
    
    except FileNotFoundError:
        print(f"[-] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[-] Error loading candidates: {e}", file=sys.stderr)
        sys.exit(1)


def filter_candidates(
    candidates: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """Apply Stage 1 structural validation and filtering.
    
    Args:
        candidates: Raw candidate list
        
    Returns:
        Tuple of (filtered_candidates, statistics)
    """
    print("\n[*] STAGE 1: Structural Validation & Honeypot Detection")
    print(f"[*] Processing {len(candidates)} candidates...")
    
    filtered_candidates = []
    stats = {
        "total_input": len(candidates),
        "honeypots_detected": 0,
        "failed_strategic_fit": 0,
        "invalid_data": 0,
        "passed_stage_1": 0,
        "filter_reasons": {}
    }
    
    for i, candidate in enumerate(candidates):
        if (i + 1) % 10000 == 0:
            print(f"[*] Processed {i + 1} candidates...")
        
        # Validate data integrity
        is_valid, reason = validate_candidate_data(candidate)
        if not is_valid:
            stats["invalid_data"] += 1
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
            continue
        
        # Check for honeypot
        if is_honeypot(candidate):
            stats["honeypots_detected"] += 1
            stats["filter_reasons"]["Honeypot"] = stats["filter_reasons"].get("Honeypot", 0) + 1
            continue
        
        # Check strategic fit
        is_valid, reason = check_strategic_fit(candidate)
        if not is_valid:
            stats["failed_strategic_fit"] += 1
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
            continue
        
        filtered_candidates.append(candidate)
        stats["passed_stage_1"] += 1
    
    print(f"\n[+] Stage 1 Results:")
    print(f"    - Total Input: {stats['total_input']}")
    print(f"    - Honeypots Detected: {stats['honeypots_detected']}")
    print(f"    - Failed Strategic Fit: {stats['failed_strategic_fit']}")
    print(f"    - Invalid Data: {stats['invalid_data']}")
    print(f"    - Passed Stage 1: {stats['passed_stage_1']}")
    print(f"\n[+] Filter Reasons:")
    for reason, count in sorted(stats["filter_reasons"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {reason}: {count}")
    
    return filtered_candidates, stats


def build_embeddings(
    candidates: List[Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for all candidates using sentence-transformers.
    
    Creates a text representation of each candidate profile and generates
    embeddings using a pre-trained transformer model optimized for semantic
    similarity.
    
    Args:
        candidates: Filtered candidate list
        model_name: HuggingFace model identifier
        batch_size: Batch size for embedding generation
        
    Returns:
        Tuple of (embedding_matrix, candidate_ids)
    """
    print(f"\n[*] Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"[+] Model loaded successfully")
    except Exception as e:
        print(f"[-] Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n[*] Generating embeddings for {len(candidates)} candidates...")
    
    candidate_ids = []
    texts_to_embed = []
    
    # Build text representation of each candidate
    for candidate in candidates:
        candidate_ids.append(candidate.get("id", ""))
        
        # Construct comprehensive candidate profile text
        profile_parts = []
        
        # Add current role and company
        experience = candidate.get("experience", [])
        if experience:
            current_job = experience[0]
            title = current_job.get("title", "")
            company = current_job.get("company_name", "")
            if title or company:
                profile_parts.append(f"Current: {title} at {company}")
        
        # Add all skills
        skills = candidate.get("skills", [])
        if skills:
            skill_names = [s.get("name", "") for s in skills if s.get("name")]
            if skill_names:
                profile_parts.append(f"Skills: {' '.join(skill_names)}")
        
        # Add education
        education = candidate.get("education", [])
        if education:
            edu_parts = []
            for edu in education:
                degree = edu.get("degree", "")
                field = edu.get("field", "")
                if degree or field:
                    edu_parts.append(f"{degree} in {field}" if field else degree)
            if edu_parts:
                profile_parts.append(f"Education: {' '.join(edu_parts)}")
        
        # Add certifications
        certifications = candidate.get("certifications", [])
        if certifications:
            cert_names = [c.get("name", "") for c in certifications if c.get("name")]
            if cert_names:
                profile_parts.append(f"Certifications: {' '.join(cert_names)}")
        
        # Combine all parts
        full_text = " ".join(profile_parts) if profile_parts else candidate.get("name", "")
        texts_to_embed.append(full_text)
    
    # Generate embeddings in batches
    print(f"[*] Computing embeddings with batch size {batch_size}...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"[+] Generated embeddings shape: {embeddings.shape}")
    
    return embeddings, candidate_ids


def save_precomputed_index(
    embeddings: np.ndarray,
    candidate_ids: List[str],
    candidates: List[Dict],
    output_path: str = "precomputed_pool.npz"
) -> None:
    """Save precomputed embeddings and metadata to NPZ file.
    
    Args:
        embeddings: Embedding matrix (N, embedding_dim)
        candidate_ids: List of candidate IDs
        candidates: List of candidate profiles
        output_path: Path to output NPZ file
    """
    print(f"\n[*] Saving precomputed index to {output_path}...")
    
    try:
        # Create metadata dictionary
        metadata = {
            "candidate_count": len(candidates),
            "embedding_dim": embeddings.shape[1],
            "candidate_ids": np.array(candidate_ids, dtype=object),
        }
        
        # Save arrays and metadata
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            candidate_ids=np.array(candidate_ids, dtype=object),
            candidate_profiles=np.array(candidates, dtype=object),
            candidate_count=np.array([len(candidates)]),
            embedding_dim=np.array([embeddings.shape[1]])
        )
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[+] Precomputed index saved: {output_path} ({file_size_mb:.2f} MB)")
        print(f"[+] Metadata:")
        print(f"    - Candidates: {len(candidates)}")
        print(f"    - Embedding Dimension: {embeddings.shape[1]}")
    
    except Exception as e:
        print(f"[-] Error saving index: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main precomputation orchestration."""
    parser = argparse.ArgumentParser(
        description="Precomputation: Load, filter, and embed candidates"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to candidates file (jsonl or jsonl.gz)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="precomputed_pool.npz",
        help="Path to output NPZ index"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: limit to N candidates for testing"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*70)
    print("REDROB CANDIDATE RANKER - Precomputation Stage")
    print("="*70)
    
    # Load raw candidates
    candidates, total_loaded = load_candidates(args.input, args.sample)
    
    # Stage 1: Filter and validate
    filtered_candidates, stats = filter_candidates(candidates)
    
    # Stage 2: Build embeddings
    embeddings, candidate_ids = build_embeddings(
        filtered_candidates,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Save precomputed index
    save_precomputed_index(
        embeddings,
        candidate_ids,
        filtered_candidates,
        output_path=args.output
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n[+] Precomputation Complete")
    print(f"[+] Total Time: {elapsed_time:.2f} seconds")
    print(f"[+] Throughput: {len(filtered_candidates) / elapsed_time:.0f} candidates/sec")
    print("="*70)


if __name__ == "__main__":
    main()
