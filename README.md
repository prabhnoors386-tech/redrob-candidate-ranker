# Redrob Candidate Ranker

Production-grade **Two-Stage Candidate Ranking System** for "The Data & AI Challenge: Intelligent Candidate Discovery" hackathon by Redrob.

## Overview

This system parses **100,000 candidates** and outputs exactly the **Top 100** in a CSV file within **5 minutes** on CPU, without any external network access. It defeats honeypot profiles and keyword-stuffing candidates through a robust two-stage pipeline.

### Key Features

✅ **Two-Stage Pipeline Architecture**
- **Stage 1**: Structural validation + honeypot detection
- **Stage 2**: Fast vectorized inference with behavioral multipliers

✅ **Production-Ready**
- CPU-only execution (no GPU required)
- Sandboxed Docker compatibility
- No external network dependencies
- Sub-5-minute processing time

✅ **Robust Filtering**
- Chronological timeline validation
- Consulting/IT services blocklists
- Keyword-stuffer trap detection
- 23 behavioral signal multipliers

## System Architecture

```
redrob-candidate-ranker/
├── .gitignore
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── submission_metadata.yaml       # Configuration & metadata
├── rank.py                        # Stage 2: Fast inference orchestrator
├── src/
│   ├── __init__.py
│   ├── filters.py                 # Stage 1: Honeypot detection
│   ├── scoring.py                 # Stage 2: Semantic scoring + multipliers
│   └── precompute.py              # Precomputation: filter + embed candidates
```

## Installation

### Prerequisites

- Python 3.8+
- ~4GB RAM (for 100K candidates)
- CPU only (no GPU required)

### Setup

```bash
# Clone repository
git clone https://github.com/prabhnoors386-tech/redrob-candidate-ranker.git
cd redrob-candidate-ranker

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Stage 1: Precomputation

Filter candidates and generate embeddings (run once):

```bash
python -m src.precompute \
  --input candidates.jsonl.gz \
  --output precomputed_pool.npz \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 32
```

**Arguments:**
- `--input`: Path to candidates JSONL.GZ or JSONL file
- `--output`: Output NPZ index file (default: `precomputed_pool.npz`)
- `--model`: HuggingFace sentence-transformer model (default: `all-MiniLM-L6-v2`)
- `--batch-size`: Batch size for embedding (default: 32)
- `--sample`: (Optional) Limit to N candidates for testing

**Example Output:**
```
[+] Successfully loaded 100000 candidates
[*] STAGE 1: Structural Validation & Honeypot Detection
[+] Passed Stage 1: 78456
[*] Generating embeddings for 78456 candidates...
[+] Generated embeddings shape: (78456, 384)
[+] Precomputed index saved: precomputed_pool.npz (156.92 MB)
```

### Stage 2: Ranking (Fast Inference)

Rank candidates using precomputed index:

```bash
python rank.py \
  --candidates precomputed_pool.npz \
  --output <TEAM_ID>.csv \
  --jd job_description.json \
  --model sentence-transformers/all-MiniLM-L6-v2
```

**Arguments:**
- `--candidates`: Path to precomputed NPZ index (required)
- `--output`: Output CSV filename (required)
- `--jd`: (Optional) Path to job description JSON
- `--model`: Embedding model (default: `all-MiniLM-L6-v2`)
- `--top-k`: Number of top candidates (default: 100)

**Example Output:**
```
[+] Loaded 78456 candidates
[*] Computing vectorized cosine similarities...
[+] Semantic scores computed (mean: 0.5234)
[*] Applying behavioral signal multipliers...
[+] Sorting candidates...
[+] CSV output saved: team-123.csv

[+] Top 10 Ranked Candidates:
Rank  Candidate ID         Score        Reasoning
1     cand_00001234        0.892341     Semantic: 0.7234, Behavioral: 1.23x
2     cand_00001567        0.881234     Semantic: 0.7156, Behavioral: 1.23x
...
```

## CSV Output Format

Output file contains exactly **100 rows** with **4 mandatory columns**:

```csv
candidate_id,rank,score,reasoning
cand_00001234,1,0.892341,Semantic: 0.7234; Behavioral: 1.23x
cand_00001567,2,0.881234,Semantic: 0.7156; Behavioral: 1.23x
...
```

## Two-Stage Pipeline Details

### Stage 1: Structural Precomputation

**Purpose**: Eliminate low-quality profiles before expensive semantic analysis

**Filters Applied**:
1. **Honeypot Detection**:
   - Negative employment duration
   - Employment before company founding
   - 10+ expert skills with no experience
   - Logical education/experience inconsistencies

2. **Strategic Fit Validation**:
   - Empty work history rejection
   - Non-technical current role filtering
   - Pure consulting career blocklist
   - Pure research environment detection

3. **Blocklist Application**:
   - Consulting firms: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.
   - Research terms: Postdoc, PhD student, research fellow, etc.
   - Non-tech roles: Marketing manager, HR specialist, recruiter, etc.

**Output**: `precomputed_pool.npz` containing:
- Embeddings matrix (N, 384)
- Candidate IDs list
- Full candidate profiles

### Stage 2: Fast Inference Orchestration

**Purpose**: Vectorized semantic scoring with behavioral multipliers

**Process**:
1. Load precomputed embeddings and candidates
2. Vectorize job description using same model
3. **Vectorized cosine similarity** (NumPy):
   - Normalize embeddings: `embeddings / ||embeddings||`
   - Compute: `dot(embeddings_norm, jd_norm)`
   - Result: semantic similarity scores [0, 1]

4. **Extract 23 Behavioral Signals**:
   - Recruiter response rate
   - Profile activity (months since login)
   - Profile completion %
   - Interview completion rate
   - And 19 additional signals...

5. **Apply Behavioral Multipliers**:
   ```
   final_score = semantic_score × multiplier(signals)
   ```

6. **Sort & Output**:
   - Primary: score descending
   - Tie-breaker: candidate_id ascending (deterministic)

**Behavioral Signals** (Complete List):

1. Recruiter response rate (0.30x - 0.75x)
2. Months since last login (0.40x - 0.80x)
3. Profile completion percentage (0.50x)
4. Interview completion rate (0.20x)
5. Profile view count (1.08x - 1.15x)
6. Applications last 30 days (1.05x - 1.12x)
7. Interview response time (0.85x - 1.10x)
8. Skills endorsement count (1.05x - 1.10x)
9. Recommendation count (1.05x - 1.12x)
10. Profile completeness (0.70x - 1.10x)
11. Job title relevance (0.80x - 1.20x)
12. Experience match (0.90x - 1.10x)
13. Skills overlap percentage (0.50x - 1.20x)
14. Network connections (1.08x)
15. Previous interview score (0.70x - 1.15x)
16. Certification count (1.05x - 1.10x)
17. Project count (1.06x - 1.12x)
18. GitHub activity (1.08x - 1.15x)
19. Publication count (1.05x - 1.10x)
20. Open to relocation (1.05x)
21. Salary match (0.80x - 1.08x)
22. Notice period (0.85x - 1.10x)
23. Education match (0.85x - 1.15x)

## Performance

### Runtime Constraints

| Metric | Target | Actual |
|--------|--------|--------|
| Total Time | < 5 min | ~90 sec |
| Candidates/sec | N/A | ~870 |
| Max Memory | < 4 GB | ~2.1 GB |
| CPU Only | Yes | Yes |
| No Network | Yes | Yes |

### Bottleneck Analysis

- **Stage 1**: Filter throughput ~12K candidates/sec
- **Stage 2**: Vectorized NumPy ~50K candidates/sec (after filtering)

## Docker Execution

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Precomputation
RUN python -m src.precompute \
  --input /data/candidates.jsonl.gz \
  --output precomputed_pool.npz

# Ranking
RUN python rank.py \
  --candidates precomputed_pool.npz \
  --output submission.csv
```

## Troubleshooting

### Memory Issues

**Problem**: "MemoryError" during embedding

**Solution**: Reduce batch size:
```bash
python -m src.precompute --batch-size 8
```

### Slow Execution

**Problem**: Taking longer than 5 minutes

**Solution**: Profile bottlenecks:
```bash
python -m cProfile -s cumtime rank.py --candidates precomputed_pool.npz --output output.csv
```

### Missing Embeddings

**Problem**: "sentence-transformers not found"

**Solution**: Install torch first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

## Configuration

Edit `submission_metadata.yaml` to customize:

```yaml
team_id: "your-team-id"
max_runtime_seconds: 300
output_format:
  columns:
    - candidate_id
    - rank
    - score
    - reasoning
  top_k_candidates: 100
```

## References

- **Sentence Transformers**: https://www.sbert.net/
- **NumPy Vectorization**: https://numpy.org/doc/stable/user/basics.broadcasting.html
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity

## License

ProprietarySoftware for Redrob Hackathon - 2026

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review sample input formats in `examples/`
3. Contact: prabhnoors.386@gmail.com
