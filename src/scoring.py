"""Stage 2: Semantic Scoring and Behavioral Signal Multipliers.

This module implements the fast inference orchestration using vectorized
NumPy arrays and behavioral signal multipliers for final candidate ranking.
"""

import numpy as np
from typing import Optional, Dict


def calculate_semantic_score(
    candidate_vector: Optional[np.ndarray], 
    jd_vector: Optional[np.ndarray]
) -> float:
    """Calculate cosine similarity between candidate and job description embeddings.
    
    Uses vectorized NumPy operations for fast computation. Returns normalized
    cosine similarity in range [0, 1].
    
    Args:
        candidate_vector: Embedding vector for candidate profile (shape: (embedding_dim,))
        jd_vector: Embedding vector for job description (shape: (embedding_dim,))
        
    Returns:
        float: Cosine similarity score in range [0, 1]
    """
    if candidate_vector is None or jd_vector is None:
        return 0.0
    
    # Calculate dot product
    dot_product = np.dot(candidate_vector, jd_vector)
    
    # Calculate L2 norms
    norm_cand = np.linalg.norm(candidate_vector)
    norm_jd = np.linalg.norm(jd_vector)
    
    # Handle zero vectors
    if norm_cand == 0 or norm_jd == 0:
        return 0.0
    
    # Return cosine similarity
    cosine_sim = float(dot_product / (norm_cand * norm_jd))
    
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, cosine_sim))


def apply_behavioral_multipliers(base_score: float, signals: Dict) -> float:
    """Apply Redrob's behavioral signals as exact multipliers to base score.
    
    Applies 23 distinct behavioral signal multipliers that evaluate real hiring
    availability and candidate quality beyond semantic similarity.
    
    Behavioral signals include:
    1. Recruiter response rate
    2. Profile activity (months since last login)
    3. Profile completion percentage
    4. Interview completion rate
    5. Profile view count
    6. Application count (last 30 days)
    7. Interview request response time
    8. Skills endorsement count
    9. Recommendation count
    10. Profile completeness (photo, bio, links)
    11. Job title match specificity
    12. Years of experience match
    13. Skills overlap percentage
    14. Network connection count
    15. Previous interview performance
    16. Certification count
    17. Project count on portfolio
    18. GitHub activity (commits)
    19. Publication count
    20. Open to relocation
    21. Salary expectation match
    22. Notice period alignment
    23. Education level match
    
    Args:
        base_score: Base semantic similarity score (typically 0-1)
        signals: Dictionary mapping signal names to values
        
    Returns:
        float: Final score after applying all behavioral multipliers
    """
    multiplier = 1.0
    
    # Signal 1: Recruiter Response Rate
    recruiter_response_rate = signals.get("recruiter_response_rate", 1.0)
    if recruiter_response_rate < 0.15:
        multiplier *= 0.30
    elif recruiter_response_rate < 0.50:
        multiplier *= 0.75
    
    # Signal 2: Months Since Last Login (activity indicator)
    months_since_last_login = signals.get("months_since_last_login", 0)
    if months_since_last_login > 6:
        multiplier *= 0.40
    elif months_since_last_login > 3:
        multiplier *= 0.80
    
    # Signal 3: Profile Completion Percentage
    profile_completion_pct = signals.get("profile_completion_percentage", 100)
    if profile_completion_pct < 50:
        multiplier *= 0.50
    
    # Signal 4: Interview Completion Rate
    interview_completion_rate = signals.get("interview_completion_rate", 1.0)
    if interview_completion_rate < 0.30:
        multiplier *= 0.20
    
    # Signal 5: Profile View Count (normalized)
    profile_view_count = signals.get("profile_view_count", 0)
    if profile_view_count > 100:
        multiplier *= 1.15
    elif profile_view_count > 50:
        multiplier *= 1.08
    
    # Signal 6: Application Count (within last 30 days)
    applications_last_30_days = signals.get("applications_last_30_days", 0)
    if applications_last_30_days > 10:
        multiplier *= 1.12
    elif applications_last_30_days > 5:
        multiplier *= 1.05
    
    # Signal 7: Interview Request Response Time (days)
    interview_response_time_days = signals.get("interview_response_time_days", 5)
    if interview_response_time_days <= 1:
        multiplier *= 1.10
    elif interview_response_time_days > 7:
        multiplier *= 0.85
    
    # Signal 8: Skills Endorsement Count
    skills_endorsement_count = signals.get("skills_endorsement_count", 0)
    if skills_endorsement_count > 50:
        multiplier *= 1.10
    elif skills_endorsement_count > 20:
        multiplier *= 1.05
    
    # Signal 9: Recommendation Count
    recommendation_count = signals.get("recommendation_count", 0)
    if recommendation_count >= 5:
        multiplier *= 1.12
    elif recommendation_count > 0:
        multiplier *= 1.05
    
    # Signal 10: Profile Completeness (photo, bio, verified links)
    has_profile_photo = signals.get("has_profile_photo", False)
    has_bio = signals.get("has_bio", False)
    has_verified_links = signals.get("has_verified_links", False)
    
    completeness_items = sum([has_profile_photo, has_bio, has_verified_links])
    if completeness_items == 3:
        multiplier *= 1.10
    elif completeness_items == 2:
        multiplier *= 1.05
    elif completeness_items == 0:
        multiplier *= 0.70
    
    # Signal 11: Job Title Match Specificity
    job_title_relevance_score = signals.get("job_title_relevance_score", 0.5)
    multiplier *= (0.8 + (job_title_relevance_score * 0.4))
    
    # Signal 12: Years of Experience Match
    experience_match_score = signals.get("experience_match_score", 0.5)
    multiplier *= (0.9 + (experience_match_score * 0.2))
    
    # Signal 13: Skills Overlap Percentage
    skills_overlap_percentage = signals.get("skills_overlap_percentage", 0.0)
    if skills_overlap_percentage >= 0.80:
        multiplier *= 1.20
    elif skills_overlap_percentage >= 0.60:
        multiplier *= 1.10
    elif skills_overlap_percentage >= 0.40:
        multiplier *= 1.05
    elif skills_overlap_percentage < 0.20:
        multiplier *= 0.50
    
    # Signal 14: Network Connection Count
    network_connections = signals.get("network_connections", 0)
    if network_connections > 500:
        multiplier *= 1.08
    
    # Signal 15: Previous Interview Performance Score
    previous_interview_score = signals.get("previous_interview_score", 0.5)
    if previous_interview_score > 0.7:
        multiplier *= 1.15
    elif previous_interview_score < 0.3:
        multiplier *= 0.70
    
    # Signal 16: Certification Count
    certification_count = signals.get("certification_count", 0)
    if certification_count >= 5:
        multiplier *= 1.10
    elif certification_count >= 2:
        multiplier *= 1.05
    
    # Signal 17: Project Count on Portfolio
    project_count = signals.get("project_count", 0)
    if project_count >= 10:
        multiplier *= 1.12
    elif project_count >= 5:
        multiplier *= 1.06
    
    # Signal 18: GitHub Activity (commits in last 3 months)
    github_commits_3m = signals.get("github_commits_3m", 0)
    if github_commits_3m > 50:
        multiplier *= 1.15
    elif github_commits_3m > 10:
        multiplier *= 1.08
    
    # Signal 19: Publication Count (research papers, articles)
    publication_count = signals.get("publication_count", 0)
    if publication_count >= 3:
        multiplier *= 1.10
    elif publication_count >= 1:
        multiplier *= 1.05
    
    # Signal 20: Open to Relocation
    open_to_relocation = signals.get("open_to_relocation", False)
    if open_to_relocation:
        multiplier *= 1.05
    
    # Signal 21: Salary Expectation Match
    salary_match_score = signals.get("salary_match_score", 0.5)
    if salary_match_score > 0.8:
        multiplier *= 1.08
    elif salary_match_score < 0.3:
        multiplier *= 0.80
    
    # Signal 22: Notice Period Alignment (days)
    notice_period_days = signals.get("notice_period_days", 30)
    if notice_period_days <= 7:
        multiplier *= 1.10
    elif notice_period_days > 60:
        multiplier *= 0.85
    
    # Signal 23: Education Level Match
    education_match_score = signals.get("education_match_score", 0.5)
    multiplier *= (0.85 + (education_match_score * 0.3))
    
    return base_score * multiplier
