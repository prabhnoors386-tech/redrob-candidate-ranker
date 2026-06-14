"""Stage 1: Structural Validation and Honeypot Detection.

This module implements honeypot detection and strategic fit validation
to eliminate low-quality candidates before semantic scoring.
"""

import json
from datetime import datetime
from typing import Tuple

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", 
    "accenture", "cognizant", "capgemini", "hcl", "tech mahindra",
    "deloitte", "pwc", "kpmg", "ey", "atos", "fujitsu", "ibm"
}

RESEARCH_TERMS = {
    "academic lab", "research fellow", "postdoc", "postdoctoral", 
    "phd student", "graduate research assistant", "pure research",
    "research intern", "visiting scholar", "research scientist"
}

NON_TECH_TRAPS = {
    "marketing manager", "sales executive", "recruiter", 
    "hr specialist", "content writer", "business development",
    "project manager", "product manager", "account manager",
    "sales manager", "team lead", "scrum master"
}


def is_honeypot(candidate: dict) -> bool:
    """Detect honeypot profiles with inconsistent chronological data.
    
    Honeypots are identified by:
    - Negative employment duration
    - Employment start before company founding
    - Claiming 10+ expert skills with no years of experience
    - Logical inconsistencies in education/experience timeline
    
    Args:
        candidate: Candidate profile dictionary
        
    Returns:
        bool: True if profile is detected as honeypot, False otherwise
    """
    experience = candidate.get("experience", [])
    skills = candidate.get("skills", [])
    
    total_claimed_years = 0
    
    # Check experience chronology
    for job in experience:
        start_year = job.get("start_year")
        end_year = job.get("end_year") or datetime.now().year
        company_founded = job.get("company_founded_year")
        
        if start_year and end_year:
            duration = end_year - start_year
            
            # Negative duration is impossible
            if duration < 0:
                return True
            
            total_claimed_years += duration
            
            # Started before company was founded
            if company_founded and start_year < company_founded:
                return True
    
    # Check skill expertise claims
    expert_count = 0
    for skill in skills:
        level = skill.get("proficiency", "").lower()
        years_used = skill.get("years_used", 0)
        
        if level == "expert":
            expert_count += 1
            # Expert claim with zero years is impossible
            if years_used <= 0:
                return True
    
    # More than 10 expert skills is suspicious
    if expert_count >= 10:
        return True
    
    # Check education consistency
    education = candidate.get("education", [])
    if education:
        for edu in education:
            start_year = edu.get("start_year")
            end_year = edu.get("end_year")
            if start_year and end_year and end_year < start_year:
                return True
    
    return False


def check_strategic_fit(candidate: dict) -> Tuple[bool, str]:
    """Validate strategic fit based on work history and profile characteristics.
    
    Filters out:
    - Profiles with no work history
    - Current roles in non-technical positions (keyword stuffers)
    - Candidates with only consulting/IT services experience
    - Pure research profiles with limited industry experience
    
    Args:
        candidate: Candidate profile dictionary
        
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    experience = candidate.get("experience", [])
    if not experience:
        return False, "Empty work history"

    # Check current role for non-tech indicators
    primary_title = experience[0].get("title", "").lower() if experience else ""
    if any(trap in primary_title for trap in NON_TECH_TRAPS):
        return False, "Keyword stuffer trap detected via non-technical current title"

    # Validate company list
    companies = [
        job.get("company_name", "").lower().strip() 
        for job in experience if job.get("company_name")
    ]
    if not companies:
        return False, "No valid companies listed"

    # Check for pure consulting background
    is_pure_consulting = all(
        any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies
    )
    if is_pure_consulting:
        return False, "Disqualified: Entire career spent at consulting/IT services firms"

    # Check for pure research background
    all_titles = " ".join([job.get("title", "").lower() for job in experience])
    if any(term in all_titles for term in RESEARCH_TERMS):
        if is_pure_consulting or len(companies) <= 1:
            return False, "Disqualified: Profile reflects pure research environments"

    return True, "Valid Candidate Profile"


def validate_candidate_data(candidate: dict) -> Tuple[bool, str]:
    """Perform comprehensive data validation on candidate profile.
    
    Args:
        candidate: Candidate profile dictionary
        
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    # Check required fields
    if not candidate.get("id"):
        return False, "Missing candidate ID"
    
    # Run honeypot check
    if is_honeypot(candidate):
        return False, "Honeypot profile detected"
    
    # Run strategic fit check
    is_valid, reason = check_strategic_fit(candidate)
    if not is_valid:
        return False, reason
    
    return True, "Valid candidate"
