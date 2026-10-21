"""Tests for the TEER project roadmap implementation."""

import os
import tempfile
from pathlib import Path

def test_plan_roadmap_content():
    """Test that PLAN.md contains proper roadmap structure."""
    plan_path = Path("PLAN.md")
    assert plan_path.exists(), "PLAN.md should exist"
    
    content = plan_path.read_text()
    assert "# TEER Project Plan" in content
    assert "## Objective" in content
    assert "## Milestone" in content
    assert "## Tasks" in content
    assert "## Success Criteria" in content
    
def test_progress_updated():
    """Test that PROGRESS.md has been updated with roadmap milestone."""
    progress_path = Path("PROGRESS.md")
    assert progress_path.exists(), "PROGRESS.md should exist"
    
    content = progress_path.read_text()
    # Check for roadmap-related entries
    assert "roadmap" in content.lower() or "plan" in content.lower()
