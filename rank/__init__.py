"""Rank stage: score every class against every CQ and write rankings/{run}__{view}.csv (CPU only)."""

from .rank import run_rank, write_rankings
