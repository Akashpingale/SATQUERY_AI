"""Downloadable analysis reports.

A report is a deterministic, self-contained HTML rendering of a *persisted* analysis record.
It contains only real, stored data - the query, the agent's task/tool selection, the answer,
the confidence and its signals, the evidence (with images embedded), and the observable trace.
Nothing is fabricated, and private chain-of-thought is never included.
"""

from .html_report import build_report_filename, render_analysis_report

__all__ = ["build_report_filename", "render_analysis_report"]
