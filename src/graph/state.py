# src/graph/state.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class Message(TypedDict):
    speaker: str        # "Commander" | "Rationalist" | "Dramatist"
    message: str        # text
    citations: Optional[str]


class Challenges(TypedDict, total=False):
    rationalist_on_commander: Dict[str, Any]
    rationalist_on_dramatist: Dict[str, Any]
    commander_rebuttal: Dict[str, Any]
    dramatist_reconcile: Dict[str, Any]


class Round1(TypedDict, total=False):
    commander: Dict[str, Any]
    rationalist: Dict[str, Any]
    dramatist: Dict[str, Any]


class GraphState(TypedDict, total=False):
    # inputs
    query: str
    dialogue_rounds: int

    # outputs we’re building
    round1: Round1
    dialogue: List[Message]
    challenges: Challenges
    synthesis: Dict[str, Any]

    # internal working memory
    thread: List[Message]           # public transcript (speaker/message)
    role_order: List[str]           # fixed order for live dialogue
    pending_target: Optional[str]   # next targeted speaker (from @mentions)
    rotation_index: int             # pointer for round-robin
    phase: str                      # "round1"|"dialogue"|"challenges"|"synthesis"|"done"
    turns_remaining: int

    # agent bag (created in init node)
    _agents: Dict[str, Any]