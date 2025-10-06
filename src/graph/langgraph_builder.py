# src/graph/langgraph_builder.py
from __future__ import annotations
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.agents.commander import CommanderAgent
from src.agents.rationalist import RationalistAgent
from src.agents.dramatist import DramatistAgent
from src.agents.synthesizer import SynthesizerAgent

# ---------- helpers ----------
ROLE_ORDER = ["Rationalist", "Commander", "Dramatist"]

def _parse_target(text: str) -> str | None:
    """Detect direct mentions like @Commander in agent messages."""
    low = (text or "").lower()
    if "@commander" in low:   return "Commander"
    if "@rationalist" in low: return "Rationalist"
    if "@dramatist" in low:   return "Dramatist"
    return None

def _get_agent(role: str, C, R, D):
    """Return the corresponding agent object by role name."""
    return {"Commander": C, "Rationalist": R, "Dramatist": D}[role]

def _safe_call(fn, fallback: str) -> Dict[str, Any]:
    """
    Execute an agent method safely; if it errors or yields empty, return a minimal fallback.
    """
    try:
        out = fn() or {}
        text = (out.get("response") or "").strip()
        if not text:
            return {"response": fallback, "citations": "", "hits": []}
        return out
    except Exception:
        return {"response": fallback, "citations": "", "hits": []}

# ---------- node implementations ----------
def init_state_node(state: GraphState) -> GraphState:
    """Initialize all agents and shared conversation state."""
    state["_agents"] = {
        "C": CommanderAgent(),
        "R": RationalistAgent(),
        "D": DramatistAgent(),
        "S": SynthesizerAgent(),
    }
    state["round1"] = {"commander": {}, "rationalist": {}, "dramatist": {}}
    state["dialogue"] = []
    state["challenges"] = {}
    state["thread"] = []
    state["role_order"] = ROLE_ORDER[:]
    state["pending_target"] = None
    state["rotation_index"] = 0
    state["phase"] = "round1"

    # normalize dialogue_rounds to int (allow env override for demos)
    env_rounds = os.getenv("DIALOGUE_ROUNDS")
    rounds = state.get("dialogue_rounds", 2)
    if env_rounds:
        try:
            rounds = int(env_rounds)
        except Exception:
            pass
    try:
        rounds = int(rounds)
    except Exception:
        rounds = 2

    state["turns_remaining"] = max(1, rounds) * len(ROLE_ORDER)
    return state

def round1_node(state: GraphState) -> GraphState:
    """Each agent independently provides their initial perspective."""
    agents = state["_agents"]
    query = state["query"]

    r1_commander = _safe_call(lambda: agents["C"].respond(query),
                              "I’ll keep this practical. Start with one concrete move and a quick checkpoint.")
    r1_rationalist = _safe_call(lambda: agents["R"].respond(query),
                                "Let’s name the assumptions, what would falsify them, and decide based on that.")
    r1_dramatist = _safe_call(lambda: agents["D"].respond(query),
                              "There’s a real tension here; acknowledge it, then choose a next beat you can own.")

    state["round1"] = {
        "commander":   r1_commander,
        "rationalist": r1_rationalist,
        "dramatist":   r1_dramatist,
    }

    state["thread"] = [
        {"speaker": "Commander",   "message": r1_commander["response"],   "citations": r1_commander.get("citations", "")},
        {"speaker": "Rationalist", "message": r1_rationalist["response"], "citations": r1_rationalist.get("citations", "")},
        {"speaker": "Dramatist",   "message": r1_dramatist["response"],   "citations": r1_dramatist.get("citations", "")},
    ]

    state["phase"] = "dialogue"
    return state

def dialogue_node(state: GraphState) -> GraphState:
    """Simulate agent-to-agent conversation: questions, responses, and insights."""
    if state["turns_remaining"] <= 0:
        state["phase"] = "challenges"
        return state

    agents = state["_agents"]
    query = state["query"]
    thread = state["thread"]

    # targeted reply (if mentioned) or round-robin fallback
    pending = state.get("pending_target")
    if pending:
        role = pending
        state["pending_target"] = None
    else:
        idx = state["rotation_index"] % len(state["role_order"])
        role = state["role_order"][idx]
        state["rotation_index"] += 1

    agent = _get_agent(role, agents["C"], agents["R"], agents["D"])
    msg = _safe_call(lambda: agent.converse(query, thread),
                     "Noted. One clear tension and a small next step to move us forward.")
    text = (msg.get("response") or "").strip()
    if text:
        turn = {
            "speaker": role,
            "message": text,
            "citations": msg.get("citations", "")
        }
        state["dialogue"].append(turn)
        thread.append({"speaker": role, "message": text, "citations": turn["citations"]})

        tgt = _parse_target(text)
        if tgt and tgt != role:
            state["pending_target"] = tgt
        # only decrement if we produced a turn
        state["turns_remaining"] -= 1

    if state["turns_remaining"] <= 0:
        state["phase"] = "challenges"
    return state

def challenges_node(state: GraphState) -> GraphState:
    """Challenge round: agents critique and respond to each other."""
    agents = state["_agents"]
    r1 = state["round1"]

    ch_r_on_c = _safe_call(lambda: agents["R"].challenge(r1["commander"]["response"]),
                           "Explicit assumption, concrete test, fallback path.")
    ch_r_on_d = _safe_call(lambda: agents["R"].challenge(r1["dramatist"]["response"]),
                           "Name the premise, propose a falsifier, and a backup route.")
    rebut_m   = _safe_call(lambda: agents["C"].rebuttal(r1["rationalist"]["response"]),
                           "Fair point noted. I’ll narrow scope and set a quick check-in.")
    recon_d   = _safe_call(lambda: agents["D"].reconcile(r1["commander"]["response"], r1["rationalist"]["response"]),
                           "Shared backbone, live tension, one next beat we agree on.")

    state["challenges"] = {
        "rationalist_on_commander": ch_r_on_c,
        "rationalist_on_dramatist": ch_r_on_d,
        "commander_rebuttal": rebut_m,
        "dramatist_reconcile": recon_d,
    }
    state["phase"] = "synthesis"
    return state

def synthesis_node(state: GraphState) -> GraphState:
    """Synthesizer creates unified insight from all prior discussion."""
    agents = state["_agents"]
    query = state["query"]

    persona_msgs = [
        ("commander",   state["round1"]["commander"]),
        ("rationalist", state["round1"]["rationalist"]),
        ("dramatist",   state["round1"]["dramatist"]),
        *[(d["speaker"].lower(), {"response": d["message"], "citations": d.get("citations", "")})
          for d in state["dialogue"]],
    ]
    final = _safe_call(lambda: agents["S"].synthesize(query, persona_msgs),
                       "Consensus: one practical path, one test for success, and next steps.")
    state["synthesis"] = final
    state["phase"] = "done"
    return state

def router(state: GraphState) -> str:
    """
    Control flow logic — returns keys that match add_conditional_edges mappings:
      'dialogue' | 'challenges' | 'synthesis' | 'done'
    """
    phase = state.get("phase", "")
    if phase == "round1":
        return "dialogue"    # after round1 we always go to dialogue
    if phase == "dialogue":
        return "dialogue"
    if phase == "challenges":
        return "challenges"
    if phase == "synthesis":
        return "synthesis"
    return "done"

def build_graph():
    """Builds and compiles the LangGraph workflow."""
    g = StateGraph(GraphState)

    g.add_node("init",       init_state_node)
    g.add_node("round1",     round1_node)
    g.add_node("dialogue",   dialogue_node)
    g.add_node("challenges", challenges_node)
    g.add_node("synthesis",  synthesis_node)

    g.set_entry_point("init")

    # init -> round1
    g.add_conditional_edges("init", lambda s: "round1")

    # router-controlled transitions
    g.add_conditional_edges("round1", router, {
        "dialogue": "dialogue",
        "challenges": "challenges",
        "synthesis": "synthesis",
        "done": END,
    })
    g.add_conditional_edges("dialogue", router, {
        "dialogue": "dialogue",
        "challenges": "challenges",
        "synthesis": "synthesis",
        "done": END,
    })
    g.add_conditional_edges("challenges", router, {
        "synthesis": "synthesis",
        "done": END,
    })
    g.add_conditional_edges("synthesis", router, {
        "done": END,
    })

    return g.compile()