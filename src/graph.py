from langgraph.graph import StateGraph, END
from src.state import ReviewState
from src.timing import timed
from src.agents.planner import planner
from src.agents.search_strategy import search_strategy
from src.agents.retrieval import retrieval
from src.agents.screening import screening
from src.agents.extraction import extraction
from src.agents.synthesis import synthesis
from src.agents.critic import critic
from src.agents.report import report


def route_after_critic(state):
    """Back to synthesis while the critic requests a revision (the critic caps the number of passes)."""
    return "synthesis" if state.get("needs_revision") else "report"


def _add_tail(g):
    g.add_node("synthesis", timed("synthesis", synthesis))
    g.add_node("critic", timed("critic", critic))
    g.add_node("report", timed("report", report))
    g.add_edge("synthesis", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"synthesis": "synthesis", "report": "report"})
    g.add_edge("report", END)


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("planner", timed("planner", planner))
    g.add_node("search_strategy", timed("search_strategy", search_strategy))
    g.add_node("retrieval", timed("retrieval", retrieval))
    g.add_node("screening", timed("screening", screening))
    g.add_node("extraction", timed("extraction", extraction))
    g.set_entry_point("planner")
    g.add_edge("planner", "search_strategy")
    g.add_edge("search_strategy", "retrieval")
    g.add_edge("retrieval", "screening")
    g.add_edge("screening", "extraction")
    g.add_edge("extraction", "synthesis")
    _add_tail(g)
    return g.compile()


def build_synthesis_graph(entry="synthesis"):
    """synthesis -> critic (-> revisions) -> report on an existing run. entry='critic' reuses synthesis.json."""
    g = StateGraph(ReviewState)
    g.set_entry_point(entry)
    _add_tail(g)
    return g.compile()
