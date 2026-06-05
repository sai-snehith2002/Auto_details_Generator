from agents.agent_cold_dm import ColdDMAgent
from agents.agent_form_fill import FormFillAgent
from agents.agent_job_email import JobEmailAgent
from agents.agent_referral_email import ReferralEmailAgent
from agents.base_agent import BaseAgent

AGENTS: dict[str, BaseAgent] = {
    "form": FormFillAgent(),
    "dm": ColdDMAgent(),
    "refmail": ReferralEmailAgent(),
    "jobemail": JobEmailAgent(),
}

AGENT_ROUTES = {
    "form": "/agent/form",
    "dm": "/agent/dm",
    "refmail": "/agent/refmail",
    "jobemail": "/agent/jobemail",
}

AGENT_TEMPLATES = {
    "form": "agent_form_fill.html",
    "dm": "agent_cold_dm.html",
    "refmail": "agent_referral_email.html",
    "jobemail": "agent_job_email.html",
}


def get_agent(agent_id: str) -> BaseAgent | None:
    return AGENTS.get(agent_id)
