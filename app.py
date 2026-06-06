import json

from flask import Flask, flash, redirect, render_template, request, session, url_for

from agents.registry import AGENT_ROUTES, AGENT_TEMPLATES, AGENTS, get_agent

ROUTE_ENDPOINTS = {
    "form": "agent_form",
    "dm": "agent_dm",
    "refmail": "agent_refmail",
    "jobemail": "agent_jobemail",
}

from config import FLASK_DEBUG, FLASK_SECRET_KEY
from user_profile.profile_store import load_profile, profile_is_populated, save_profile
from rag.embedder import index_profile

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY


def _parse_list_field(value: str) -> list[str]:
    if not value or not value.strip():
        return []
    return [line.strip() for line in value.strip().split("\n") if line.strip()]


def _form_to_profile(form) -> dict:
    profile = load_profile()

    profile["personal_info"] = {
        "full_name": form.get("full_name", ""),
        "email": form.get("email", ""),
        "phone": form.get("phone", ""),
        "linkedin_url": form.get("linkedin_url", ""),
        "github_url": form.get("github_url", ""),
        "portfolio_url": form.get("portfolio_url", ""),
        "location": form.get("location", ""),
    }
    profile["compensation"] = {
        "current_ctc": form.get("current_ctc", ""),
        "expected_ctc": form.get("expected_ctc", ""),
        "notice_period": form.get("notice_period", ""),
        "years_of_experience": form.get("years_of_experience", ""),
    }
    profile["summary"] = form.get("summary", "")

    profile["skills"] = {
        "programming_languages": _parse_list_field(form.get("programming_languages", "")),
        "frameworks_and_libraries": _parse_list_field(form.get("frameworks_and_libraries", "")),
        "databases": _parse_list_field(form.get("databases", "")),
        "cloud_and_devops": _parse_list_field(form.get("cloud_and_devops", "")),
        "ai_ml": _parse_list_field(form.get("ai_ml", "")),
        "tools": _parse_list_field(form.get("tools", "")),
        "soft_skills": _parse_list_field(form.get("soft_skills", "")),
    }

    profile["achievements_and_awards"] = _parse_list_field(form.get("achievements_and_awards", ""))

    education_json = form.get("education_json", "[]")
    work_json = form.get("work_experience_json", "[]")
    projects_json = form.get("projects_json", "[]")
    publications_json = form.get("publications_json", "[]")
    certifications_json = form.get("certifications_json", "[]")

    try:
        profile["education"] = json.loads(education_json)
        profile["work_experience"] = json.loads(work_json)
        profile["projects"] = json.loads(projects_json)
        profile["publications"] = json.loads(publications_json)
        profile["certifications"] = json.loads(certifications_json)
    except json.JSONDecodeError:
        flash("Invalid JSON in one of the list sections. Profile not saved.", "error")
        return profile

    return profile


@app.route("/")
def index():
    return redirect(url_for("profile_page"))


@app.route("/profile", methods=["GET", "POST"])
def profile_page():
    if request.method == "POST":
        profile = _form_to_profile(request.form)
        save_profile(profile)
        try:
            chunk_count = index_profile(profile)
            flash(f"Profile saved and indexed ({chunk_count} chunks).", "success")
        except Exception as exc:
            flash(f"Profile saved but indexing failed: {exc}", "error")
        return redirect(url_for("profile_page"))

    profile = load_profile()
    return render_template(
        "profile.html",
        profile=profile,
        profile_json=json.dumps(profile, indent=2),
        is_populated=profile_is_populated(profile),
    )


def _agent_inputs(agent_id: str, form) -> dict:
    field_map = {
        "form": ["form_questions", "job_description", "company_name"],
        "dm": ["recipient_name", "recipient_role", "company", "context", "job_posting_link"],
        "refmail": [
            "contact_name",
            "contact_role",
            "company_name",
            "job_title",
            "job_link",
            "job_description",
            "relationship",
        ],
        "jobemail": [
            "recipient_name",
            "company_name",
            "job_title",
            "job_description",
        ],
    }
    return {f: form.get(f, "") for f in field_map.get(agent_id, [])}


def _handle_agent(agent_id: str):
    agent = get_agent(agent_id)
    if not agent:
        flash("Unknown agent.", "error")
        return redirect(url_for("profile_page"))

    if request.method == "GET":
        return render_template(
            AGENT_TEMPLATES[agent_id],
            agent=agent,
            inputs=session.get(f"inputs_{agent_id}", {}),
        )

    inputs = _agent_inputs(agent_id, request.form)
    action = request.form.get("action", "generate")

    if action == "refine":
        previous = session.get("agent_output", "")
        feedback = request.form.get("refine_feedback", "")
        if not previous or not feedback:
            flash("Provide refinement feedback.", "error")
            return _render_result(agent, agent_id, previous, inputs)
        try:
            output = agent.refine(inputs, previous, feedback)
            session["agent_output"] = output
            session[f"inputs_{agent_id}"] = inputs
        except Exception as exc:
            flash(f"Refinement failed: {exc}", "error")
            output = previous
        return _render_result(agent, agent_id, output, inputs)

    try:
        output = agent.generate(inputs)
        session["agent_output"] = output
        session["agent_id"] = agent_id
        session[f"inputs_{agent_id}"] = inputs
    except Exception as exc:
        flash(f"Generation failed: {exc}", "error")
        return render_template(AGENT_TEMPLATES[agent_id], agent=agent, inputs=inputs)

    return _render_result(agent, agent_id, output, inputs)


def _render_result(agent, agent_id, output, inputs):
    return render_template(
        "result.html",
        agent=agent,
        output=output,
        inputs=inputs,
        back_url=url_for(ROUTE_ENDPOINTS[agent_id]),
    )


@app.route("/agent/form", methods=["GET", "POST"])
def agent_form():
    return _handle_agent("form")


@app.route("/agent/dm", methods=["GET", "POST"])
def agent_dm():
    return _handle_agent("dm")


@app.route("/agent/refmail", methods=["GET", "POST"])
def agent_refmail():
    return _handle_agent("refmail")


@app.route("/agent/jobemail", methods=["GET", "POST"])
def agent_jobemail():
    return _handle_agent("jobemail")


@app.context_processor
def inject_nav():
    return {
        "nav_agents": [
            {"id": aid, "name": AGENTS[aid].agent_name, "route": AGENT_ROUTES[aid]}
            for aid in AGENTS
        ]
    }


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, port=5000)