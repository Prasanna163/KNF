import datetime as _dt
import streamlit as st


def init_state():
    st.session_state.setdefault("analysis_history", [])
    st.session_state.setdefault("prepared_context", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("last_xtb_log", None)
    st.session_state.setdefault("last_cmd", None)
    st.session_state.setdefault("optimized_xyz", None)
    st.session_state.setdefault("workflow_results", [])


def clear_session():
    keys = [
        "prepared_context",
        "last_error",
        "last_xtb_log",
        "last_cmd",
        "optimized_xyz",
        "workflow_results",
    ]
    for k in keys:
        st.session_state.pop(k, None)


def record_run(name: str, mode: str, status: str, duration_seconds: float | None = None):
    history = st.session_state.setdefault("analysis_history", [])
    history.append(
        {
            "name": name,
            "mode": mode,
            "status": status,
            "duration_seconds": duration_seconds,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        }
    )


def summary_metrics():
    history = st.session_state.get("analysis_history", [])
    total = len(history)
    if total == 0:
        return {
            "total": 0,
            "success_rate": 0.0,
            "avg_time": 0.0,
            "last_run": "N/A",
        }

    success = sum(1 for h in history if h.get("status") == "success")
    durations = [h["duration_seconds"] for h in history if h.get("duration_seconds") is not None]
    avg_time = sum(durations) / len(durations) if durations else 0.0
    return {
        "total": total,
        "success_rate": (success / total) * 100.0,
        "avg_time": avg_time,
        "last_run": history[-1]["timestamp"],
    }

