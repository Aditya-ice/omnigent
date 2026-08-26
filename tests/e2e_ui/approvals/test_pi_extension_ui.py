r"""E2E: a Pi extension's ``ctx.ui`` dialog renders as a card (mock, no real Pi).

A Pi extension that calls ``ctx.ui.confirm|select|input|editor`` used to hang:
the pi-native wrap draws the dialog in a TUI nobody is attached to. The wrap now
POSTs it to ``POST /v1/sessions/{session_id}/hooks/pi-extension-ui``; the server
parks the elicitation and the SPA renders it on the existing ApprovalCard /
AskUserQuestionForm, returning the answer so the extension unblocks.

A background thread POSTs the hook directly — the same shortcut
``test_ask_user_question.py`` takes for Claude's question tool — so these run in
seconds and need no ``pi`` binary, tmux, or model credentials. The sibling
``test_cursor_native_approval.py`` covers the real-CLI shape for cursor-native
and is nightly-gated for exactly that reason.

Three behaviours, one per test:

- ``select`` stamps ``ask_user_question`` → the form renders the extension's own
  options and the chosen label flows back.
- ``confirm`` is a binary card → Approve returns ``accept``.
- Reject returns a verdict rather than erroring, which is what lets the caller
  hand Pi its cancelled value without killing the turn.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest
from playwright.sync_api import Page, expect

_log = logging.getLogger(__name__)

_APPROVAL_CARD = '[data-testid="approval-card"]'
_FORM = '[data-testid="ask-user-question-form"]'
_SUBMIT = '[data-testid="ask-user-question-submit"]'

_MOCK_ELICITATION_TIMEOUT_MS = 15_000

# The permission-gate shape a Pi extension uses: the tool asks before acting.
_SELECT_TITLE = 'deploy_prod requests approval for build "v2.4.0"'
_ALLOW_ONCE = "Allow once"
_DENY = "Deny"
_CONFIRM_TITLE = "Confirm production deploy"
_CONFIRM_MESSAGE = "Really deploy v2.4.0 to production right now?"


def _pending_elicitations(base_url: str, session_id: str) -> list[dict]:
    """Return the session snapshot's pending elicitation events (owner view)."""
    resp = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=10.0)
    resp.raise_for_status()
    return resp.json().get("pending_elicitations") or []


def _wait_for(predicate, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> None:
    """Poll *predicate* until truthy or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError("condition not met within timeout")


def _post_extension_ui(
    base_url: str,
    session_id: str,
    request: dict,
    holder: dict,
) -> threading.Thread:
    """Park a Pi extension UI dialog on *session_id*.

    The hook blocks server-side until the web verdict lands — that block is the
    extension waiting on its ``ctx.ui`` promise — so it runs on its own thread;
    join it after answering the card.

    :param base_url: Server base URL.
    :param session_id: Session to raise the dialog on.
    :param request: The Pi ``extension_ui_request`` fields (``method``, ``title``,
        and whichever of ``message`` / ``options`` that method carries).
    :param holder: Dict the thread writes ``response`` / ``error`` into.
    :returns: The started thread.
    """
    elicitation_id = f"elicit_pi_{request['method']}_{'cd' * 12}"

    def _post() -> None:
        try:
            resp = httpx.post(
                f"{base_url}/v1/sessions/{session_id}/hooks/pi-extension-ui",
                json={"elicitation_id": elicitation_id, "request": request},
                timeout=60.0,
            )
            resp.raise_for_status()
            holder["response"] = resp.json()
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_post, daemon=True)
    thread.start()
    return thread


def _join_hook(thread: threading.Thread, holder: dict) -> dict:
    """Join the hook thread and return its JSON verdict."""
    thread.join(timeout=30)
    if "error" in holder:
        error = holder["error"]
        raise AssertionError(f"pi extension UI hook failed: {error}") from error
    assert "response" in holder, "pi extension UI hook returned no verdict"
    return holder["response"]


@pytest.mark.timeout(90)
def test_pi_extension_select_renders_options_and_returns_the_choice(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """``ctx.ui.select`` → AskUserQuestion form → chosen label flows back to Pi."""
    base_url, session_id = seeded_session
    _log.info("seeded session ready: base_url=%s session_id=%s", base_url, session_id)

    holder: dict = {}
    hook_thread = _post_extension_ui(
        base_url,
        session_id,
        {
            "id": "uuid-select",
            "method": "select",
            "title": _SELECT_TITLE,
            "options": [_ALLOW_ONCE, "Allow always", _DENY],
        },
        holder,
    )

    # Let the server park the elicitation before the SPA tries to render it.
    page.wait_for_timeout(500)
    page.goto(f"{base_url}/c/{session_id}")

    card = (
        page.locator(f'{_APPROVAL_CARD}[data-state="pending"]')
        .filter(has=page.locator(_FORM))
        .first
    )
    expect(card).to_be_visible(timeout=_MOCK_ELICITATION_TIMEOUT_MS)
    # The header names Pi, not the default "Claude has questions" — the prompt
    # came from a Pi extension and a reviewer of the card should see that.
    expect(card.get_by_text("Pi needs your input")).to_be_visible()

    form = card.locator(_FORM)
    expect(form.get_by_text(_SELECT_TITLE, exact=True)).to_be_visible()
    # Only the extension's own options: Pi's select cannot accept free text, so
    # a custom-input row would let Submit answer with something it never offered.
    for label in (_ALLOW_ONCE, "Allow always", _DENY):
        expect(form.get_by_text(label, exact=True)).to_be_visible()
    assert _pending_elicitations(base_url, session_id), "server has no parked elicitation"

    form.get_by_role("radio", name=_ALLOW_ONCE).check()
    form.locator(_SUBMIT).click()

    expect(page.locator(f'{_APPROVAL_CARD}[data-state="responded"]').first).to_be_visible(
        timeout=_MOCK_ELICITATION_TIMEOUT_MS
    )

    verdict = _join_hook(hook_thread, holder)
    assert verdict["action"] == "accept", verdict
    assert _ALLOW_ONCE in str(verdict.get("content")), verdict
    _wait_for(lambda: not _pending_elicitations(base_url, session_id))


@pytest.mark.timeout(90)
def test_pi_extension_confirm_renders_a_binary_card_and_approves(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """``ctx.ui.confirm`` → binary Approve/Reject card → Approve returns accept."""
    base_url, session_id = seeded_session

    holder: dict = {}
    hook_thread = _post_extension_ui(
        base_url,
        session_id,
        {
            "id": "uuid-confirm",
            "method": "confirm",
            "title": _CONFIRM_TITLE,
            "message": _CONFIRM_MESSAGE,
        },
        holder,
    )

    page.wait_for_timeout(500)
    page.goto(f"{base_url}/c/{session_id}")

    card = page.locator(f'{_APPROVAL_CARD}[data-state="pending"]').first
    expect(card).to_be_visible(timeout=_MOCK_ELICITATION_TIMEOUT_MS)
    # A confirm is binary — no question form, just the two buttons.
    expect(card.locator(_FORM)).to_have_count(0)
    # Both halves of the dialog survive: the mapper joins title and message with
    # a blank line, which the card must not collapse into one run-on sentence.
    expect(card.get_by_text(_CONFIRM_TITLE, exact=False)).to_be_visible()
    expect(card.get_by_text(_CONFIRM_MESSAGE, exact=False)).to_be_visible()

    card.get_by_role("button", name="Approve").click()

    responded = page.locator(f'{_APPROVAL_CARD}[data-state="responded"]').first
    expect(responded).to_be_visible(timeout=_MOCK_ELICITATION_TIMEOUT_MS)

    verdict = _join_hook(hook_thread, holder)
    assert verdict["action"] == "accept", verdict
    _wait_for(lambda: not _pending_elicitations(base_url, session_id))


@pytest.mark.timeout(90)
def test_pi_extension_reject_returns_a_verdict_rather_than_failing(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """Reject answers the dialog instead of erroring.

    The caller turns this verdict into Pi's cancelled value (``false`` for a
    confirm), so the extension unblocks and the gated tool is the only thing
    that stops — the turn keeps running. A Reject that instead failed the hook
    would strand the extension exactly as the original hang did.
    """
    base_url, session_id = seeded_session

    holder: dict = {}
    hook_thread = _post_extension_ui(
        base_url,
        session_id,
        {
            "id": "uuid-reject",
            "method": "confirm",
            "title": _CONFIRM_TITLE,
            "message": _CONFIRM_MESSAGE,
        },
        holder,
    )

    page.wait_for_timeout(500)
    page.goto(f"{base_url}/c/{session_id}")

    card = page.locator(f'{_APPROVAL_CARD}[data-state="pending"]').first
    expect(card).to_be_visible(timeout=_MOCK_ELICITATION_TIMEOUT_MS)
    card.get_by_role("button", name="Reject").click()

    responded = page.locator(f'{_APPROVAL_CARD}[data-state="responded"]').first
    expect(responded).to_be_visible(timeout=_MOCK_ELICITATION_TIMEOUT_MS)

    verdict = _join_hook(hook_thread, holder)
    assert verdict["action"] == "decline", verdict
    _wait_for(lambda: not _pending_elicitations(base_url, session_id))
