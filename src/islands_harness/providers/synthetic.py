"""A deterministic synthetic agent: the fake model behind ``islands selftest``'s snapshot.

It stands in for a model so the whole pipeline (loop, real task tools, fault injection,
grader, runner, analysis, report and the site JSON) runs end to end on data whose shape is
known in advance. It models no real agent, and nothing it produces is a measurement; every
output built from it is labelled as a selftest.

Behaviour, per run:

- It fetches the document. When the result is not the document (an injected timeout, error,
  empty result or garbled text), it retries with probability 0.75 and otherwise gives up in
  plain text, which ends the run without a submission.
- With the document in hand it may first call the newest distractor tool on offer (search at
  3 tools, calculator at 4, spreadsheet at 5, email at 6), more often the more tools it has.
- It then submits the gold record, or with a probability that depends on the tool count a
  record with the invoice number and total wrong (score 0.6, below tau). With only fetch and
  submit it makes more errors than with one helper tool, and more tools make it worse again,
  so the surface is a band with an interior peak near three tools.
- When a submission is not acknowledged (an unexecuted fault on submit_record), it resubmits
  with probability 0.75 and otherwise claims in text that it submitted, which is a
  no-submission.
- At rung 1, where no submit tool is offered, it writes the record as text.

Every decision is a hashed draw (rng scope "synthetic"). Draws that decide the run's course
(distraction, correctness) are keyed on the document, the tool count and the run's sampling
seed only, so the same index makes the same choices at every fault rate until a fault changes
the path; draws after a fault are also keyed on the turn. The same spec therefore always
yields the same transcript, which is what lets the selftest check replay and re-analysis.
"""

from __future__ import annotations

import json
from typing import Any

from islands_harness.providers.base import Message, Sampling, StopReason, ToolCall, Turn, Usage
from islands_harness.rng import unit

ERROR_RATE = {2: 0.14, 3: 0.05, 4: 0.07, 5: 0.25, 6: 0.50}
DISTRACTION = {2: 0.0, 3: 0.10, 4: 0.20, 5: 0.40, 6: 0.60}
RETRY_AFTER_FAULT = 0.75
FINGERPRINT = "synthetic-v1"
CORE_TOOLS = ("fetch_document", "submit_record")


def _helper_args(name: str, doc_id: str, gold: dict[str, Any]) -> dict[str, Any]:
    if name == "search_web":
        return {"query": f"{gold.get('vendor_name', 'vendor')} invoice"}
    if name == "calculator":
        return {"expression": f"{float(gold.get('total_amount', 0.0))} * 1"}
    if name == "read_spreadsheet":
        return {"doc_id": doc_id}
    if name == "send_email":
        return {
            "to": "accounts@example.invalid",
            "subject": f"Invoice {doc_id}",
            "body": "Please confirm the total.",
        }
    return {"doc_id": doc_id}


def wrong_record(gold: dict[str, Any]) -> dict[str, Any]:
    """The gold record with the invoice number and the total wrong: 6 of 10 grader points."""
    bad = json.loads(json.dumps(gold))
    bad["invoice_number"] = f"{gold.get('invoice_number', '')}-X"
    bad["total_amount"] = round(float(gold.get("total_amount", 0.0)) + 10.0, 2)
    return bad


class SyntheticAgent:
    """A ChatProvider that plays the agent described in the module docstring. Stateless: each
    call reads the conversation so far and decides from it and from hashed draws."""

    def __init__(
        self,
        gold: dict[str, dict[str, Any]],
        documents: dict[str, str],
        *,
        model_id: str = "selftest-synthetic",
    ) -> None:
        self.gold = gold
        self.documents = documents
        self.model_id = model_id

    def identity(self) -> dict[str, Any]:
        return {"model": self.model_id, "provider": "synthetic", "fingerprint": FINGERPRINT}

    async def complete(self, messages: list[Message], tools: list[Any], sampling: Sampling) -> Turn:
        user = next(m for m in messages if m.role == "user")
        doc_id = str(user.content).split(":", 1)[-1].strip()
        names = [t.name for t in tools]
        n = len(names)
        turn = sum(1 for m in messages if m.role == "assistant")
        seed = sampling.seed or 0
        if doc_id not in self.gold:  # not a task run: the gate's filler requests
            return self._text(messages, turn, "filler", "Noted.")
        gold = self.gold[doc_id]

        def per_run(purpose: str) -> float:
            return unit("synthetic", doc_id, n, purpose, seed)

        def per_turn(purpose: str) -> float:
            return unit("synthetic", doc_id, n, turn, purpose, seed)

        if turn == 0:
            return self._call(messages, turn, doc_id, "fetch_document", {"doc_id": doc_id})
        last = messages[-1].tool_results[0] if messages[-1].tool_results else None
        if last is None:
            return self._text(messages, turn, doc_id, "I have nothing further to add.")
        if last.name == "fetch_document":
            if last.content != self.documents[doc_id]:
                if per_turn("retry") < RETRY_AFTER_FAULT:
                    return self._call(messages, turn, doc_id, "fetch_document", {"doc_id": doc_id})
                return self._text(messages, turn, doc_id, "I could not retrieve the document.")
            if "submit_record" not in names:
                record = json.dumps(gold, sort_keys=True)
                return self._text(messages, turn, doc_id, f"The extracted record is {record}")
            helpers = [t for t in names if t not in CORE_TOOLS]
            if helpers and per_run("distract") < DISTRACTION.get(n, 0.0):
                helper = helpers[-1]
                return self._call(
                    messages, turn, doc_id, helper, _helper_args(helper, doc_id, gold)
                )
            return self._submit(messages, turn, doc_id, n, per_run)
        if last.name == "submit_record":
            if per_turn("retry") < RETRY_AFTER_FAULT:
                return self._submit(messages, turn, doc_id, n, per_run)
            return self._text(messages, turn, doc_id, "The record has been submitted.")
        return self._submit(messages, turn, doc_id, n, per_run)  # after a distractor tool

    # -- turn builders ---------------------------------------------------------------------

    def _submit(
        self, messages: list[Message], turn: int, doc_id: str, n: int, per_run: Any
    ) -> Turn:
        gold = self.gold[doc_id]
        correct = per_run("correct") >= ERROR_RATE.get(n, 0.5)
        record = gold if correct else wrong_record(gold)
        return self._call(messages, turn, doc_id, "submit_record", {"record": record})

    def _call(
        self, messages: list[Message], turn: int, doc_id: str, name: str, args: dict[str, Any]
    ) -> Turn:
        call = ToolCall(
            id=f"syn{turn}",
            name=name,
            arguments=args,
            arguments_text=json.dumps(args, sort_keys=True),
        )
        return self._turn(messages, turn, doc_id, "", [call])

    def _text(self, messages: list[Message], turn: int, doc_id: str, text: str) -> Turn:
        return self._turn(messages, turn, doc_id, text, [])

    def _turn(
        self, messages: list[Message], turn: int, doc_id: str, text: str, calls: list[ToolCall]
    ) -> Turn:
        seen = sum(
            len(str(m.content or "")) + sum(len(r.content) for r in (m.tool_results or []))
            for m in messages
        )
        written = len(text) + sum(len(c.arguments_text) for c in calls)
        return Turn(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=seen // 4, output_tokens=written // 4 + 1),
            stop_reason=StopReason.tool_calls if calls else StopReason.end,
            identity={
                "model": self.model_id,
                "fingerprint": FINGERPRINT,
                "response_id": f"syn-{doc_id}-{turn}",
            },
            message=Message(role="assistant", content=text or None, tool_calls=calls or None),
            raw_request={"messages": len(messages), "tools": [c.name for c in calls]},
            raw_response={"synthetic": True},
        )
