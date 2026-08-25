"""The attribute ontology of the operational domain.

This is the replacement for Table 5 of the paper. Where LongMemEval samples a
user attribute and builds a life around it, this samples an operational entity —
an account, a project, a procedure, a mistake — and builds a working history
around it.

The axes are chosen so that each one can produce evidence for a different
category. `PROCEDURES` carry trigger conditions and steps, which is what a
procedural-retrieval question tests. `ERRORS` carry a mistake, its cause and the
correction that followed, which is what a non-repetition question tests. The
client and project axes exist to give the distractor sessions somewhere real to
happen, so that the haystack is the same kind of material as the needle rather
than obviously off-topic filler.
"""

from __future__ import annotations

__all__ = [
    "CLIENTS",
    "ERRORS",
    "FEEDBACK_THEMES",
    "PLANS",
    "PROCEDURES",
    "PROJECT_EVENTS",
    "STACKS",
]

CLIENTS: tuple[dict[str, str], ...] = (
    {"name": "Acme Logistics", "sector": "freight and warehousing"},
    {"name": "Northwind Health", "sector": "clinical scheduling"},
    {"name": "Brightloom Retail", "sector": "point-of-sale software"},
    {"name": "Vantage Mobility", "sector": "fleet telematics"},
    {"name": "Helios Energy", "sector": "grid monitoring"},
    {"name": "Marlowe Legal", "sector": "contract automation"},
    {"name": "Ferro Manufacturing", "sector": "industrial IoT"},
    {"name": "Sablefin Capital", "sector": "portfolio reporting"},
)
"""Accounts the histories revolve around."""

PLANS: tuple[str, ...] = ("Starter", "Team", "Business", "Enterprise", "Enterprise+")
"""Contract tiers, ordered, so an upgrade or downgrade is unambiguous."""

STACKS: tuple[str, ...] = (
    "Postgres 16 on managed RDS",
    "a self-hosted Kubernetes cluster",
    "Snowflake plus dbt",
    "an on-prem VMware estate",
    "Azure Functions with Cosmos DB",
    "a Django monolith on bare metal",
)
"""What the client runs, which constrains what can plausibly go wrong."""

PROJECT_EVENTS: tuple[str, ...] = (
    "a migration cutover",
    "a quarterly business review",
    "a security questionnaire",
    "a load test before peak season",
    "an SSO rollout",
    "a data retention audit",
    "a pricing renegotiation",
    "an incident postmortem",
)
"""Occasions that generate sessions worth remembering."""

PROCEDURES: tuple[dict[str, str], ...] = (
    {
        "title": "Production deploy rollback",
        "trigger": "a deploy fails partway through a schema migration",
        "steps": (
            "stop the rollout before the next batch, "
            "restore the pre-migration snapshot rather than re-running the "
            "migration forward, verify row counts on the three largest tables, "
            "then re-enable writes and announce on the incident channel"
        ),
    },
    {
        "title": "New client onboarding",
        "trigger": "a signed contract arrives for a new account",
        "steps": (
            "create the tenant in staging first, load the client's sample "
            "export to check the field mapping, only then create the "
            "production tenant, and schedule the handover call within five "
            "working days"
        ),
    },
    {
        "title": "Escalation to the account owner",
        "trigger": "a client reports the same defect twice within a week",
        "steps": (
            "collect the two ticket ids and the reproduction, notify the "
            "account owner before replying to the client, agree the message "
            "with them, and put a follow-up in the calendar for 48 hours later"
        ),
    },
    {
        "title": "Release note publication",
        "trigger": "a version is tagged and the build has gone green",
        "steps": (
            "draft the notes from the merged pull requests, have the account "
            "owner check anything customer-visible, publish to the changelog "
            "page, and only then send the client email"
        ),
    },
    {
        "title": "Restoring a client from backup",
        "trigger": "a client reports data loss after a bulk operation",
        "steps": (
            "freeze writes for that tenant, restore into a scratch database "
            "rather than over the live one, diff the affected tables against "
            "the live copy, and get the client's confirmation before merging "
            "anything back"
        ),
    },
    {
        "title": "Handling a security questionnaire",
        "trigger": "a prospect or client sends a security review document",
        "steps": (
            "check the answer bank for a previous version of the same "
            "questionnaire, flag every answer that has drifted since it was "
            "last sent, get sign-off on anything about subprocessors, and log "
            "the submitted copy against the account"
        ),
    },
)
"""Repeatable operating procedures, each with the condition that triggers it."""

ERRORS: tuple[dict[str, str], ...] = (
    {
        "mistake": "ran a bulk update against the production tenant without a "
        "dry run first",
        "consequence": "eleven thousand rows were written with the wrong "
        "timezone offset",
        "correction": "always run the bulk script with --dry-run and diff the "
        "output before touching production",
    },
    {
        "mistake": "quoted a renewal price from the old rate card",
        "consequence": "the client was given a figure fourteen percent below "
        "the approved one and it had to be walked back",
        "correction": "pull the current rate card from the pricing sheet at "
        "quote time, never from a previous email",
    },
    {
        "mistake": "sent the client email announcing a release before the "
        "changelog page was published",
        "consequence": "the link in the email 404'd for two hours",
        "correction": "publish the changelog first, check the link resolves, "
        "then send the email",
    },
    {
        "mistake": "restored a backup directly over the live database",
        "consequence": "six hours of legitimate writes were lost and had to be "
        "replayed from the audit log",
        "correction": "restore into a scratch database and diff before merging "
        "anything back",
    },
    {
        "mistake": "answered a subprocessor question on a security "
        "questionnaire from memory",
        "consequence": "the answer named a vendor that had been replaced eight "
        "months earlier",
        "correction": "answer subprocessor questions from the current vendor "
        "register and have them signed off",
    },
    {
        "mistake": "scheduled a migration cutover during the client's peak "
        "trading window",
        "consequence": "the cutover was aborted and rebooked, costing a week",
        "correction": "check the client's stated peak windows before proposing "
        "any cutover date",
    },
)
"""Mistakes with their cost and the correction that followed."""

FEEDBACK_THEMES: tuple[str, ...] = (
    "wants status updates as a short written summary, not a call",
    "wants numbers stated with their source, never rounded silently",
    "does not want to be copied on internal engineering threads",
    "wants risks flagged early even when they are still uncertain",
    "prefers decisions recorded in the ticket rather than in chat",
)
"""Working preferences the histories can state in passing."""
