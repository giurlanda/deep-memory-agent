import pytest
from langchain_core.messages import AIMessage

from deep_memory_agent import (
    READ_ONLY_MEMORY_PERMISSIONS,
    create_memory_manager_agent,
    create_memory_search_agent,
)

MEMORY_FILE = "/memory/semantic_memory/facts.md"


def tool_names(agent):
    return set(agent.nodes["tools"].bound._tools_by_name)


def write_file_call(content):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": MEMORY_FILE, "content": content},
                "id": "call-1",
            }
        ],
    )


def test_search_agent_exposes_only_recall_tools(scripted_model, memory_dir):
    agent = create_memory_search_agent(scripted_model(), memory_dir=memory_dir)

    names = tool_names(agent)

    assert {"memory_index", "memory_search", "memory_read"} <= names
    assert not names & {"memory_write", "memory_update", "memory_consolidate"}


def test_manager_agent_exposes_the_write_tools(scripted_model, memory_dir):
    agent = create_memory_manager_agent(scripted_model(), memory_dir=memory_dir)

    assert {
        "memory_index",
        "memory_search",
        "memory_read",
        "memory_write",
        "memory_update",
        "memory_consolidate",
    } <= tool_names(agent)


def test_search_agent_is_denied_writes_on_the_memory_tree(scripted_model, memory_dir):
    agent = create_memory_search_agent(
        scripted_model(write_file_call("hijacked"), AIMessage(content="done")),
        memory_dir=memory_dir,
    )
    before = (memory_dir / "semantic_memory" / "facts.md").read_text()

    result = agent.invoke({"messages": [{"role": "user", "content": "write"}]})

    denied = [
        message
        for message in result["messages"]
        if getattr(message, "name", None) == "write_file"
    ]
    assert denied
    assert "permission denied" in denied[0].content
    assert (memory_dir / "semantic_memory" / "facts.md").read_text() == before


def test_manager_agent_may_write_on_the_memory_tree(scripted_model, memory_dir):
    agent = create_memory_manager_agent(
        scripted_model(
            write_file_call("# Facts\n\nedited\n"), AIMessage(content="done")
        ),
        memory_dir=memory_dir,
    )

    agent.invoke({"messages": [{"role": "user", "content": "write"}]})

    assert "edited" in (memory_dir / "semantic_memory" / "facts.md").read_text()


def test_read_only_permissions_deny_writes_under_memory():
    (rule,) = READ_ONLY_MEMORY_PERMISSIONS

    assert rule.mode == "deny"
    assert rule.operations == ["write"]
    assert "/memory/**" in rule.paths


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_scaffold_the_tree(factory, scripted_model, memory_dir):
    factory(scripted_model(), memory_dir=memory_dir)

    assert (memory_dir / "index.md").exists()
    assert (memory_dir / "episodic_memory" / "index.md").exists()


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_accept_an_explicit_backend(
    factory, scripted_model, backend, memory_dir
):
    factory(scripted_model(), backend=backend)

    assert (memory_dir / "index.md").exists()


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_reject_both_memory_dir_and_backend(
    factory, scripted_model, backend, memory_dir
):
    with pytest.raises(ValueError, match="not both"):
        factory(scripted_model(), memory_dir=memory_dir, backend=backend)


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_require_one_of_them(factory, scripted_model):
    with pytest.raises(ValueError, match="one of memory_dir or backend"):
        factory(scripted_model())
