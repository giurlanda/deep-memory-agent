import pytest
from langchain_core.messages import AIMessage

from deep_memory_agent import (
    READ_ONLY_MEMORY_PERMISSIONS,
    create_memory_manager_agent,
    create_memory_search_agent,
)
from deep_memory_agent.prompts import SEMANTIC_MANAGER_BLOCK, SEMANTIC_READER_BLOCK

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


def system_prompt(agent, model):
    """Return the system prompt the agent actually sent to the model.

    Read off a real invocation rather than out of the compiled graph: what
    matters is the prompt that reaches the model, and the graph's internals are
    deepagents', not ours, to depend on.
    """
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    return model.seen[0][0].content


def test_search_agent_exposes_only_recall_tools(scripted_model, memory_dir):
    agent = create_memory_search_agent(scripted_model(), memory_dir=memory_dir)

    names = tool_names(agent)

    assert {"memory_index", "memory_search", "memory_get", "memory_read"} <= names
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


# ---------------------------------------------------------- semantic search --


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_no_semantic_tools_without_embeddings_and_store(
    factory, scripted_model, memory_dir
):
    agent = factory(scripted_model(), memory_dir=memory_dir)

    assert not tool_names(agent) & {"semantic_ingest", "semantic_search"}


def test_manager_gets_both_semantic_tools(
    scripted_model, memory_dir, embeddings, vector_store
):
    agent = create_memory_manager_agent(
        scripted_model(),
        memory_dir=memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    assert {"semantic_ingest", "semantic_search"} <= tool_names(agent)


def test_search_agent_gets_the_search_tool_but_never_the_ingest_one(
    scripted_model, memory_dir, embeddings, vector_store
):
    agent = create_memory_search_agent(
        scripted_model(),
        memory_dir=memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    names = tool_names(agent)

    # Withholding the tool is the only guarantee here: the filesystem
    # permissions cannot deny a call that talks to an external vector store.
    assert "semantic_search" in names
    assert "semantic_ingest" not in names


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_reject_embeddings_without_a_vector_store(
    factory, scripted_model, memory_dir, embeddings
):
    with pytest.raises(ValueError, match="both `embeddings` and `vector_store`"):
        factory(scripted_model(), memory_dir=memory_dir, embeddings=embeddings)


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_factories_reject_a_vector_store_without_embeddings(
    factory, scripted_model, memory_dir, vector_store
):
    with pytest.raises(ValueError, match="both `embeddings` and `vector_store`"):
        factory(scripted_model(), memory_dir=memory_dir, vector_store=vector_store)


@pytest.mark.parametrize(
    ("factory", "block"),
    [
        (create_memory_search_agent, SEMANTIC_READER_BLOCK),
        (create_memory_manager_agent, SEMANTIC_MANAGER_BLOCK),
    ],
)
def test_the_semantic_prompt_block_is_appended_only_when_active(
    factory, block, scripted_model, memory_dir, embeddings, vector_store
):
    plain_model, semantic_model = scripted_model(), scripted_model()
    plain = factory(plain_model, memory_dir=memory_dir)
    semantic = factory(
        semantic_model,
        memory_dir=memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    assert block not in system_prompt(plain, plain_model)
    assert block in system_prompt(semantic, semantic_model)


@pytest.mark.parametrize(
    "factory", [create_memory_search_agent, create_memory_manager_agent]
)
def test_a_custom_prompt_is_left_exactly_as_given(
    factory, scripted_model, memory_dir, embeddings, vector_store
):
    model = scripted_model()
    agent = factory(
        model,
        memory_dir=memory_dir,
        system_prompt="mine",
        embeddings=embeddings,
        vector_store=vector_store,
    )

    # An override replaces the built-in prompt whole, semantic section
    # included: a caller who restates the layout rules also owns this part.
    assert system_prompt(agent, model) == "mine"


def test_search_k_reaches_the_semantic_tool(
    scripted_model, memory_dir, embeddings, vector_store
):
    agent = create_memory_manager_agent(
        scripted_model(),
        memory_dir=memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
        search_k=3,
    )

    tool = agent.nodes["tools"].bound._tools_by_name["semantic_search"]
    assert tool.args_schema.model_fields["k"].default == 3
