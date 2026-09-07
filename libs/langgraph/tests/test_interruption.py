import operator
import uuid
from typing import Annotated

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, interrupt

pytestmark = pytest.mark.anyio


def test_interruption_without_state_updates(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_2",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_3",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ()
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


async def test_interruption_without_state_updates_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    async def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke(initial_input, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_2",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_3",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ()
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


class TestResumeMapZeroMatch:
    """A hex-keyed dict resume is a by-id map only if it targets a pending
    interrupt id (#8836).

    Before the fix, shape-only detection made zero-match resumes silent
    no-ops: nothing delivered, nothing raised, thread left interrupted.
    """

    def test_single_interrupt_uuid_keyed_dict_delivered_verbatim(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        class State(TypedDict):
            log: Annotated[list[str], operator.add]

        def gate(state: dict) -> dict:
            return {"log": [f"approved:{interrupt('ask')!r}"]}

        builder = StateGraph(State)
        builder.add_node("gate", gate)
        builder.add_edge(START, "gate")
        builder.add_edge("gate", END)
        graph = builder.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": []}, thread)

        ticket_id = uuid.uuid4().hex
        out = graph.invoke(Command(resume={ticket_id: "approve"}), thread)
        # dict delivered verbatim to the single pending interrupt
        assert len(out["log"]) == 1
        assert ticket_id in out["log"][0]
        assert graph.get_state(thread).next == ()

    def test_single_interrupt_resume_by_actual_id_still_maps(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        class State(TypedDict):
            log: Annotated[list[str], operator.add]

        def gate(state: dict) -> dict:
            return {"log": [f"approved:{interrupt('ask')!r}"]}

        builder = StateGraph(State)
        builder.add_node("gate", gate)
        builder.add_edge(START, "gate")
        builder.add_edge("gate", END)
        graph = builder.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": []}, thread)

        snap = graph.get_state(thread)
        rid = [i.id for t in snap.tasks for i in t.interrupts][0]
        out = graph.invoke(Command(resume={rid: "approve"}), thread)
        # map mode: interrupt receives the mapped value, not the dict
        assert out["log"] == ["approved:'approve'"]
        assert graph.get_state(thread).next == ()

    def test_nested_parallel_sibling_resume_keeps_map_semantics(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        """Resuming a non-first nested sibling by id must stay a map (#8836).

        Top-level _pending_interrupts() surfaces one id per parent task, so
        a zero-match guard keyed on it alone would deliver the map verbatim
        and corrupt the value. The unpoisoned id count keeps this a map.
        """

        class State(TypedDict):
            log: Annotated[list[str], operator.add]

        def leaf1(state: dict) -> dict:
            return {"log": [f"s1:{interrupt('s1')!r}"]}

        def leaf2(state: dict) -> dict:
            return {"log": [f"s2:{interrupt('s2')!r}"]}

        child = StateGraph(State)
        child.add_node("leaf1", leaf1)
        child.add_node("leaf2", leaf2)
        child.add_edge(START, "leaf1")
        child.add_edge(START, "leaf2")
        child.add_edge("leaf1", END)
        child.add_edge("leaf2", END)

        parent = StateGraph(State)
        parent.add_node("sub", child.compile())
        parent.add_edge(START, "sub")
        parent.add_edge("sub", END)
        graph = parent.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": []}, thread)

        snap = graph.get_state(thread, subgraphs=True)
        ids: dict[str, str] = {}
        for st in snap.tasks:
            for i in st.state.tasks:
                for leaf_interrupt in i.interrupts:
                    ids[leaf_interrupt.value] = leaf_interrupt.id
        target = ids["s2"]  # not the first sibling
        graph.invoke(Command(resume={target: "v"}), thread)
        snap2 = graph.get_state(thread, subgraphs=True)
        # the mapped value, not the raw dict, must be delivered to s2
        assert snap2.tasks[0].state.values["log"] == ["s2:'v'"]
        # sibling s1 must remain interrupted; leaf2 has completed
        assert snap2.tasks[0].state.next == ("leaf1",)

    def test_zero_match_after_stale_resume_write_stays_noop(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        """Poisoned _pending_interrupts() must not turn a no-op into delivery.

        Node a has two sequential interrupts; resuming the first leaves a
        stale same-task RESUME write that hides the second from
        _pending_interrupts(). A later zero-match hex dict must remain a
        silent no-op (main behavior), not get delivered to the hidden
        interrupt.
        """

        class State(TypedDict):
            log: Annotated[list[str], operator.add]

        def a(state: dict) -> dict:
            first = interrupt("a0")
            second = interrupt("a1")
            return {"log": [f"a:{first}:{second}"]}

        def b(state: dict) -> dict:
            return {"log": [f"b:{interrupt('b0')!r}"]}

        builder = StateGraph(State)
        builder.add_node("a", a)
        builder.add_node("b", b)
        builder.add_edge(START, "a")
        builder.add_edge(START, "b")
        builder.add_edge("a", END)
        builder.add_edge("b", END)
        graph = builder.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": []}, thread)

        snap = graph.get_state(thread)
        ids = {t.interrupts[0].value: t.interrupts[0].id for t in snap.tasks}
        # resume a0 only; a1 becomes hidden by the stale same-task RESUME
        graph.invoke(Command(resume={ids["a0"]: "va0"}), thread)

        # zero-match hex dict: must not be delivered to the hidden a1
        graph.invoke(Command(resume={uuid.uuid4().hex: "PAYLOAD"}), thread)
        state = graph.get_state(thread)
        assert state.values["log"] == []
        pending_vals = {i.value for t in state.tasks for i in t.interrupts}
        assert pending_vals == {"a1", "b0"}, pending_vals

    def test_empty_dict_resume_single_pending_delivered_verbatim(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        """An empty dict resume vacuously passes the all-hex check; with one
        pending interrupt and zero matching keys it is a value, not a map."""

        class State(TypedDict):
            log: str

        def gate(state: dict) -> dict:
            return {"log": f"approved:{interrupt('ask')!r}"}

        builder = StateGraph(State)
        builder.add_node("gate", gate)
        builder.add_edge(START, "gate")
        builder.add_edge("gate", END)
        graph = builder.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": ""}, thread)

        out = graph.invoke(Command(resume={}), thread)
        assert out["log"] == "approved:{}"
        assert graph.get_state(thread).next == ()

    def test_multiple_interrupts_partial_map_unchanged(
        self, sync_checkpointer: BaseCheckpointSaver
    ) -> None:
        class State(TypedDict):
            log: Annotated[list[str], operator.add]

        def gate_a(state: dict) -> dict:
            return {"log": [f"A:{interrupt('a')!r}"]}

        def gate_b(state: dict) -> dict:
            return {"log": [f"B:{interrupt('b')!r}"]}

        builder = StateGraph(State)
        builder.add_node("gate_a", gate_a)
        builder.add_node("gate_b", gate_b)
        builder.add_edge(START, "gate_a")
        builder.add_edge(START, "gate_b")
        builder.add_edge("gate_a", END)
        builder.add_edge("gate_b", END)
        graph = builder.compile(checkpointer=sync_checkpointer)
        thread = {"configurable": {"thread_id": "1"}}
        graph.invoke({"log": []}, thread)

        snap = graph.get_state(thread)
        ids = {t.interrupts[0].value: t.interrupts[0].id for t in snap.tasks}
        graph.invoke(Command(resume={ids["a"]: "va"}), thread)
        state = graph.get_state(thread)
        assert state.values["log"] == ["A:'va'"]
        assert state.next == ("gate_b",)
