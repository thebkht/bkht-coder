"""The plan artifact: the one thing a turn keeps when it loses everything else."""

from __future__ import annotations

import pytest

from bkht.coder.plan import MAX_STEP_CHARS, MAX_STEPS, Plan


def test_an_empty_plan_is_falsey_and_renders_as_nothing():
    # Falsiness is what keeps the reminder off every request in a session that
    # never wrote a plan, which is most of them.
    made = Plan()
    assert not made
    assert made.render() == ""
    assert made.progress() == (0, 0)


def test_steps_are_numbered_from_one_and_start_unticked():
    made = Plan()
    made.set(["read reviewer.py", "say what it does"])
    assert made.render() == "1. [ ] read reviewer.py\n2. [ ] say what it does"


def test_ticking_a_step_marks_only_that_one():
    made = Plan()
    made.set(["one", "two", "three"])
    step = made.tick(2)
    assert step.text == "two"
    assert made.progress() == (1, 3)
    assert made.render().splitlines()[1] == "2. [x] two"


def test_a_step_number_outside_the_plan_says_how_many_there_are():
    # The model picks a number from a list it was shown, and gets it wrong; the
    # message has to be enough to correct from without another round trip.
    made = Plan()
    made.set(["one", "two"])
    with pytest.raises(IndexError, match="the plan has 2"):
        made.tick(5)
    with pytest.raises(IndexError):
        made.tick(0)


def test_ticking_before_there_is_a_plan_says_so():
    with pytest.raises(IndexError, match="no plan yet"):
        Plan().tick(1)


def test_blank_steps_are_dropped_and_long_ones_are_cut():
    made = Plan()
    made.set(["  keep  ", "", "   ", "x" * (MAX_STEP_CHARS + 50)])
    assert len(made) == 2
    assert made.steps[0].text == "keep"
    assert len(made.steps[1].text) == MAX_STEP_CHARS


def test_a_plan_longer_than_the_cap_is_cut_to_it():
    # A model that writes twelve steps has decomposed the task instead of doing
    # it, and every step is paid for on every subsequent request.
    made = Plan()
    made.set([f"step {n}" for n in range(MAX_STEPS + 5)])
    assert len(made) == MAX_STEPS


def test_rewriting_the_plan_drops_the_ticks():
    # Deliberate. Carrying ticks across a rewrite means guessing which new step
    # each old one became, and a wrong guess reports work nobody did as done.
    made = Plan()
    made.set(["one", "two"])
    made.tick(1)
    made.set(["something else", "and another"])
    assert made.progress() == (0, 2)


def test_finished_is_false_for_an_empty_plan():
    # Nothing to do is not the same as everything done, and the tool tells the
    # model to stop and answer when a plan is finished.
    assert Plan().finished() is False
    made = Plan()
    made.set(["only"])
    assert made.finished() is False
    made.tick(1)
    assert made.finished() is True


def test_a_plan_survives_a_round_trip_through_its_record():
    made = Plan()
    made.set(["one", "two"])
    made.tick(2)
    back = Plan.from_record(made.as_record())
    assert back.render() == made.render()


def test_a_malformed_record_yields_the_steps_it_can_read():
    # The transcript is append-only and a session killed mid-write can leave a
    # partial line. A plan is not worth refusing to resume a session over.
    back = Plan.from_record([{"text": "good"}, "not a dict", {"text": "  "}, None])
    assert [step.text for step in back.steps] == ["good"]
    assert Plan.from_record(None).steps == []
