import pytest

import paged_kv


def test_second_prompt_reuses_the_first_prompts_blocks():
    s = paged_kv.Scheduler(num_blocks=64, block_size=4, max_batch_size=8, prefix_caching=True)
    system = list(range(12))
    s.add_prompt(1, system + [100, 101], max_new_tokens=16)
    s.add_prompt(2, system + [200, 201], max_new_tokens=16)

    step = s.step()
    assert step.admitted == [1, 2]
    assert step.prefills == [(1, 0, 14), (2, 12, 2)]
    blocks = {seq: table for seq, _, table in s.running()}
    assert blocks[1][:3] == blocks[2][:3]


def test_preemption_and_finish_release_every_block():
    # 16 tokens of capacity: two growing sequences cannot both fit.
    s = paged_kv.Scheduler(num_blocks=4, block_size=4, max_batch_size=4)
    s.add_request(1, 6, max_new_tokens=8)
    s.add_request(2, 6, max_new_tokens=8)
    preempted = []
    for _ in range(4):
        preempted += s.step().preempted
    assert preempted == [2]

    s.finish(1)
    s.finish(2)
    assert (s.num_running, s.num_waiting, s.utilization) == (0, 0, 0.0)
    with pytest.raises(ValueError):
        s.finish(1)


def test_engine_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        paged_kv.Engine(tmp_path)
    with pytest.raises(ValueError):
        paged_kv.Scheduler(0, 4, 4)
