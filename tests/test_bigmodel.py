"""ai-room bigmodel: measure the device, then the arithmetic of what fits and how fast it can possibly go."""

import pytest

from ai_room import bigmodel as B
from ai_room.cli import main


def test_shapes_parse():
    d = B.parse_shape("27")
    m = B.parse_shape("30.5/3.3")
    assert not d.moe and d.total_b == 27 and d.file_gb == pytest.approx(27 * 4.9 / 8)
    assert m.moe and m.token_gb == pytest.approx(3.3 * 4.9 / 8) and m.file_gb == pytest.approx(30.5 * 4.9 / 8)


def test_a_model_that_fits_is_limited_by_memory_bandwidth():
    r = B.plan_one(B.ModelShape("dense 1.5B", 1.5), ram_free_gb=8, memory_gbps=20, storage_gbps=1)
    assert r["fits_in_ram"] and r["max_tok_s"] == pytest.approx(20 / (1.5 * 4.9 / 8), rel=1e-3)


def test_moe_is_faster_than_dense_of_the_same_total_size():
    dense = B.plan_one(B.ModelShape("d", 30.5), 64, 20, 1)
    moe = B.plan_one(B.ModelShape("m", 30.5, 3.3), 64, 20, 1)
    assert moe["max_tok_s"] / dense["max_tok_s"] == pytest.approx(30.5 / 3.3, rel=1e-3)


def test_streaming_lets_it_run_but_storage_sets_the_speed():
    moe = B.plan_one(B.ModelShape("m", 30.5, 3.3), ram_free_gb=8, memory_gbps=20, storage_gbps=1)
    assert not moe["fits_in_ram"] and "streamed" in moe["how"]
    assert 0 < moe["max_tok_s_nothing_cached"] <= moe["max_tok_s_with_cache"] < B.plan_one(B.ModelShape("m", 30.5, 3.3), 64, 20, 1)["max_tok_s"]
    dense70 = B.plan_one(B.ModelShape("d", 70), ram_free_gb=8, memory_gbps=20, storage_gbps=1)
    assert dense70["max_tok_s"] < 0.1                      # a dense 70B streamed from 1 GB/s storage: far below 1 token/second


def test_an_moe_whose_shared_weights_do_not_fit_is_refused():
    r = B.plan_one(B.ModelShape("huge", 1000, 200), ram_free_gb=4, memory_gbps=20, storage_gbps=1)
    assert r["max_tok_s"] == 0.0 and "does not fit" in r["how"]


def test_device_measurements_are_real_numbers(tmp_path):
    assert B.memory_bandwidth_gbps(mb=8, rounds=1) > 0
    st = B.storage_read_gbps(str(tmp_path), mb=8)
    assert st["gbps"] > 0 and not any(tmp_path.iterdir())    # the test file is removed
    ram = B.ram_gb()
    assert ram["total"] is None or ram["total"] > 0


def test_plan_command_prints_a_table(capsys, monkeypatch):
    monkeypatch.setattr(B, "measure_device", lambda folder=None: {"ram_gb": {"total": 8, "available": 6}, "cores": 8, "memory_gbps": 20.0, "storage": {"gbps": 1.0, "cache_dropped": True, "folder": "/tmp"}})
    main(["bigmodel", "plan", "--models", "27,30.5/3.3", "--target", "1200"])
    out = capsys.readouterr().out
    assert "dense 27B" in out and "MoE 30.5B (3.3B active)" in out and "short of 1200 tok/s" in out


def test_bench_needs_a_file():
    with pytest.raises(SystemExit):
        main(["bigmodel", "bench"])
