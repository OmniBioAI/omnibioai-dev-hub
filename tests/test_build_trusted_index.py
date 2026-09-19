"""Tests for the Phase 18 trusted-index builder's embedding backpressure.

These exercise embed_metadata() in isolation with a fake embed_fn -- no real
Ollama calls, no real FAISS writes. Batching/cooldown, per-attempt retry
telemetry, and per-vector dimension validation are all pure logic here and
don't need the network or faiss.
"""

from unittest.mock import patch

import pytest

from scripts.build_trusted_index import embed_metadata


def _meta(chunk_id: str, source: str = "repo:README.md@abc") -> dict:
    return {"chunk_id": chunk_id, "source": source, "text": f"text for {chunk_id}"}


def _ok_embed_fn(dim: int = 768):
    def _embed(text, model=None, on_attempt=None):  # noqa: ARG001
        if on_attempt:
            on_attempt(1, True)
        return [0.1] * dim
    return _embed


def test_embed_metadata_batches_and_cools_down():
    metadata = [_meta(f"c{i}") for i in range(7)]
    with patch("scripts.build_trusted_index.time.sleep") as mock_sleep:
        result = embed_metadata(
            metadata, batch_size=3, cooldown_seconds=2.5,
            embed_fn=_ok_embed_fn(), progress=False,
        )
    # 7 chunks, batch_size=3 -> cooldown fires after chunk 3 and chunk 6, not after the final (7th, partial) batch.
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(2.5)
    assert result["stats"]["successful_embeddings"] == 7
    assert result["stats"]["failed_embeddings"] == 0
    assert result["stats"]["batch_size"] == 3
    assert result["stats"]["cooldown_seconds"] == 2.5


def test_embed_metadata_does_not_cool_down_after_final_batch():
    metadata = [_meta(f"c{i}") for i in range(3)]
    with patch("scripts.build_trusted_index.time.sleep") as mock_sleep:
        embed_metadata(metadata, batch_size=3, cooldown_seconds=1.0, embed_fn=_ok_embed_fn(), progress=False)
    mock_sleep.assert_not_called()


def test_embed_metadata_counts_retries_without_swallowing_eventual_failure():
    """A chunk that exhausts retries must show up as a failure, and every
    attempt (including the ones that failed before eventually succeeding on
    another chunk) must be counted."""
    calls = {"n": 0}

    def flaky_then_fail(text, model=None, on_attempt=None):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            # First chunk: fails attempt 1, succeeds attempt 2.
            if on_attempt:
                on_attempt(1, False)
                on_attempt(2, True)
            return [0.1] * 768
        # Second chunk: exhausts all retries and raises.
        if on_attempt:
            on_attempt(1, False)
            on_attempt(2, False)
            on_attempt(3, False)
        raise RuntimeError("ollama unreachable")

    metadata = [_meta("c0"), _meta("c1")]
    with patch("scripts.build_trusted_index.time.sleep"):
        result = embed_metadata(
            metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=flaky_then_fail, progress=False,
            second_chance=False,  # isolate pass-1 counting; the second-chance path is tested separately
        )

    stats = result["stats"]
    assert stats["successful_embeddings"] == 1
    assert stats["failed_embeddings"] == 1
    assert stats["total_embedding_attempts"] == 5  # 2 for c0, 3 for c1
    assert stats["retry_count"] == 3  # attempt_number > 1: one for c0, two for c1
    assert result["embedding_failures"][0]["chunk_id"] == "c1"
    assert result["embedding_failures"][0]["error"] == "ollama unreachable"


def test_embed_metadata_second_chance_recovers_a_transient_failure():
    """A chunk that fails the first pass but succeeds on the bounded
    second-chance retry must end up counted as successful, not failed."""
    calls = {"n": 0}

    def fails_first_pass_only(text, model=None, on_attempt=None):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            if on_attempt:
                on_attempt(1, False)
                on_attempt(2, False)
                on_attempt(3, False)
            raise RuntimeError("transient")
        if on_attempt:
            on_attempt(1, True)
        return [0.1] * 768

    metadata = [_meta("c0")]
    with patch("scripts.build_trusted_index.time.sleep") as mock_sleep:
        result = embed_metadata(
            metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=fails_first_pass_only, progress=False,
        )

    stats = result["stats"]
    assert stats["successful_embeddings"] == 1
    assert stats["failed_embeddings"] == 0
    assert stats["second_chance_recovered"] == 1
    assert result["embedding_failures"] == []
    assert result["indexed_meta"][0]["chunk_id"] == "c0"
    # The recovery cooldown (default 3x batch cooldown) must actually be used.
    assert any(call.args == (3.0,) for call in mock_sleep.call_args_list)


def test_embed_metadata_second_chance_gives_up_after_one_retry():
    """A chunk that fails both the first pass and the second chance is a
    final, reported failure -- it is never retried a third time."""

    def always_fails(text, model=None, on_attempt=None):  # noqa: ARG001
        if on_attempt:
            on_attempt(1, False)
            on_attempt(2, False)
            on_attempt(3, False)
        raise RuntimeError("permanently down")

    metadata = [_meta("c0")]
    with patch("scripts.build_trusted_index.time.sleep"):
        result = embed_metadata(metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=always_fails, progress=False)

    stats = result["stats"]
    assert stats["successful_embeddings"] == 0
    assert stats["failed_embeddings"] == 1
    assert stats["second_chance_recovered"] == 0
    assert stats["total_embedding_attempts"] == 6  # 3 attempts x 2 passes
    assert result["embedding_failures"][0]["chunk_id"] == "c0"


def test_embed_metadata_second_chance_disabled_reports_pass1_failure_immediately():
    def always_fails(text, model=None, on_attempt=None):  # noqa: ARG001
        if on_attempt:
            on_attempt(1, False)
        raise RuntimeError("down")

    metadata = [_meta("c0")]
    with patch("scripts.build_trusted_index.time.sleep") as mock_sleep:
        result = embed_metadata(
            metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=always_fails, progress=False,
            second_chance=False,
        )
    assert result["stats"]["failed_embeddings"] == 1
    mock_sleep.assert_not_called()  # no batch boundary reached, and no second-chance cooldown either


def test_embed_metadata_rejects_wrong_dimension_without_corrupting_other_chunks():
    def wrong_dim_for_one(text, model=None, on_attempt=None):  # noqa: ARG001
        if on_attempt:
            on_attempt(1, True)
        if "bad" in text:
            return [0.1] * 5  # wrong dimension
        return [0.1] * 768

    metadata = [_meta("bad-chunk"), _meta("good-chunk")]
    with patch("scripts.build_trusted_index.time.sleep"):
        result = embed_metadata(metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=wrong_dim_for_one, progress=False)

    assert result["stats"]["successful_embeddings"] == 1
    assert result["stats"]["failed_embeddings"] == 1
    assert "Embedding dimension mismatch" in result["embedding_failures"][0]["error"]
    assert result["indexed_meta"][0]["chunk_id"] == "good-chunk"


def test_embed_metadata_dedups_repeated_chunk_ids():
    metadata = [_meta("dup"), _meta("dup"), _meta("unique")]
    with patch("scripts.build_trusted_index.time.sleep"):
        result = embed_metadata(metadata, batch_size=10, cooldown_seconds=1.0, embed_fn=_ok_embed_fn(), progress=False)
    assert result["stats"]["successful_embeddings"] == 2
    assert result["stats"]["duplicate_chunks_removed"] == 1


@pytest.mark.parametrize("total,batch_size,expected_cooldowns", [(1, 5, 0), (5, 5, 0), (6, 5, 1), (10, 5, 1)])
def test_embed_metadata_cooldown_count_matches_batch_boundaries(total, batch_size, expected_cooldowns):
    metadata = [_meta(f"c{i}") for i in range(total)]
    with patch("scripts.build_trusted_index.time.sleep") as mock_sleep:
        embed_metadata(metadata, batch_size=batch_size, cooldown_seconds=1.0, embed_fn=_ok_embed_fn(), progress=False)
    assert mock_sleep.call_count == expected_cooldowns
