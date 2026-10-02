from __future__ import annotations

import pickle

import noisevault as nv
from noisevault.errors import FingerprintMismatch, LayoutError, NoiseVaultError, did_you_mean


def test_an_error_keeps_its_next_step_apart_and_prints_both() -> None:
    error = FingerprintMismatch("ibm_fez is nv:06404cefa54f", hint="load ibm_fez@2025-02-26")
    assert (error.message, error.hint) == ("ibm_fez is nv:06404cefa54f", "load ibm_fez@2025-02-26")
    assert str(error) == "ibm_fez is nv:06404cefa54f; load ibm_fez@2025-02-26"
    assert isinstance(error, ValueError)


def test_an_error_without_a_hint_prints_its_message_alone() -> None:
    error = LayoutError("qubit 9 is not on the device")
    assert error.hint is None and str(error) == error.message == "qubit 9 is not on the device"


def test_a_hint_survives_pickling() -> None:
    copy = pickle.loads(pickle.dumps(NoiseVaultError("no source", hint="pull it again")))
    assert (copy.message, copy.hint, str(copy)) == (
        "no source",
        "pull it again",
        "no source; pull it again",
    )


def test_did_you_mean_quotes_the_closest_choice_or_says_nothing() -> None:
    assert did_you_mean("ibm_fezz", ["ibm_fez", "ibm_kyiv"]) == "did you mean 'ibm_fez'? "
    assert did_you_mean("zzz", ["ibm_fez"]) == ""


def test_unreadable_source_data_is_caught_as_a_noisevault_error_or_a_value_error() -> None:
    assert issubclass(nv.SourceDataError, NoiseVaultError)
    assert issubclass(nv.SourceDataError, ValueError)
    assert "SourceDataError" in nv.__all__
