from __future__ import annotations

from noisevault.providers.csv_import import import_ibm_calibration_csv
from noisevault.schema import validate_snapshot


def test_csv_import(tmp_path):
    path = tmp_path / "calibration.csv"
    path.write_text(
        "qubit,t1_us,t2_us,frequency_ghz,readout_error,prob_meas0_prep1,prob_meas1_prep0\n"
        "0,100,80,4.9,0.02,0.02,0.02\n"
        "1,110,90,5.0,0.03,0.03,0.03\n",
        encoding="utf-8",
    )
    snapshot = import_ibm_calibration_csv(path, backend_name="csv_demo")
    assert snapshot.num_qubits == 2
    assert validate_snapshot(snapshot).valid
