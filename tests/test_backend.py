"""Tests for the UDA xarray backend."""

from unittest.mock import Mock, call

import numpy as np
import pyuda
import pytest
import xarray as xr

from uda_xarray.mappings import SignalMappings, SignalRange


def test_open_uda_dataset(mocker):
    """Test loading a 1D UDA signal as an xarray Dataset."""
    # Create mock signal object
    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]

    mock_signal.units = "A"
    mock_signal.time = Mock()
    mock_signal.time.data = np.array([0.0, 1.0, 2.0])
    mock_signal.time.label = "time"
    mock_signal.errors = Mock()
    mock_signal.errors.data = np.array([0.1, 0.1, 0.1])

    # Mock the pyuda Client
    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    ds = xr.open_dataset("uda://ip:30421", engine="uda")

    # Verify the client was called correctly
    mock_client.get.assert_called_once_with("ip", "30421")

    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time",)
    assert "time" in ds.coords

    assert ds["data"].shape == ds["time"].shape
    assert "units" in ds["data"].attrs
    assert ds["data"].attrs["uda_name"] == "ip"


def test_open_uda_dataset_2d(mocker):
    """Test loading a 2D UDA signal as an xarray Dataset."""
    # Create mock 2D signal object
    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_signal.shape = (2, 2)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0])
    dim2 = Mock(label="channel")
    dim2.data = np.array([0, 1])
    mock_signal.dims = [dim1, dim2]
    mock_signal.units = "eV"
    mock_signal.errors = Mock()
    mock_signal.errors.data = np.array([[0.1, 0.1], [0.1, 0.1]])
    mock_signal.time = Mock()
    mock_signal.time.data = np.array([0.0, 1.0])
    mock_signal.time.label = "time"

    # Mock the pyuda Client
    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    ds = xr.open_dataset("uda://AYE_TE:30421", engine="uda")

    mock_client.get.assert_called_once_with("AYE_TE", "30421")

    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "channel")
    assert "time" in ds.coords
    assert "channel" in ds.coords


def test_open_uda_dataset_video(mocker):
    """Test loading a UDA video signal as an xarray Dataset."""
    mock_signal = Mock()
    mock_signal.is_color = False
    mock_signal.frame_times = np.array([0.0, 0.033, 0.066])
    frame1 = Mock()
    frame1.k = np.array([[10, 20], [30, 40]])
    frame2 = Mock()
    frame2.k = np.array([[15, 25], [35, 45]])
    frame3 = Mock()
    frame3.k = np.array([[20, 30], [40, 50]])
    mock_signal.frames = [frame1, frame2, frame3]
    mock_signal.height = 2
    mock_signal.width = 2
    mock_signal.duration = 0.066
    mock_signal.num_frames = 3
    mock_signal.name = "rba"
    mock_signal.description = "Mock video signal"
    mock_signal.units = "counts"

    mock_client = Mock()
    mock_client.get_images.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Image",
    )

    ds = xr.open_dataset("uda://rba:30421", engine="uda")
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "height", "width")
    assert "time" in ds.coords
    assert ds.sizes["time"] == 3


def test_open_uda_dataset_invalid_signal(mocker):
    """Test that an invalid UDA signal raises a RuntimeError."""
    # Mock the pyuda Client to raise an exception
    mock_client = Mock()
    mock_client.get.side_effect = pyuda.ServerException("Signal not found")
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    try:
        xr.open_dataset("uda://invalid_signal:99999", engine="uda")
    except RuntimeError as e:
        assert "Could not open UDA dataset" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised"


def test_open_uda_dataset_invalid_format():
    """Test that an invalid UDA URL format raises a ValueError."""
    try:
        xr.open_dataset("invalid_format", engine="uda")
    except ValueError as e:
        assert (
            "UDA dataset must be specified as uda://<signal_name>:<shot_number>"
            in str(e)
        )
    else:
        assert False, "Expected ValueError was not raised"

    try:
        xr.open_dataset("http://invalid_scheme:12345", engine="uda")
    except ValueError as e:
        assert "UDA dataset must start with the uda:// scheme" in str(e)
    else:
        assert False, "Expected ValueError was not raised"


def test_open_datatree(mocker):
    """Test loading a 1D UDA signal as an xarray DataTree."""
    # Create mock signal object
    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]

    mock_signal.units = "A"
    mock_signal.time = Mock()
    mock_signal.time.data = np.array([0.0, 1.0, 2.0])
    mock_signal.time.label = "time"
    mock_signal.errors = Mock()
    mock_signal.errors.data = np.array([0.1, 0.1, 0.1])

    # Mock the pyuda Client
    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    dt = xr.open_datatree("uda://ip:30421", engine="uda")

    # Verify the client was called correctly
    mock_client.get.assert_called_once_with("ip", "30421")

    # Verify it's a DataTree object
    assert isinstance(dt, xr.DataTree)

    # Access the dataset from the root node
    ds = dt.to_dataset()
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time",)
    assert "time" in ds.coords
    assert ds["data"].shape == ds["time"].shape
    assert "units" in ds["data"].attrs
    assert ds["data"].attrs["uda_name"] == "ip"


def test_open_datatree_2d(mocker):
    """Test loading a 2D UDA signal as an xarray DataTree."""
    # Create mock 2D signal object
    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_signal.shape = (2, 2)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0])
    dim2 = Mock(label="channel")
    dim2.data = np.array([0, 1])
    mock_signal.dims = [dim1, dim2]
    mock_signal.units = "eV"
    mock_signal.errors = Mock()
    mock_signal.errors.data = np.array([[0.1, 0.1], [0.1, 0.1]])
    mock_signal.time = Mock()
    mock_signal.time.data = np.array([0.0, 1.0])
    mock_signal.time.label = "time"

    # Mock the pyuda Client
    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    dt = xr.open_datatree("uda://AYE_TE:30421", engine="uda")

    mock_client.get.assert_called_once_with("AYE_TE", "30421")

    # Verify it's a DataTree object
    assert isinstance(dt, xr.DataTree)

    # Access the dataset from the root node
    ds = dt.to_dataset()
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "channel")
    assert "time" in ds.coords
    assert "channel" in ds.coords


def test_open_datatree_video(mocker):
    """Test loading a UDA video signal as an xarray DataTree."""
    mock_signal = Mock()
    mock_signal.is_color = False
    mock_signal.frame_times = np.array([0.0, 0.033, 0.066])
    frame1 = Mock()
    frame1.k = np.array([[10, 20], [30, 40]])
    frame2 = Mock()
    frame2.k = np.array([[15, 25], [35, 45]])
    frame3 = Mock()
    frame3.k = np.array([[20, 30], [40, 50]])
    mock_signal.frames = [frame1, frame2, frame3]
    mock_signal.height = 2
    mock_signal.width = 2
    mock_signal.duration = 0.066
    mock_signal.num_frames = 3
    mock_signal.name = "rba"
    mock_signal.description = "Mock video signal"
    mock_signal.units = "counts"

    mock_client = Mock()
    mock_client.get_images.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Image",
    )

    dt = xr.open_datatree("uda://rba:30421", engine="uda")

    # Verify it's a DataTree object
    assert isinstance(dt, xr.DataTree)

    # Access the dataset from the root node
    ds = dt.to_dataset()
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "height", "width")
    assert "time" in ds.coords
    assert ds.sizes["time"] == 3


@pytest.mark.live
def test_open_uda_dataset_video_frame():
    """Test loading a single video frame from a real UDA server."""
    ds = xr.open_dataset(
        "uda://rba:30421", engine="uda", drop_variables=None, frame_number=1
    )
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "height", "width")
    assert "time" in ds.coords
    assert ds.sizes["time"] == 1


@pytest.mark.live
def test_real_data_2d():
    """Test loading real 2D data from a live UDA server."""
    ds = xr.open_dataset("uda://AYE_TE:30421", engine="uda")
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "radial index")
    assert "time" in ds.coords
    assert "radial index" in ds.coords


def test_remapped_signal_uses_legacy_name(mocker):
    """Requesting a canonical signal name for an old shot should transparently
    remap it to the legacy stored name before calling the UDA client."""

    # Build a minimal in-memory mapping:
    #   canonical "AYC_TE" -> "AYC_TE" for shots 22832-30471
    #                       -> "ATM_TE" for shots 1-22831
    fake_mappings = SignalMappings(
        mappings={
            "AYC_TE": [
                SignalRange(shot_min=22832, shot_max=30471, name="AYC_TE"),
                SignalRange(shot_min=1, shot_max=22831, name="ATM_TE"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]
    mock_signal.units = "eV"
    mock_signal.time = Mock(label="time", data=np.array([0.0, 1.0, 2.0]))
    mock_signal.errors = Mock(data=np.array([0.01, 0.01, 0.01]))

    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    # Shot 15000 is in the legacy range -> should call client with "ATM_TE"
    ds = xr.open_dataset("uda://AYC_TE:15000", engine="uda")

    mock_client.get.assert_called_once_with("ATM_TE", "15000")
    assert ds["data"].attrs["uda_name"] == "ATM_TE"


def test_remapped_signal_uses_modern_name(mocker):
    """Requesting a canonical signal name for a recent shot should keep the
    modern stored name."""

    fake_mappings = SignalMappings(
        mappings={
            "AYC_TE": [
                SignalRange(shot_min=22832, shot_max=30471, name="AYC_TE"),
                SignalRange(shot_min=1, shot_max=22831, name="ATM_TE"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]
    mock_signal.units = "eV"
    mock_signal.time = Mock(label="time", data=np.array([0.0, 1.0, 2.0]))
    mock_signal.errors = Mock(data=np.array([0.01, 0.01, 0.01]))

    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    # Shot 30000 is in the modern range -> should call client with "AYC_TE"
    ds = xr.open_dataset("uda://AYC_TE:30000", engine="uda")

    mock_client.get.assert_called_once_with("AYC_TE", "30000")
    assert ds["data"].attrs["uda_name"] == "AYC_TE"


def test_mastu_hcam_remapped_to_sanx_name(mocker):
    """For a MASTU shot in the SAnx05 range, the canonical HCAM channel name
    should be remapped to the SAnx05 digitiser path."""

    # /XSX/HCAM/L/CH01/DATA is the canonical name; for shots 46353-49476 the
    # data was stored as /xsx/SAnx05-01/ch14.
    fake_mappings = SignalMappings(
        mappings={
            "/XSX/HCAM/L/CH01/DATA": [
                SignalRange(
                    shot_min=49904, shot_max=51056, name="/XSX/HCAM/L/CH01/DATA"
                ),
                SignalRange(shot_min=49476, shot_max=49904, name="/XSX/HCAM/L/CH01"),
                SignalRange(shot_min=46353, shot_max=49476, name="/xsx/SAnx05-01/ch14"),
                SignalRange(shot_min=44395, shot_max=46353, name="/XSX/HCAM/L/CH01"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]
    mock_signal.units = "V"
    mock_signal.time = Mock(label="time", data=np.array([0.0, 1.0, 2.0]))
    mock_signal.errors = Mock(data=np.array([0.01, 0.01, 0.01]))

    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    # Shot 47000 falls in the SAnx05 range -> should call client with ch14 path
    ds = xr.open_dataset("uda:///XSX/HCAM/L/CH01/DATA:47000", engine="uda")

    mock_client.get.assert_called_once_with("/xsx/SAnx05-01/ch14", "47000")
    assert ds["data"].attrs["uda_name"] == "/xsx/SAnx05-01/ch14"


def test_mastu_hcam_modern_name_passthrough(mocker):
    """For the most-recent MASTU shot range, the canonical HCAM channel name
    should be used unchanged (it already matches the stored path)."""

    fake_mappings = SignalMappings(
        mappings={
            "/XSX/HCAM/L/CH01/DATA": [
                SignalRange(
                    shot_min=49904, shot_max=51056, name="/XSX/HCAM/L/CH01/DATA"
                ),
                SignalRange(shot_min=49476, shot_max=49904, name="/XSX/HCAM/L/CH01"),
                SignalRange(shot_min=46353, shot_max=49476, name="/xsx/SAnx05-01/ch14"),
                SignalRange(shot_min=44395, shot_max=46353, name="/XSX/HCAM/L/CH01"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0, 3.0])
    mock_signal.shape = (3,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0, 2.0])
    mock_signal.dims = [dim1]
    mock_signal.units = "V"
    mock_signal.time = Mock(label="time", data=np.array([0.0, 1.0, 2.0]))
    mock_signal.errors = Mock(data=np.array([0.01, 0.01, 0.01]))

    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    # Shot 50000 is in the most-recent range -> canonical name used as-is
    ds = xr.open_dataset("uda:///XSX/HCAM/L/CH01/DATA:50000", engine="uda")

    mock_client.get.assert_called_once_with("/XSX/HCAM/L/CH01/DATA", "50000")
    assert ds["data"].attrs["uda_name"] == "/XSX/HCAM/L/CH01/DATA"


def test_unmapped_signal_passes_through(mocker):
    """A signal not present in the mappings should be forwarded as-is."""

    fake_mappings = SignalMappings(mappings={})
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock(spec=pyuda.Signal)
    mock_signal.data = np.array([1.0, 2.0])
    mock_signal.shape = (2,)
    dim1 = Mock(label="time")
    dim1.data = np.array([0.0, 1.0])
    mock_signal.dims = [dim1]
    mock_signal.units = "A"
    mock_signal.time = Mock(label="time", data=np.array([0.0, 1.0]))
    mock_signal.errors = Mock(data=np.array([0.1, 0.1]))

    mock_client = Mock()
    mock_client.get.return_value = mock_signal
    mocker.patch("pyuda.Client", return_value=mock_client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    ds = xr.open_dataset("uda://SOME_UNKNOWN_SIGNAL:30421", engine="uda")

    mock_client.get.assert_called_once_with("SOME_UNKNOWN_SIGNAL", "30421")
    assert ds["data"].attrs["uda_name"] == "SOME_UNKNOWN_SIGNAL"


def test_resolve_ignores_case_and_leading_slash():
    mappings = SignalMappings(
        mappings={
            "/ESM/VLOOP/DYNAMIC": [
                SignalRange(shot_min=1, shot_max=10, name="/ESM/VLOOP_DYNAMIC")
            ]
        }
    )
    for name in ("/ESM/VLOOP/DYNAMIC", "esm/vloop/dynamic", "/esm/vloop/dynamic"):
        assert mappings.resolve(name, 5) == "/ESM/VLOOP_DYNAMIC"
    assert mappings.resolve("esm/vloop/dynamic", 11) is None


def test_bundled_mastu_renames():
    mappings = SignalMappings.from_file()
    # Shot with only the old name in the catalogue.
    assert mappings.resolve("esm/vloop/dynamic", 43477) == "/ESM/VLOOP_DYNAMIC"
    assert mappings.resolve("esm/density/nebar", 43482) == "/ESM/NEBAR"
    assert mappings.resolve("xdc/ai/cpu1/rogext_TF", 40240) == "/XDC/AI/CPU1/TF_CURRENT"
    # Shots with the new name are not remapped.
    assert mappings.resolve("esm/vloop/dynamic", 43478) is None
    assert mappings.resolve("esm/vloop/dynamic", 52000) is None
    assert mappings.resolve("xdc/ai/cpu1/rogext_TF", 40300) is None


def _signal_with_meta(meta):
    """A 1D mock signal whose ``meta`` is what pyuda hands back with get_meta set."""
    signal = Mock(spec=pyuda.Signal)
    signal.data = np.array([1.0, 2.0, 3.0])
    signal.dims = [Mock(label="time", data=np.array([0.0, 1.0, 2.0]))]
    signal.units = "T"
    signal.time = Mock(label="time", data=np.array([0.0, 1.0, 2.0]))
    signal.errors = Mock(data=np.array([0.0, 0.0, 0.0]))
    signal.meta = meta
    return signal


def _open_signal(mocker, url, meta):
    """Open ``url`` against a mock client returning a signal with ``meta``."""
    client = Mock()
    client.get.return_value = _signal_with_meta(meta)
    client.get_property.return_value = False
    mocker.patch("pyuda.Client", return_value=client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )
    return xr.open_dataset(url, engine="uda"), client


def test_mast_signal_records_file_and_pass(mocker):
    """A MAST server reports the file name, format, pass and pass date."""
    meta = {
        "signal_name": "EFM_BVAC_VAL",
        "signal_alias": "EFM_BVAC_VAL",
        "path": "",
        "filename": "efm0247.79",
        "format": "IDA3",
        "exp_number": 24779,
        "pass": 1,
        "pass_date": "2010-07-15",
    }
    ds, _ = _open_signal(mocker, "uda://EFM_BVAC_VAL:24779", meta)

    attrs = ds["data"].attrs
    assert attrs["uda_filename"] == "efm0247.79"
    assert attrs["uda_format"] == "IDA3"
    assert attrs["uda_pass"] == 1
    assert attrs["uda_pass_date"] == "2010-07-15"
    assert "uda_path" not in attrs


def test_mastu_signal_records_path_and_pass(mocker):
    """A MAST-U server reports the full path and the pass, as bytes, and leaves the
    file name and pass date empty."""
    meta = {
        "signal_name": b"/AMC/PLASMA_CURRENT",
        "signal_alias": b"",
        "path": b"/net/raidsrvr/data/MAST_Data/49000/LATEST/amc049000.nc",
        "filename": b"",
        "format": b"NETCDF",
        "exp_number": 49000,
        "pass": 4,
        "pass_date": b"",
    }
    ds, _ = _open_signal(mocker, "uda:///AMC/PLASMA_CURRENT:49000", meta)

    attrs = ds["data"].attrs
    assert attrs["uda_path"] == "/net/raidsrvr/data/MAST_Data/49000/LATEST/amc049000.nc"
    assert attrs["uda_format"] == "NETCDF"
    assert attrs["uda_pass"] == 4
    assert "uda_filename" not in attrs
    assert "uda_pass_date" not in attrs


def test_raw_signal_records_no_pass(mocker):
    """A raw file has pass -1, which means no pass, so neither the pass nor its date
    is recorded."""
    meta = {
        "path": "",
        "filename": "xma0247.79",
        "format": "IDA3",
        "pass": -1,
        "pass_date": "2010-03-16",
    }
    ds, _ = _open_signal(mocker, "uda://XMA_CCBV01:24779", meta)

    attrs = ds["data"].attrs
    assert attrs["uda_filename"] == "xma0247.79"
    assert "uda_pass" not in attrs
    assert "uda_pass_date" not in attrs


def test_server_reporting_no_file_or_pass_adds_no_attributes(mocker):
    """A server that reports no file or pass leaves the attributes as before."""
    ds, _ = _open_signal(mocker, "uda://ip:30421", {})

    assert set(ds["data"].attrs) == {"units", "uda_name"}


def test_pass_in_url_is_requested(mocker):
    """A pass after the shot number is sent to UDA as ``shot/pass``."""
    ds, client = _open_signal(mocker, "uda://EFM_BVAC_VAL:24779/0", {"pass": 0})

    client.get.assert_called_once_with("EFM_BVAC_VAL", "24779/0")
    assert ds["data"].attrs["uda_pass"] == 0


def test_another_pass_from_the_server_is_refused(mocker):
    """A server whose catalogue ignores the pass would return another one, so a
    reported pass that differs from the one asked for is an error."""
    with pytest.raises(
        RuntimeError, match="asked for pass 0, but the server returned pass 1"
    ):
        _open_signal(mocker, "uda://EFM_BVAC_VAL:24779/0", {"pass": 1})


def test_no_pass_from_the_server_is_refused(mocker):
    """A server that reports no pass can't show it read the one asked for."""
    with pytest.raises(
        RuntimeError, match="asked for pass 0, but the server returned no pass"
    ):
        _open_signal(mocker, "uda://EFM_BVAC_VAL:24779/0", {})


def test_get_meta_is_put_back(mocker):
    """get_meta is set for the request and then put back as it was."""
    _, client = _open_signal(mocker, "uda://ip:30421", {})

    assert client.set_property.call_args_list == [
        call("get_meta", True),
        call("get_meta", False),
    ]


def test_get_meta_is_put_back_when_the_request_fails(mocker):
    """get_meta is put back even when UDA refuses the request."""
    client = Mock()
    client.get.side_effect = pyuda.ServerException("Signal not found")
    client.get_property.return_value = False
    mocker.patch("pyuda.Client", return_value=client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    with pytest.raises(RuntimeError):
        xr.open_dataset("uda://invalid_signal:99999", engine="uda")

    assert client.set_property.call_args_list[-1] == call("get_meta", False)


def test_anything_but_a_signal_is_refused(mocker):
    """client.get can return a Video, StructuredData, String or Tree, which can't be
    read as a signal."""
    client = Mock()
    client.get.return_value = Mock(spec=pyuda.StructuredData)
    mocker.patch("pyuda.Client", return_value=client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Signal",
    )

    with pytest.raises(RuntimeError, match="not a signal"):
        xr.open_dataset("uda://ip:30421", engine="uda")


def test_pass_for_image_is_refused(mocker):
    """Images are read without a pass, so asking for one is an error."""
    client = Mock()
    mocker.patch("pyuda.Client", return_value=client)
    mocker.patch(
        "uda_xarray.main.UDABackendEntrypoint._get_signal_type",
        return_value="Image",
    )

    with pytest.raises(ValueError, match="a pass can only be given for signals"):
        xr.open_dataset("uda://rba:30421/0", engine="uda")

    client.get_images.assert_not_called()


@pytest.mark.live
def test_real_mast_passes():
    """MAST shot 24779 has two passes of EFM_BVAC_VAL, both in the same file name."""
    latest = xr.open_dataset("uda://EFM_BVAC_VAL:24779", engine="uda")
    first = xr.open_dataset("uda://EFM_BVAC_VAL:24779/0", engine="uda")

    assert latest["data"].attrs["uda_pass"] == 1
    assert first["data"].attrs["uda_pass"] == 0
    assert first["data"].attrs["uda_filename"] == "efm0247.79"
    assert first["data"].attrs["uda_pass_date"] == "2010-03-16"


@pytest.mark.live
def test_real_mastu_pass():
    """A MAST-U pass is reported with the path of that pass's file."""
    ds = xr.open_dataset("uda:///AMC/PLASMA_CURRENT:49000/0", engine="uda")

    assert ds["data"].attrs["uda_pass"] == 0
    assert ds["data"].attrs["uda_path"].endswith("/49000/Pass0/amc049000.nc")
