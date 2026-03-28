import pytest
import xarray as xr
import numpy as np
import pyuda
from unittest.mock import Mock, patch

from uda_xarray.mappings import SignalMappings, SignalRange


def test_open_uda_dataset(mocker):
    # Create mock signal object
    mock_signal = Mock()
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
    mock_client.get.assert_called_once_with("ip", 30421)

    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time",)
    assert "time" in ds.coords

    assert ds["data"].shape == ds["time"].shape
    assert "units" in ds["data"].attrs
    assert ds["data"].attrs["uda_name"] == "ip"


def test_open_uda_dataset_2d(mocker):
    # Create mock 2D signal object
    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("AYE_TE", 30421)

    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "channel")
    assert "time" in ds.coords
    assert "channel" in ds.coords


def test_open_uda_dataset_video(mocker):
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
    # Create mock signal object
    mock_signal = Mock()
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
    mock_client.get.assert_called_once_with("ip", 30421)

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
    # Create mock 2D signal object
    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("AYE_TE", 30421)

    # Verify it's a DataTree object
    assert isinstance(dt, xr.DataTree)

    # Access the dataset from the root node
    ds = dt.to_dataset()
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "channel")
    assert "time" in ds.coords
    assert "channel" in ds.coords


def test_open_datatree_video(mocker):
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


@pytest.mark.skip(reason="Requires access to real UDA server with video data")
def test_open_uda_dataset_video_frame(mocker):
    ds = xr.open_dataset(
        "uda://rba:30421", engine="uda", drop_variables=None, frame_number=1
    )
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "height", "width")
    assert "time" in ds.coords
    assert ds.sizes["time"] == 1


@pytest.mark.skip(reason="Requires access to real UDA server with video data")
def test_real_data_2d():
    ds = xr.open_dataset("uda://AYE_TE:30421", engine="uda")
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time", "radial_index")
    assert "time" in ds.coords
    assert "radial_index" in ds.coords


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

    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("ATM_TE", 15000)
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

    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("AYC_TE", 30000)
    assert ds["data"].attrs["uda_name"] == "AYC_TE"


def test_mastu_hcam_remapped_to_sanx_name(mocker):
    """For a MASTU shot in the SAnx05 range, the canonical HCAM channel name
    should be remapped to the SAnx05 digitiser path."""

    # /XSX/HCAM/L/CH01/DATA is the canonical name; for shots 46353-49476 the
    # data was stored as /xsx/SAnx05-01/ch14.
    fake_mappings = SignalMappings(
        mappings={
            "/XSX/HCAM/L/CH01/DATA": [
                SignalRange(shot_min=49904, shot_max=51056, name="/XSX/HCAM/L/CH01/DATA"),
                SignalRange(shot_min=49476, shot_max=49904, name="/XSX/HCAM/L/CH01"),
                SignalRange(shot_min=46353, shot_max=49476, name="/xsx/SAnx05-01/ch14"),
                SignalRange(shot_min=44395, shot_max=46353, name="/XSX/HCAM/L/CH01"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("/xsx/SAnx05-01/ch14", 47000)
    assert ds["data"].attrs["uda_name"] == "/xsx/SAnx05-01/ch14"


def test_mastu_hcam_modern_name_passthrough(mocker):
    """For the most-recent MASTU shot range, the canonical HCAM channel name
    should be used unchanged (it already matches the stored path)."""

    fake_mappings = SignalMappings(
        mappings={
            "/XSX/HCAM/L/CH01/DATA": [
                SignalRange(shot_min=49904, shot_max=51056, name="/XSX/HCAM/L/CH01/DATA"),
                SignalRange(shot_min=49476, shot_max=49904, name="/XSX/HCAM/L/CH01"),
                SignalRange(shot_min=46353, shot_max=49476, name="/xsx/SAnx05-01/ch14"),
                SignalRange(shot_min=44395, shot_max=46353, name="/XSX/HCAM/L/CH01"),
            ]
        }
    )
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("/XSX/HCAM/L/CH01/DATA", 50000)
    assert ds["data"].attrs["uda_name"] == "/XSX/HCAM/L/CH01/DATA"


def test_unmapped_signal_passes_through(mocker):
    """A signal not present in the mappings should be forwarded as-is."""

    fake_mappings = SignalMappings(mappings={})
    mocker.patch("uda_xarray.main._MAPPINGS", fake_mappings)

    mock_signal = Mock()
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

    mock_client.get.assert_called_once_with("SOME_UNKNOWN_SIGNAL", 30421)
    assert ds["data"].attrs["uda_name"] == "SOME_UNKNOWN_SIGNAL"
