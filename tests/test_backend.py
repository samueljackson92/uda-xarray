import xarray as xr
import numpy as np
import pyuda
from unittest.mock import Mock


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

    ds = xr.open_dataset("uda://AYE_TE:30421", engine="uda")

    mock_client.get.assert_called_once_with("AYE_TE", 30421)

    assert ds["data"].name == "data"
    assert ds["data"].dims == ("channel", "time")
    assert "time" in ds.coords
    assert "channel" in ds.coords


def test_open_uda_dataset_invalid_signal(mocker):
    # Mock the pyuda Client to raise an exception
    mock_client = Mock()
    mock_client.get.side_effect = pyuda.ServerException("Signal not found")
    mocker.patch("pyuda.Client", return_value=mock_client)

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
