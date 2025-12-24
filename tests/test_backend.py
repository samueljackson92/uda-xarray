import xarray as xr


def test_open_uda_dataset():
    ds = xr.open_dataset("uda://ip:30421", engine="uda")
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time",)
    assert "time" in ds.coords

    assert ds["data"].shape == ds["time"].shape
    assert "units" in ds["data"].attrs
    assert ds["data"].attrs["uda_name"] == "ip"


def test_open_uda_dataset_invalid_signal():
    try:
        xr.open_dataset("uda://invalid_signal:99999", engine="uda")
    except RuntimeError as e:
        assert "Could not open UDA dataset" in str(e)
    else:
        assert False, "Expected RuntimeError was not raised"


def test_2d_signal_not_supported():
    try:
        xr.open_dataset("uda://AYE_TE:30421", engine="uda")
    except NotImplementedError as e:
        assert "UDA backend currently only supports 1D signals" in str(e)
    else:
        assert False, "Expected NotImplementedError was not raised"
