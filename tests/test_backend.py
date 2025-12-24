import xarray as xr


def test_open_uda_dataset():
    ds = xr.open_dataset("uda://ip:30421", engine="uda")
    assert ds["data"].name == "data"
    assert ds["data"].dims == ("time",)
    assert "time" in ds.coords

    assert ds["data"].shape == ds["time"].shape
    assert "units" in ds["data"].attrs
    assert ds["data"].attrs["uda_name"] == "ip"
