"""Xarray UDA backend entrypoint."""

import pyuda
import xarray as xr
from xarray.backends import BackendEntrypoint


class UDABackendEntrypoint(BackendEntrypoint):
    """Xarray UDA backend entrypoint."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ) -> xr.Dataset:
        """Open a UDA dataset given a signal name and shot number."""
        name, shot = filename_or_obj.rsplit(":", maxsplit=1)
        name = name.replace("uda://", "")
        shot = int(shot)

        client = pyuda.Client()
        try:
            signal = client.get(name, shot)
        # pylint: disable=c-extension-no-member
        except (pyuda.ServerException, pyuda.cpyuda.ClientException) as e:
            raise RuntimeError(f"Could not open UDA dataset {filename_or_obj}") from e

        dim_data = {dim.label: dim.data for dim in signal.dims}

        # Rename time dimension to just "time" if we can.
        if signal.time.label in dim_data:
            dim_data["time"] = dim_data.pop(signal.time.label)

        item = xr.DataArray(
            signal.data,
            coords=dim_data,
            attrs={"units": signal.units, "uda_name": name},
        )

        error = xr.DataArray(
            signal.errors.data,
            coords=dim_data,
        )

        dataset = xr.Dataset(data_vars={"data": item, "error": error})
        return dataset

    def open_datatree(self, filename_or_obj, *, drop_variables=None):
        raise NotImplementedError("UDA backend does not support open_datatree")

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        return filename_or_obj.startswith("uda://")

    description = "Use UDA data in Xarray"

    url = "https://github.com/samueljackson92/uda-xarray"
