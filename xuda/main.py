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
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ) -> xr.Dataset:
        """Open a UDA dataset given a signal name and shot number."""
        name, shot = filename_or_obj.rsplit(":", maxsplit=1)
        name = name.replace("uda://", "")
        shot = int(shot)

        client = pyuda.Client()
        try:
            signal = client.get(name, shot)
        except (pyuda.ServerException, pyuda.cpyuda.ClientException) as e:
            raise RuntimeError(f"Could not open UDA dataset {filename_or_obj}") from e

        item = xr.DataArray(
            signal.data,
            dims=["time"],
            coords={"time": signal.time.data},
            attrs=dict(units=signal.units, uda_name=name),
        )

        error = xr.DataArray(
            signal.errors.data,
            dims=["time"],
            coords={"time": signal.time.data},
            attrs=dict(units=signal.units, uda_name=name),
        )

        return xr.Dataset(data_vars={"data": item, "error": error})

    def open_datatree(self, filename_or_obj, *, drop_variables=None):
        raise NotImplementedError("UDA backend does not support open_datatree")

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        return filename_or_obj.startswith("uda://")

    description = "Use UDA data in Xarray"

    url = "https://link_to/your_backend/documentation"
