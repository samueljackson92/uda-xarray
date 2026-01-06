"""Xarray UDA backend entrypoint."""

from enum import Enum
from typing import Optional

import numpy as np
import pyuda
import xarray as xr
from mast.mast_client import ListType
from xarray.backends import BackendEntrypoint


class SignalType(str, Enum):
    """Enum for UDA signal types."""

    SIGNAL = "Signal"
    IMAGE = "Image"


Signal = pyuda.Signal
Video = pyuda.Video


class UDABackendEntrypoint(BackendEntrypoint):
    """Xarray UDA backend entrypoint."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        frame_number: Optional[int] = None,  # noqa: F821
    ) -> xr.Dataset:
        """Open a UDA dataset given a signal name and shot number.

        Parameters
        ----------
        filename_or_obj : str
            UDA dataset specified as uda://<signal_name>:<shot_number>
        drop_variables : list, optional
            Variables to drop from the dataset (not used).
        frame_number : int, optional
            Frame number to extract from an image signal (if applicable).
        """

        if ":" not in filename_or_obj:
            raise ValueError(
                "UDA dataset must be specified as uda://<signal_name>:<shot_number>"
            )

        if "uda://" not in filename_or_obj:
            raise ValueError("UDA dataset must start with the uda:// scheme")

        name, shot = filename_or_obj.rsplit(":", maxsplit=1)
        name = name.replace("uda://", "")
        shot = int(shot)

        client = pyuda.Client()
        signal_type = self._get_signal_type(client, name, shot)

        try:
            if signal_type == SignalType.SIGNAL:
                signal = client.get(name, shot)
                dataset = self._handle_signal(name, signal)
            elif signal_type == SignalType.IMAGE:
                kwargs = {}
                if frame_number is not None:
                    kwargs["frame_number"] = frame_number
                signal = client.get_images(name, shot, **kwargs)
                dataset = self._handle_images(name, signal, frame_number=frame_number)
            else:
                raise NotImplementedError(f"Signal type {signal_type} not supported")
        # pylint: disable=c-extension-no-member
        except (pyuda.ServerException, pyuda.cpyuda.ClientException) as e:
            raise RuntimeError(f"Could not open UDA dataset {filename_or_obj}") from e

        return dataset

    def _handle_signal(self, name: str, signal: Signal) -> xr.Dataset:
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

    def _handle_images(
        self, name: str, video: Video, frame_number: Optional[int] = None
    ) -> xr.Dataset:
        attrs = {
            name: getattr(video, name)
            for name in dir(video)
            if not name.startswith("_") and not callable(getattr(video, name))
        }

        attrs.pop("frame_times")
        attrs.pop("frames")

        attrs["CLASS"] = "IMAGE"
        attrs["IMAGE_VERSION"] = "1.2"

        time = np.atleast_1d(video.frame_times)
        if frame_number is not None:
            time = time[frame_number : frame_number + 1]
        coords = {"time": xr.DataArray(time, dims=["time"], attrs={"units": "s"})}

        if video.is_color:
            frames = [np.dstack((frame.r, frame.g, frame.b)) for frame in video.frames]
            frames = np.stack(frames)
            if frames.shape[1] != video.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width", "channel"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_TRUECOLOR"
            attrs["INTERLACE_MODE"] = "INTERLACE_PIXEL"
        else:
            frames = [frame.k for frame in video.frames]
            frames = np.stack(frames)
            frames = np.atleast_3d(frames)
            if frames.shape[1] != video.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_INDEXED"

        dataset = xr.DataArray(frames, dims=dim_names, coords=coords, attrs=attrs)
        dataset = dataset.to_dataset(name="data")
        dataset.attrs["uda_name"] = name
        return dataset

    def _get_signal_type(
        self, client: pyuda.Client, name: str, shot: int
    ) -> SignalType:
        sources = client.list(ListType.SOURCES, shot=shot)
        source_types = {s.source_alias: s.type for s in sources}

        if name in source_types and source_types[name] == "Image":
            return SignalType.IMAGE
        return SignalType.SIGNAL

    def open_datatree(self, filename_or_obj, *, drop_variables=None):
        raise NotImplementedError("UDA backend does not support open_datatree")

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        return filename_or_obj.startswith("uda://")

    description = "Use UDA data in Xarray"

    url = "https://github.com/samueljackson92/uda-xarray"
