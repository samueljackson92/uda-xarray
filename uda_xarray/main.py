"""Xarray UDA backend entrypoint."""

from enum import Enum
from typing import Optional

import os
import numpy as np
import pyuda
import xarray as xr
from mast.mast_client import ListType
from xarray.backends import BackendEntrypoint

from uda_xarray.mappings import SignalMappings

_MAPPINGS = SignalMappings.from_file()


# Set up UDA environment variables with defaults if not already set. This is required for
# the pyuda client to work correctly outside of Freia.
os.environ["UDA_HOST"] = os.environ.get("UDA_HOST", "uda2.mast.l")
os.environ["UDA_META_PLUGINNAME"] = os.environ.get("UDA_META_PLUGINNAME", "MASTU_DB")
os.environ["UDA_METANEW_PLUGINNAME"] = os.environ.get(
    "UDA_METANEW_PLUGINNAME", "MAST_DB"
)


class SignalType(str, Enum):
    """Enum for UDA signal types."""

    SIGNAL = "Signal"
    IMAGE = "Image"


Signal = pyuda.Signal
Video = pyuda.Video


def _text(value) -> str:
    """A metadata string from pyuda, which hands some of them over as bytes."""
    if isinstance(value, bytes):
        return value.decode()
    return value if isinstance(value, str) else ""


def _file_and_pass_attrs(meta: dict) -> dict:
    """The archive file and pass a signal was read from, as far as the server says.

    Servers fill in different fields and leave the rest empty: MAST gives the file name
    and pass date, MAST-U the full path. Only fields with a value become attributes, so a
    server that reports none of them adds none. A pass of -1 means the file has no pass,
    as for raw data, and then neither the pass nor its date is recorded.
    """
    attrs = {}
    for field in ("path", "filename", "format"):
        value = _text(meta.get(field))
        if value:
            attrs[f"uda_{field}"] = value

    pass_number = meta.get("pass")
    if isinstance(pass_number, int) and pass_number >= 0:
        attrs["uda_pass"] = pass_number
        pass_date = _text(meta.get("pass_date"))
        if pass_date:
            attrs["uda_pass_date"] = pass_date
    return attrs


class UDABackendEntrypoint(BackendEntrypoint):
    """Xarray UDA backend entrypoint."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        frame_number: Optional[int] = None,  # noqa: F821
        **kwargs,
    ) -> xr.Dataset:
        """Open a UDA dataset given a signal name, shot number and, optionally, a pass.

        Parameters
        ----------
        filename_or_obj : str
            UDA dataset specified as uda://<signal_name>:<shot_number>, or
            uda://<signal_name>:<shot_number>/<pass> to read a particular pass of a
            signal. Without a pass, UDA serves the latest one.
        drop_variables : list, optional
            Variables to drop from the dataset (not used).
        frame_number : int, optional
            Frame number to extract from an image signal (if applicable).
        """

        if ":" not in filename_or_obj:
            raise ValueError(
                "UDA dataset must be specified as "
                "uda://<signal_name>:<shot_number>[/<pass>]"
            )

        if "uda://" not in filename_or_obj:
            raise ValueError("UDA dataset must start with the uda:// scheme")

        name, shot_and_pass = filename_or_obj.rsplit(":", maxsplit=1)
        name = name.replace("uda://", "")
        shot, _, pass_number = shot_and_pass.partition("/")
        shot = int(shot)
        pass_number = int(pass_number) if pass_number else None

        name = _MAPPINGS.resolve(name, shot) or name

        client = pyuda.Client()
        signal_type = self._get_signal_type(client, name, shot)

        if signal_type == SignalType.IMAGE and pass_number is not None:
            raise ValueError(
                f"{filename_or_obj}: a pass can only be given for signals, not images"
            )

        try:
            if signal_type == SignalType.SIGNAL:
                signal = self._get_signal(client, name, shot, pass_number)
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

    def _get_signal(
        self, client: pyuda.Client, name: str, shot: int, pass_number: Optional[int]
    ) -> Signal:
        """Get a signal along with the archive file and pass it was read from.

        pyuda fills in ``signal.meta`` only while the client's ``get_meta`` property is
        set. Properties apply to every client in the process, so it is set for this
        request only and then put back as it was.

        ``client.get`` returns whatever the server sent: a Signal, or a Video,
        StructuredData, String or Tree. Anything other than a Signal is refused here.

        UDA sends a pass to the server's metadata catalogue, and each site's catalogue
        decides what it means. When a pass is asked for, the server must report that same
        pass back; otherwise a catalogue that ignored it could hand back another pass
        unnoticed.
        """
        shot_and_pass = str(shot) if pass_number is None else f"{shot}/{pass_number}"
        previous = client.get_property("get_meta")
        client.set_property("get_meta", True)
        try:
            signal = client.get(name, shot_and_pass)
        finally:
            client.set_property("get_meta", previous)

        if not isinstance(signal, Signal):
            raise RuntimeError(
                f"{name} for shot {shot}: UDA returned a {type(signal).__name__}, "
                "not a signal"
            )

        if pass_number is not None:
            reported = _file_and_pass_attrs(signal.meta).get("uda_pass")
            if reported != pass_number:
                returned = "no pass" if reported is None else f"pass {reported}"
                raise RuntimeError(
                    f"{name} for shot {shot}: asked for pass {pass_number}, "
                    f"but the server returned {returned}"
                )
        return signal

    def _handle_signal(self, name: str, signal: Signal) -> xr.Dataset:
        dim_data = {dim.label: dim.data for dim in signal.dims}

        # Rename time dimension to just "time" if we can.
        if signal.time.label in dim_data:
            dim_data = {"time": dim_data.pop(signal.time.label), **dim_data}

        item = xr.DataArray(
            signal.data,
            coords=dim_data,
            attrs={
                "units": signal.units,
                "uda_name": name,
                **_file_and_pass_attrs(signal.meta),
            },
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

    def open_datatree(self, filename_or_obj, *, drop_variables=None, **kwargs):
        dataset = self.open_dataset(
            filename_or_obj, drop_variables=drop_variables, **kwargs
        )
        return xr.DataTree(dataset)

    def open_groups_as_dict(self, filename_or_obj, **kwargs):
        """Open groups as a dictionary (not supported for UDA backend)."""
        raise NotImplementedError("UDA backend does not support opening groups")

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "frame_number"]
    supports_groups = True

    def guess_can_open(self, filename_or_obj):
        return filename_or_obj.startswith("uda://")

    description = "Use UDA data in Xarray"

    url = "https://github.com/samueljackson92/uda-xarray"
