# xuda

An xarray interface for UDA (Universal Data Access) that enables seamless integration of UDA data sources with the xarray ecosystem.

## Overview

xuda provides a backend plugin for xarray that allows you to access UDA data sources using xarray's familiar API. It automatically handles the conversion of UDA signals to xarray DataArrays and Datasets, making it easy to work with UDA data in scientific Python workflows.

## Features

- **xarray Integration**: Access UDA data using xarray's `open_dataset` function
- **Automatic Conversion**: Converts UDA signals to xarray DataArrays with proper coordinates and metadata
- **Error Handling**: Includes error data alongside signal data
- **URI-based Access**: Simple URI scheme (`uda://signal:shot`) for accessing data

## Installation

```bash
pip install xuda
```

Or using uv:

```bash
uv pip install xuda
```

## Usage

Open a UDA dataset using xarray:

```python
import xarray as xr

# Open a UDA signal by name and shot number
ds = xr.open_dataset("uda://ip:30421", engine="uda")

# Access the data
data = ds["data"]
errors = ds["error"]

# The dataset includes time coordinates
time = ds["time"]

# Access metadata
units = ds["data"].attrs["units"]
signal_name = ds["data"].attrs["uda_name"]
```

The URI format is: `uda://<signal_name>:<shot_number>`

## Data Structure

When you open a UDA dataset, xuda creates an xarray Dataset with:

- **data**: The signal data as a DataArray
- **error**: The error data as a DataArray
- **time**: Time coordinates (dimension)
- **attrs**: Metadata including units and UDA signal name

## Limitations

- Currently only supports 1D signals (2D signals will raise `NotImplementedError`)
- Requires a working UDA client connection

## Requirements

- Python >= 3.11, < 3.13
- uda >= 2.9.2
- xarray >= 2025.12.0

## Development

### Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/samueljackson92/tokamak-events.git
cd xuda
uv sync
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

The project uses ruff for linting and formatting, and pylint for additional checks:

```bash
# Run ruff
uv run ruff check xuda tests
uv run ruff format xuda tests

# Run pylint
uv run pylint xuda
```

## Contributing

Contributions are welcome! Please ensure:

1. Tests pass for any new functionality
2. Code follows the project's style guidelines (ruff and pylint)
3. Documentation is updated as needed

## License

MIT License