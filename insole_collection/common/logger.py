# flat_logger.py
"""
Flat CSV logger for plantar insole frames.

Each call to `log_frame` writes ONE frame as it arrives:
    - timestamp
    - client id (0 = reference  1 = crosstalk)
    - flattened sensor values (e.g. 253 channels)
"""

import csv
import time
from pathlib import Path
import numpy as np


class FlatCsvFrameLogger:
    """
    Simple CSV logger for 1D sensor frames.

    Usage:
        logger = FlatCsvFrameLogger("insole_frames_flat.csv")
        logger.log_frame(client=0, values=some_1d_array)
        ...
        logger.close()
    """

    def __init__(self, filepath: str, n_channels: int, add_header: bool = True):
        """
        Args:
            filepath   : path to CSV file
            n_channels : number of sensor values per frame (e.g. 253)
            add_header : whether to write a header row
        """
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_channels = int(n_channels)

        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)

        if add_header:
            header = ["timestamp", "client"]
            header += [f"ch_{i}" for i in range(self.n_channels)]
            self.writer.writerow(header)
    
    def log_frame(self, client: int, values: np.ndarray):
        """
        Args:
            client : 0 or 1 (or any integer identifying the source)
            values : 1D or ND numpy array of length == n_channels
        """
        arr = np.asarray(values).ravel().astype(float)

        if arr.size != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} values, got {arr.size}"
            )

        ts = time.time()
        row = [ts, int(client)] + arr.tolist()
        self.writer.writerow(row)

    def close(self):
        """Flush and close the underlying CSV file."""
        self.file.flush()
        self.file.close()

class csv_to_npz:
    def __init__(self, arr, threshold):
        self.arr = arr
        self.threshold = threshold

