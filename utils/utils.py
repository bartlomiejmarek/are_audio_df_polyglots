import csv
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional, Union

import torchaudio

from df_logger import time_logger


def listen_audio(audio_path: Union[str, Path]):
    """
    Listen to an audio file.
    """

    import IPython.display as ipd
    waveform, sample_rate = torchaudio.load(audio_path)
    ipd.Audio(audio_path, rate=sample_rate)


def append_to_csv(csv_file, headers, data):
    """Appends a row of data to a CSV file, adding headers if the file is empty.

    Args:
        csv_file (str): Path to the CSV file.
        headers (list): List of column headers.
        data (list): List of values to be written as a row.
    """

    try:
        # Check if the file exists and has content.
        with open(csv_file, 'r') as f:
            has_content = len(f.read()) > 0

        # Open the file in append mode ('a')
        with open(csv_file, 'a+', newline='') as f:
            writer = csv.writer(f)

            # If the file is empty or doesn't have headers, write the headers
            if not has_content:
                writer.writerow(headers)

            # Write the data row
            writer.writerow(data)

    except FileNotFoundError:
        # If file doesn't exist, create and write headers
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(data)


def timeit(title: str, field_name: Optional[str] = None) -> Callable:
    """
    A decorator that measures and prints the execution time of a class method with a custom title.
    :param title: The title of the log message.+
    :param field_name: The name of the field to be printed in the log message.
    :return: The decorated function.

    Example:
        class MyClass:
            def __init__(self):
                self.model_name = "MyModel"
                self.processor = SomeProcessor()

            @timeit("Processing data", "model_name")
            def method1(self, x, y):
                return x + y

            @timeit("Running processor", "processor")
            def method2(self):
                self.processor.process()
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def timeit_wrapper(self, *args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Format time using strftime
            time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
            if total_time < 1:  # If less than a second, show milliseconds
                time_str += f".{int(total_time * 1000):03d}"

            log_message = f'[{title}] Method {func.__name__} of class {self.__class__.__name__}'
            log_message += f' took {time_str}'

            if field_name:
                field_value = getattr(self, field_name, "Field not found")
                if isinstance(field_value, str):
                    field_info = field_value
                elif hasattr(field_value, '__class__'):
                    field_info = field_value.__class__.__name__
                else:
                    field_info = str(field_value)
                log_message += f' - {field_name}: {field_info}'

            time_logger.info(log_message)
            return result

        return timeit_wrapper

    return decorator
