from pathlib import Path
from typing import Dict, Union, List


def find_files_mapping(
        base_dir: Union[str, Path],
        pattern: str,
        path_as_str: bool = False,
) -> Dict[str, Union[str, Path]]:
    """Find files in base_dir that match the pattern."""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    return {
        file.name.split('.')[0]: file.relative_to(base_dir) if not path_as_str else str(file.relative_to(base_dir))
            for file in base_dir.rglob(pattern)
        }



