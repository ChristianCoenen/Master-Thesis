from dataclasses import dataclass
from dataclasses import field


@dataclass
class Object:
    """
    Defines an object and its properties.
    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)
