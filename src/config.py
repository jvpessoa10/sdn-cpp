from dataclasses import dataclass
from src.network import Network


@dataclass
class Config:
    n_controllers: int
    pop_size: int
    n_generations: int
    network: Network
    use_generic_operators: bool = False

    def name(self):
        return f"{self.network.topology}_c{self.n_controllers}_s{self.network.n_switches()}_pop{self.pop_size}_gen{self.n_generations}"
