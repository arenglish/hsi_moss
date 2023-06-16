from typing import TypeVar, NamedTuple, Callable, Generic, List
from .dataset import DatasetOutput
from enum import Enum
from .utils.logger import log
import warnings


class PipelineOptions(NamedTuple):
    overwrite_outputs = True
    preview_channels = (650, 550, 450)
    write_outputs: bool = True


PipelineState = TypeVar("PipelineState")


class PIPELINE_HOOKS(Enum):
    start = 0
    every = 1
    end = 2


class PipelineOperator(NamedTuple, Generic[PipelineState]):
    name: str
    input: DatasetOutput
    output: DatasetOutput | List[DatasetOutput]
    run: Callable[[PipelineState, PipelineOptions, DatasetOutput, DatasetOutput], None]
    skip_if_exists: bool = False
    hook = PIPELINE_HOOKS.every


class Pipeline(Generic[PipelineState]):
    options: PipelineOptions
    state: PipelineState
    op_type = PipelineOperator[PipelineState]
    operation_sequence: List[PipelineOperator[PipelineState]]

    def __init__(self, options: PipelineOptions = PipelineOptions()):
        self.options = options
        self.operation_sequence = []

    def register_operator(self, operator: PipelineOperator[PipelineState]):
        self.operation_sequence.append(operator)

    def register_operators(self, operators: List[PipelineOperator[PipelineState]]):
        for op in operators:
            self.operation_sequence.append(op)

    def initialize_state(self, state: PipelineState):
        self.state = state

    def run(self, overwrite_existing_outputs=True, _ops: List[str] = None):
        if _ops is None:
            ops = self.operation_sequence
        else:
            ops = [op for op in self.operation_sequence if op.name in _ops]

        for op in ops:
            # try:
            if (
                overwrite_existing_outputs is True or op.skip_if_exists is False
            ) or not op.output.filepath.exists():
                print(f"Running op:{op.name} on {op.input.filepath}")
                op.run(self.state, self.options, op.input, op.output)
            else:
                print(f"Skipping op:{op.name} on {op.input.filepath}")
            # except Exception as e:
            #     log.error(f"ERROR: Failed op:{op.name} on {op.input.filepath}")
