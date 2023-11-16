import logging
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.pydantic_v1 import root_validator

from .covalent import Result, _check_covalent_installed, _CovalentBase

# Handle case that ``covalent-cloud`` is not installed.
# Prevent import errors when loading this module.
_COVALENT_CLOUD_SDK_INSTALLED = False
try:
    import covalent_cloud as cc
    from covalent_cloud import CloudExecutor
    from covalent_cloud.dispatch_management.results_manager import FutureResult
    _COVALENT_CLOUD_SDK_INSTALLED = True
except ImportError:
    _CC_EXECUTOR_TYPE = Any
    _CC_RESULT_TYPE = Any
else:
    _CC_EXECUTOR_TYPE = CloudExecutor
    _CC_RESULT_TYPE = FutureResult


logger = logging.getLogger(__name__)


class CovalentCloud(_CovalentBase):

    """Covalent Cloud wrapper for LLMs.

    Requires the ``covalent-cloud`` Python package:
        .. code-block:: bash

            pip install covalent-cloud

    This wrapper can be used to run either a single function, or an entire
    Covalent workflow (i.e. a ``Lattice``) on the Covalent Cloud platform.

    Example:
        .. code-block:: python

            # Single function example.

            from langchain.llms import CovalentCloud
            import covalent_cloud as cc

            cc.save_api_key(API_KEY)
            executor = cc.CloudExecutor(env="my-env", num_cpus=12, num_gpus=6)
            covalent = CovalentCloud(func=my_func, executor=executor)

    Example:
        .. code-block:: python

            # Covalent Cloud workflow example.

            from langchain.llms import CovalentCloud
            import covalent as ct
            import covalent_cloud as cc

            small_executor = cc.CloudExecutor(env="my-env", num_cpus=2)
            medium_executor = cc.CloudExecutor(env="my-other-env", num_cpus=4)
            gpu_executor = cc.CloudExecutor(env="my-env", num_cpus=12, num_gpus=6)

            @ct.electron(executor=light_executor)
            def task(*inputs, flag=True):
                ...

            @ct.electron(executor=medium_executor)
            def other_task(filename, save=True, folder="my-folder"):
                ...

            @ct.electron(executor=gpu_executor)
            def gpu_task(**options):
                ...

            @ct.lattice(executor=small_executor, workflow_executor=small_executor)
            def covalent_workflow(prompt):
                # uses tasks defined above
                ...

            covalent = CovalentCloud(func=covalent_workflow, api_key=API_KEY)
    """

    def __init__(self, **kwargs):
        _check_covalent_installed(cloud=True)
        super().__init__(**kwargs)

        if self.api_key is not None:
            # Update Covalent Cloud API key.
            cc.save_api_key(self.api_key)

    @root_validator(pre=True)
    def validate_executor(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Make sure the ``executor`` attribute is a Covalent Cloud ``CloudExecutor`` instance."""

        executor = values.get("executor")

        # Skip check if no executor is specified.
        if executor is not None:
            if not isinstance(executor, _CC_EXECUTOR_TYPE):
                _type = type(executor).__name__
                raise ValueError(
                    f"Invalid executor ({_type}) for Covalent Cloud."
                )
        return values

    @property
    def _llm_type(self) -> str:
        "covalent-cloud"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

        # Convert ``self.func`` to a Covalent lattice.
        lattice = self._to_lattice()

        # Dispatch the lattice.
        dispatch_id = cc.dispatch(
            lattice, **self.dispatch_kwargs)(prompt, **kwargs)

        # Request the result...
        result_obj = cc.get_result(dispatch_id, wait=True)

        if result_obj.status != Result.COMPLETED:
            # Download available information.
            debug_attrs = ["result", "inputs", "error", "nodes"]
            _download_attributes(result_obj, *debug_attrs)
            raise RuntimeError(
                f"Dispatch did not succeed. Returned result object:\n\n{result_obj}"
            )

        # Download result.
        _download_attributes(result_obj, "result")
        return result_obj.result.value


def _download_attributes(result_obj: _CC_RESULT_TYPE, *names) -> None:
    """Download select attributes of a ``FutureResult``."""

    for name in names:
        if name == "nodes":
            node_outputs = result_obj.get_all_node_outputs()
            for v in node_outputs.values():
                v.load()
        else:
            attr = getattr(result_obj, name)
            attr.load()
