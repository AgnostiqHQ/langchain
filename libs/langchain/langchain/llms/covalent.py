import logging
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Extra, Field, root_validator

# Handle case that ``covalent`` is not installed.
# Prevent import errors when loading this module.
_COVALENT_SDK_INSTALLED = False
try:
    import covalent as ct
    from covalent._results_manager.result import Result
    from covalent._workflow.electron import Electron
    from covalent._workflow.lattice import Lattice
    from covalent.executor.base import AsyncBaseExecutor, BaseExecutor
    from covalent_dispatcher._cli.service import _read_pid
    _COVALENT_SDK_INSTALLED = True
except ImportError:
    _ELECTRON_TYPE = Any
    _LATTICE_TYPE = Any
    _CT_ASYNC_EXECUTOR_TYPE = Any
    _CT_EXECUTOR_TYPE = Any
else:
    _ELECTRON_TYPE = Electron
    _LATTICE_TYPE = Lattice
    _CT_ASYNC_EXECUTOR_TYPE = AsyncBaseExecutor
    _CT_EXECUTOR_TYPE = BaseExecutor

# Handle case that ``covalent-cloud`` is not installed.
# Prevent import errors when loading this module.
_COVALENT_CLOUD_SDK_INSTALLED = False
try:
    from covalent_cloud import CloudExecutor
    _COVALENT_CLOUD_SDK_INSTALLED = True
except ImportError:
    _CC_EXECUTOR_TYPE = Any
else:
    _CC_EXECUTOR_TYPE = CloudExecutor

# Alias for brevity.
# Union of all valid executor types across Covalent and Covalent Cloud.
_EXECUTOR_TYPE = Union[str, _CT_ASYNC_EXECUTOR_TYPE,
                       _CT_EXECUTOR_TYPE, _CC_EXECUTOR_TYPE]


logger = logging.getLogger(__name__)


def _check_covalent_installed(cloud: bool) -> None:
    """Check that the Covalent SDK is installed."""
    if cloud:
        if not _COVALENT_CLOUD_SDK_INSTALLED:
            raise RuntimeError(
                "The Covalent Cloud SDK is not installed. To install, run\n\n"
                "pip install covalent-cloud"
            )
    else:
        if not _COVALENT_SDK_INSTALLED:
            raise RuntimeError(
                "The Covalent SDK is not installed. To install, run\n\n"
                "pip install covalent"
            )


class _CovalentBase(LLM):
    """Base class for ``Covalent`` and ``CovalentCloud``."""

    func: Union[Callable, _ELECTRON_TYPE, _LATTICE_TYPE]
    """
    Function, task, or workflow to run with Covalent.
    Must be callable as ``func(prompt, **kwargs)``.
    """

    executor: Optional[_EXECUTOR_TYPE] = None
    """
    Optional Covalent executor for ``func``.
    Ignored if ``func`` is a Covalent lattice.
    """

    lattice_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to ``ct.lattice``.
    Ignored if ``func`` is a Covalent lattice."""

    dispatch_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to ``ct.dispatch`` or ``cc.dispatch``."""

    api_key: Optional[str] = None
    """API key for Covalent Cloud. Required if not set manually with ``cc.save_api_key()``."""

    class Config:
        """Configuration for this Pydantic object."""

        extra = Extra.forbid

    def _covalent_server_ready(self) -> bool:
        """Returns True if a local Covalent Server is currently running."""

        _config = ct.get_config()["dispatcher"]

        pid = _read_pid(_config["cache_dir"] + "/ui.pid")
        server_up = psutil.pid_exists(pid) and pid != -1

        return server_up and _config.get("address") and _config.get("port")

    def _start_covalent(self) -> None:
        """Start local Covalent server."""

        if self._covalent_server_ready():
            return

        try:
            # Start local Covalent server. Suppress output.
            subprocess.run(
                "covalent start -d",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError("Failed to start Covalent.") from err

        while not self._covalent_server_ready():
            logger.debug("Waiting for Covalent server to start...")
            time.sleep(1)

    def _to_lattice(self) -> _LATTICE_TYPE:
        """Transform ``self.func`` into a Covalent lattice."""

        if isinstance(self.func, _LATTICE_TYPE):
            # Do nothing.
            return self.func

        if isinstance(self.func, _ELECTRON_TYPE):
            if self.executor is not None:
                # Override with user-specified the executor.
                self.func.set_metadata("executor", self.executor)
            return ct.lattice(self.func, **self.lattice_kwargs)

        return ct.lattice(ct.electron(self.func))


class Covalent(_CovalentBase):

    """Covalent wrapper for LLMs.

    Requires the ``covalent`` Python package:
        .. code-block:: bash

            pip install covalent

    This wrapper can be used to run either a single function, or an entire
    Covalent workflow (i.e. a ``Lattice``) across multiple local/cloud backends.

    Example:
        .. code-block:: python

            # Single function example.

            from langchain.llms import Covalent
            import covalent as ct

            executor = ct.executor.AWSBatchExecutor()  # e.g., assume defaults configured

            covalent = Covalent(func=my_func, executor=executor)

    Example:
        .. code-block:: python

            # Covalent workflow example.

            from langchain.llms import Covalent
            import covalent as ct

            # e.g., assume all defaults configured
            gcp_executor = ct.executor.GCPExecutor()
            ssh_executor = ct.executor.SSHExecutor()
            slurm_executor = ct.executor.SlurmExecutor()

            @ct.electron(executor=gcp_executor)
            def task(*inputs, flag=True):
                ...

            @ct.electron(executor=ssh_executor)
            def other_task(filename, save=True, folder="my-folder"):
                ...

            @ct.electron(executor=slurm_executor)
            def onprem_task(**options):
                ...

            @ct.lattice
            def covalent_workflow(prompt):
                # uses tasks defined above
                ...

            covalent = Covalent(func=covalent_workflow)
    """

    def __init__(self, **kwargs):
        _check_covalent_installed(cloud=False)
        super().__init__(**kwargs)

    @root_validator(pre=True)
    def validate_executor(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Make sure the ``executor`` attribute is a Covalent ``BaseExecutor`` instance."""

        executor = values.get("executor")

        # Skip check if no executor is specified.
        if executor is not None:
            if not isinstance(executor, _CT_EXECUTOR_TYPE):
                if not isinstance(executor, str):
                    _type = type(executor).__name__
                    raise ValueError(
                        f"Invalid executor ({_type}) for Covalent."
                    )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "covalent"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

        # Start a local Covalent Server.
        self._start_covalent()

        # Convert ``self.func`` to a Covalent lattice.
        lattice = self._to_lattice()

        # Dispatch the lattice.
        dispatch_id = ct.dispatch(
            lattice, **self.dispatch_kwargs)(prompt, **kwargs)

        # Request the result...
        result_obj = ct.get_result(dispatch_id, wait=True)
        if result_obj.status != Result.COMPLETED:
            raise RuntimeError(
                f"Dispatch did not succeed. Returned result object:\n\n{result_obj}"
            )

        return result_obj.result
