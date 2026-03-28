"""GeneratePoolsUseCase: delegate pool generation to the forge-connector JAR."""

from __future__ import annotations

from pathlib import Path


class GeneratePoolsUseCase:
    """Creates the output directory and invokes the connector to generate sealed pools."""

    def execute(
        self,
        set_code: str,
        pool_count: int,
        pools_path: Path,
        connector,
    ) -> None:
        """Generate sealed pools and write them to pools_path/pools.txt.

        Args:
            set_code: MTG set code (e.g. "RVR").
            pool_count: Number of sealed pools to generate.
            pools_path: Directory where pools.txt will be written.
            connector: PoolConnector (or compatible) instance.

        Raises:
            FileNotFoundError: If the JAR or Java is not found.
            RuntimeError: If the connector subprocess fails.
        """
        pools_path.mkdir(parents=True, exist_ok=True)
        connector.generate(set_code, pool_count, pools_path)
