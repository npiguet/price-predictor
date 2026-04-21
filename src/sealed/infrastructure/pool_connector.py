"""PoolConnector: subprocess call to the forge-connector JAR for pool generation."""

from __future__ import annotations

from pathlib import Path

from price_predictor.infrastructure.forge_jvm import run_forge_worker


class PoolConnector:
    """Invokes the forge-connector JAR to generate sealed pools."""

    def generate(
        self,
        set_code: str | None,
        pool_count: int,
        pools_path: Path,
    ) -> int:
        """Generate sealed pools by invoking PoolMain via the connector JAR.

        Args:
            set_code: MTG set code (e.g. ``RVR``). When ``None``, the Java
                worker selects a random sealed-legal set per pool (the
                ``--set`` argument is omitted from the subprocess command).
            pool_count: Number of sealed pools to generate.
            pools_path: Directory where pools.txt will be written.

        Returns:
            Process exit code (0 = success).

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
            RuntimeError: If the subprocess exits with a non-zero code.
        """
        main_args: list[str] = []
        if set_code is not None:
            main_args.extend(["--set", set_code])
        main_args.extend([
            "--size", str(pool_count),
            "--pools-path", str(pools_path),
        ])

        result = run_forge_worker(
            "com.pricepredictor.connector.PoolMain",
            main_args=main_args,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"PoolMain exited with code {result.returncode} for set={set_code}"
            )

        return result.returncode
