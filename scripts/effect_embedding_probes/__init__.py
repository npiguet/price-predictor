"""The effect-embedding probe battery: what one checkpoint's ``e`` holds.

A package so its modules import one another by name. Each script is still run
by path, and puts ``scripts/`` on ``sys.path`` before importing the battery.

Nothing here does work at import time.
"""
