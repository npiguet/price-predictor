# forge-connector

Java 17+ command-line workers that the Python packages run against the sibling
MTG Forge checkout (`../forge`): card-script conversion, sealed pool
generation, self-play match workers, draft workers and scorer evaluation. The
repository-root `CLAUDE.md` lists each main class and the command that spawns
it, and its "Corpus file formats" section is the contract for everything the
workers write.

## Build

Forge must be installed into the local Maven repository first
(`mvn install -DskipTests` in `../forge`). Then:

```bash
cd forge-connector
mvn install -DskipTests
```

The fat JAR the Python side runs is
`target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`.

## Tests

`mvn install` runs the unit tests. Tests that need an initialised Forge
environment are tagged `integration` and run with `mvn test -Pintegration`.
