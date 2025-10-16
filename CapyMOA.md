# CapyMOA

CapyMOA is a datastream learning framework that integrates the [Massive Online Analysis
(MOA)](https://moa.cms.waikato.ac.nz/) library with the python ecosystem.

To build MOA for use with CapyMOA run:
```bash
cd moa
mvn package -DskipTests -Dmaven.javadoc.skip=true -Dlatex.skipBuild=true
```
This will create a `target/moa-*-jar-with-dependencies.jar` file that can be used by
CapyMOA. To let CapyMOA know where this file is, set the `CAPYMOA_MOA_JAR` environment
variable to the path of this file.

You can do this temporarily in your terminal session with:
```bash
export CAPYMOA_MOA_JAR=/path/to/moa/target/moa-*-jar-with-dependencies.jar
```
To check that CapyMOA can find MOA, run:
```bash
python -c "import capymoa; capymoa.about()"
# CapyMOA 0.10.0
#   CAPYMOA_DATASETS_DIR: .../datasets
#   CAPYMOA_MOA_JAR:      .../moa/moa/target/moa-2024.07.2-SNAPSHOT-jar-with-dependencies.jar
#   CAPYMOA_JVM_ARGS:     ['-Xmx8g', '-Xss10M']
#   JAVA_HOME:            /usr/lib/jvm/java-21-openjdk
#   MOA version:          aa955ebbcbd99e9e1d19ab16582e3e5a6fca5801ba250e4d164c16a89cf798ea
#   JAVA version:         21.0.7
```
