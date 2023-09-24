from pathlib import Path

import tensorflow as tf
from tfx.orchestration.experimental.interactive.interactive_context import (
    InteractiveContext,
)
from tfx.v1.components import CsvExampleGen, StatisticsGen

DATA_PATH = "https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv"


def main():
    context = InteractiveContext()

    dir_path = Path().parent.absolute()
    data_dir = dir_path.parent.parent / "data" / "taxi"
    data_dir.mkdir(parents=True, exist_ok=True)

    _ = tf.keras.utils.get_file(
        data_dir / "data.csv",
        DATA_PATH,
    )

    example_gen = CsvExampleGen(input_base=data_dir)
    context.run(example_gen)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )
    context.run(statistics_gen)

    context.show(statistics_gen.outputs["statistics"])

    # statistics_gen = StatisticsGen(
    #     examples=example_gen.outputs["examples"],
    # )
    # context.run(statistics_gen)

    for artifact in statistics_gen.outputs["statistics"].get():
        print(artifact.uri)


if __name__ == "__main__":
    main()
