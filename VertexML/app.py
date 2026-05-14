"""VertexML application entry point."""

from core.pipelines.supervised_pipeline import SupervisedPipeline


def main():
    pipeline = SupervisedPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
