from vision.embedder import DinoEmbedder


def main():

    embedder = DinoEmbedder()

    embedding = embedder.embed(

        "cropped_objects/test/object_000_00.jpg"

    )

    print()

    print("Embedding length:", len(embedding))

    print("Norm:",
          sum(x*x for x in embedding) ** 0.5)

    print()

    print("Embedder test passed.")


if __name__ == "__main__":

    main()