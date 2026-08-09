import time

from vision.embedder import DinoEmbedder


embedder = DinoEmbedder()

start = time.time()

embedder.embed("test.jpg")

print(

    f"{time.time()-start:.3f}s"

)