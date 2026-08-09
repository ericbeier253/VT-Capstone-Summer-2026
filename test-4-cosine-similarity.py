from vision.embedder import DinoEmbedder

import numpy as np


embedder = DinoEmbedder()

e1 = np.array(

    embedder.embed("test.jpg")

)

e2 = np.array(

    embedder.embed("test_copy.jpg")

)

similarity = np.dot(e1, e2)

print(similarity)