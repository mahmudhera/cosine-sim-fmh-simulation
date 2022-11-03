import src.driver as proc
import numpy as np

def test_get_cosine_similarity():
    a = np.array([0,1,0])
    b = np.array([0,0,1])
    ans = 0
    assert ans == proc.get_cosine_similarity(a,b)
