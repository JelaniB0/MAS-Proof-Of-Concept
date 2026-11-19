import numpy as np
from linucb import LinUCB

def test_initialization():
    bandit = LinUCB(n_arms=3, d=4, alpha=0.5)
    assert len(bandit.A) == 3
    assert len(bandit.b) == 3
    assert np.allclose(bandit.A[0], np.eye(4))
    assert np.allclose(bandit.b[0], np.zeros(4))


def test_initial_arm_selection():
    bandit = LinUCB(n_arms=3, d=4, alpha=0.5)
    context = np.array([1.0, 0.3, 0.2, 0.1])
    selected_arm = bandit.select_arm(context)
    assert selected_arm in [0, 1, 2]


def test_update():
    bandit = LinUCB(n_arms=3, d=4, alpha=0.5)
    context = np.array([1.0, 0.3, 0.2, 0.1])
    bandit.update(1, context, 0.8)
    assert np.allclose(
        bandit.A[1],
        np.eye(4) + np.outer(context, context)
    )
    assert np.allclose(
        bandit.b[1],
        0.8 * context
    )
    assert np.allclose(bandit.A[0], np.eye(4))
    assert np.allclose(bandit.b[2], np.zeros(4))

if __name__ == "__main__":
    test_initialization()
    test_initial_arm_selection()
    test_update()

    print("All LinUCB tests passed!")
