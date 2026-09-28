"""Fast unit tests; run with `python selective_mpd/test_mpd_core.py`."""

from mpd_core import edit_distance, fit_logistic, make_features, oracle_label, predict_probability


def main() -> None:
    assert edit_distance("I has book", "I have a book") == 2
    assert edit_distance("same", "same") == 0
    assert oracle_label("I has book", "I have a book", "I have a book") == 1
    assert oracle_label("correct", "also wrong", "correct") == 0
    stable = make_features("A sentence .", "A sentence .", "A sentence .", -0.2, -0.1)
    changed = make_features("He go .", "He goes .", "He goes home .", -0.5, -0.2)
    model = fit_logistic([stable, changed, stable, changed], [0, 1, 0, 1], steps=500)
    assert predict_probability(model, changed) > predict_probability(model, stable)
    print("All selective_mpd core tests passed")


if __name__ == "__main__":
    main()
