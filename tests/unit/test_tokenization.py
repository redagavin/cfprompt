import pytest

from cfprompt.tokenization import token_edit_distance, token_edit_distance_pct


@pytest.mark.unit
class TestTokenEditDistance:
    def test_identical_lists_zero(self):
        assert token_edit_distance([1, 2, 3], [1, 2, 3]) == 0

    def test_pure_insertion(self):
        assert token_edit_distance([1, 2, 3], [1, 2, 3, 4]) == 1

    def test_pure_deletion(self):
        assert token_edit_distance([1, 2, 3, 4], [1, 2, 3]) == 1

    def test_pure_substitution(self):
        assert token_edit_distance([1, 2, 3], [1, 9, 3]) == 1

    def test_mixed_edits(self):
        # [1,2,3,4] -> [1,9,3,5] : substitute 2->9, substitute 4->5 = 2 edits
        assert token_edit_distance([1, 2, 3, 4], [1, 9, 3, 5]) == 2

    def test_empty_to_nonempty(self):
        assert token_edit_distance([], [1, 2, 3]) == 3

    def test_nonempty_to_empty(self):
        assert token_edit_distance([1, 2, 3], []) == 3

    def test_both_empty(self):
        assert token_edit_distance([], []) == 0


@pytest.mark.unit
class TestTokenEditDistancePct:
    def test_identical_zero_pct(self):
        assert token_edit_distance_pct([1, 2, 3], [1, 2, 3]) == 0.0

    def test_one_in_three_is_about_33pct(self):
        # 1 edit on a 3-token original = 33.333...%
        assert token_edit_distance_pct([1, 2, 3], [1, 9, 3]) == pytest.approx(100.0 / 3)

    def test_uses_original_length_as_denominator(self):
        # original = 4 tokens, modified = 5 tokens, 1 insertion, percent = 25%
        assert token_edit_distance_pct([1, 2, 3, 4], [1, 2, 3, 4, 5]) == pytest.approx(25.0)

    def test_zero_length_original_raises(self):
        with pytest.raises(ValueError, match=r"empty"):
            token_edit_distance_pct([], [1, 2])
