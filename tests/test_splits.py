"""(c) Region split: no tile in both train and test. (d) Block split: no block in both train and test.

Each split is rebuilt exactly as the scripts build it, and, where present, checked against the fold
checkpoints the actual runs saved (results/step7_folds/, results/step8_folds/)."""
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from conftest import RES

SEEDS = range(5)


def region_folds(reg):
    # as in scripts/step4_compare.py and scripts/step7_combined.py (folds_for, split == "region")
    return [(reg != r, reg == r) for r in sorted(np.unique(reg))]


def block_folds(n, y, groups, seed):
    # as in scripts/step8_block_split.py (folds_for)
    return list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), y, groups))


# ------------------------------------------------------------------------------------- (c) region split
def test_regions_are_the_seven_merged_regions(metadata):
    assert sorted(metadata["region"].unique()) == [0, 1, 2, 3, 4, 5, 7]


def test_region_split_no_tile_in_train_and_test(metadata):
    reg = metadata["region"].to_numpy()
    paths = metadata["path"].to_numpy()
    assert metadata["path"].is_unique
    tested = np.zeros(len(metadata), int)
    for train, test in region_folds(reg):
        assert test.any() and train.any()
        assert not (train & test).any()
        assert not set(paths[train]) & set(paths[test])
        assert len(np.unique(reg[test])) == 1 and reg[test][0] not in set(reg[train])
        tested += test
    assert (tested == 1).all()  # every tile is tested exactly once


def test_region_split_saved_checkpoints(metadata):
    files = sorted((RES / "step7_folds").glob("region_seed*_fold*.npz"))
    if not files:
        pytest.skip("results/step7_folds/ not present")
    reg = metadata["region"].to_numpy()
    folds = region_folds(reg)
    for f in files:
        k = int(f.stem.split("fold")[1])
        test = np.load(f)["test"]
        assert np.array_equal(test, np.flatnonzero(folds[k][1])), f.name
        assert not np.isin(test, np.flatnonzero(folds[k][0])).any(), f.name


# -------------------------------------------------------------------------------------- (d) block split
def test_blocks_align_with_metadata(metadata, blocks):
    assert (blocks["path"] == metadata["path"]).all()


def test_metadata_rows_match_features(metadata, features):
    # the saved fold indices refer to features.csv rows; the split tests use metadata.csv
    assert (metadata["path"] == features["path"]).all() and (metadata["region"] == features["region"]).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_block_split_no_block_in_train_and_test(blocks, seed):
    groups, y = blocks["block"].to_numpy(), blocks["class"].to_numpy()
    tested = np.zeros(len(blocks), int)
    for tr, te in block_folds(len(blocks), y, groups, seed):
        assert not set(groups[tr]) & set(groups[te])
        tested[te] += 1
    assert (tested == 1).all()


def test_block_split_saved_checkpoints(blocks):
    files = sorted((RES / "step8_folds").glob("*_seed*_fold*.npz"))
    if not files:
        pytest.skip("results/step8_folds/ not present")
    groups = blocks["block"].to_numpy()
    all_idx = np.arange(len(blocks))
    for f in files:
        test = np.load(f)["test"]
        train = np.setdiff1d(all_idx, test)
        assert not set(groups[train]) & set(groups[test]), f.name
