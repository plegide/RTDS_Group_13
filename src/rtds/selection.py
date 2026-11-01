"""The tool's rows: the background and the explained patients.

Both depend only on the dataset index, so they are fixed before fitting and the same at every rho.
"""

from rtds import seeds


def background_rows(cfg, dataset_index):
    """Indices into the training set."""
    gen = seeds.rng(cfg["master_seed"], "background", dataset_index)
    return gen.choice(cfg["simulation"]["n_train"], cfg["tool"]["n_background"], replace=False)


def explained_patients(cfg, dataset_index):
    """Indices into the held-out set."""
    gen = seeds.rng(cfg["master_seed"], "explained", dataset_index)
    return gen.choice(cfg["simulation"]["n_test"], cfg["tool"]["n_explained"], replace=False)


def tool_inputs(cfg, dataset, dataset_index, builder):
    """Background rows (with their outcomes) and the explained patients, as model input columns."""
    bg = background_rows(cfg, dataset_index)
    explained = explained_patients(cfg, dataset_index)
    background = builder.build(dataset.train.visits[bg])
    patients = builder.build(dataset.test.visits[explained])
    return background, dataset.train.y[bg], patients
