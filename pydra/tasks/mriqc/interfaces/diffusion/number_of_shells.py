from fileformats.generic import File
import logging
import numpy as np
import pydra.mark
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {
        "return": {
            "models": list,
            "n_shells": int,
            "out_data": list,
            "b_values": list,
            "b_masks": list,
            "b_indices": list,
            "b_dict": dict,
        }
    }
)
def NumberOfShells(
    in_bvals: File, b0_threshold: float, dsi_threshold: int
) -> ty.Tuple[list, int, list, list, list, list, dict]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.number_of_shells import NumberOfShells

    """
    in_data = np.squeeze(np.loadtxt(in_bvals))
    highb_mask = in_data > b0_threshold

    original_bvals = sorted(set(np.rint(in_data[highb_mask]).astype(int)))
    round_bvals = np.round(in_data, -2).astype(int)
    shell_bvals = sorted(set(round_bvals[highb_mask]))

    if len(shell_bvals) <= dsi_threshold:
        n_shells = len(shell_bvals)
        models = [n_shells]
        out_data = round_bvals.tolist()
        b_values = shell_bvals
    else:
        # For datasets identified as DSI, fit a k-means
        grid_search = GridSearchCV(
            KMeans(), param_grid={"n_clusters": range(1, 10)}, scoring=_rms
        ).fit(in_data[highb_mask].reshape(-1, 1))

        results = np.array(
            sorted(
                zip(
                    grid_search.cv_results_["mean_test_score"] * -1.0,
                    grid_search.cv_results_["param_n_clusters"],
                )
            )
        )

        models = results[:, 1].astype(int).tolist()
        n_shells = int(grid_search.best_params_["n_clusters"])

        out_data = np.zeros_like(in_data)
        predicted_shell = np.rint(
            np.squeeze(
                grid_search.best_estimator_.cluster_centers_[
                    grid_search.best_estimator_.predict(
                        in_data[highb_mask].reshape(-1, 1)
                    )
                ],
            )
        ).astype(int)

        # If estimated shells matches direct count, probably right -- do not change b-vals
        if len(original_bvals) == n_shells:
            # Find closest b-values
            indices = np.abs(predicted_shell[:, np.newaxis] - original_bvals).argmin(
                axis=1
            )
            predicted_shell = original_bvals[indices]

        out_data[highb_mask] = predicted_shell
        out_data = np.round(out_data.astype(float), 2).tolist()
        b_values = sorted(
            np.unique(np.round(predicted_shell.astype(float), 2)).tolist()
        )

    b_masks = [(~highb_mask).tolist()] + [
        np.isclose(out_data, bvalue).tolist() for bvalue in b_values
    ]
    b_indices = [
        np.atleast_1d(np.squeeze(np.argwhere(b_mask)).astype(int)).tolist()
        for b_mask in b_masks
    ]

    b_dict = {int(round(k, 0)): value for k, value in zip([0] + b_values, b_indices)}

    return models, n_shells, out_data, b_values, b_masks, b_indices, b_dict


# Nipype methods converted into functions


def _rms(estimator, X):
    """
    Callable to pass to GridSearchCV that will calculate a distance score.

    To consider: using `MDL
    <https://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/>`__

    """
    if len(np.unique(estimator.cluster_centers_)) < estimator.n_clusters:
        return -np.inf

    # Calculate distance from assigned shell centroid
    distance = X - estimator.cluster_centers_[estimator.predict(X)]
    # Make negative so CV optimizes minimizes the error
    return -np.sqrt(distance**2).sum()
