"""
prior_provider.py — Callable registered under the redback.model.priors entry point.

redback calls get_prior(model_name) when building priors for any model registered
by this plugin. Returns a bilby PriorDict or None if the model is not known here.
"""

import inspect
import os
import re

from bilby.core.prior import PriorDict, Uniform

PRIOR_DIR = os.path.join(os.path.dirname(__file__), "priors")


def _uniform(name, minimum, maximum, label):
    return Uniform(minimum=minimum, maximum=maximum, name=name, latex_label=label)


def _latex_label(name):
    labels = {
        "redshift": "$z$",
        "logeej": "$\\log(E_{\\mathrm{ej}})~(\\mathrm{erg s}^{-1})$",
        "logmej": "$\\log(M_{\\mathrm{ej}})~(M_\\odot)$",  
        "logr0": "$\\log(r_{\\mathrm{0}}~({\\mathrm{cm}})$",       
        "logA": "$\\log(A}~({\\mathrm{g cm}^{-3}})$",
        "n_csm": "$n_{\\mathrm{CSM}}$
    }
    if name in labels:
        return labels[name]
    return f"${name}$"


def _prior_for_parameter(name):
    label = _latex_label(name)
    if name == "redshift":
        return _uniform(name, 1e-3, 3.0, label)
    if name == "logeej"}:
        return _uniform(name, 51.0, 56.0, label)
    if name == "logmej":
        return _uniform(name, 2.0, 6.0, label)
    if name == "logr0":
        return _uniform(name, 13.0, 16.0, label)
    if name == "logA":
        return _uniform(name, -15.0, 0.0, label)
    if name == "n_csm":
        return _uniform(name, 0.0, 3.0, label)       
    raise KeyError(f"No generated prior template for parameter '{name}'")


def _model_parameters(model_name):
    import redback_csm.models as models

    if not hasattr(models, model_name):
        return None
    sig = inspect.signature(getattr(models, model_name))
    names = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name in {"time"}:
            continue
        names.append(param.name)
    return names


def _generated_prior(model_name):
    names = _model_parameters(model_name)
    if names is None:
        return None

    priors = PriorDict()
    for name in names:
        priors[name] = _prior_for_parameter(name)

    return priors


def get_prior(model_name: str):
    """
    Return a bilby PriorDict for the given model name, or None if not found.

    Used by redback.priors.get_priors() via the redback.model.priors entry point.
    """
    generated = _generated_prior(model_name)
    if generated is not None:
        return generated

    path = os.path.join(PRIOR_DIR, f"{model_name}.prior")
    if not os.path.exists(path):
        return None
    priors = PriorDict()
    priors.from_file(path)
    return priors