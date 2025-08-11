"""opt_frame.py - Frame for optimising a model using an optimiser and a loss function."""
# TODO: typing!

from typing import Callable

import jax.numpy as jnp
from equinox import (
    apply_updates,
    combine,
    filter,
    filter_jit,
    filter_value_and_grad,
    is_array,
    partition,
)
from jax.tree_util import tree_map
from optax import GradientTransformation  # type: ignore[import]
from tqdm import tqdm

from spectracles.model.parameter import is_parameter, is_trainable
from spectracles.model.share_module import ShareModule


def get_opt_filter_spec(model: ShareModule) -> Callable:
    return tree_map(is_trainable, model, is_leaf=is_parameter)  # type: ignore[no-any-return]


class OptimiserFrame:
    def __init__(
        self,
        model: ShareModule,
        loss_fn: Callable[..., float],
        optimiser: GradientTransformation,
        get_filter_spec_fn: Callable[[ShareModule], Callable] = get_opt_filter_spec,
        Δloss_criterion: float = None,
    ):
        # Check sensible input first
        if not isinstance(model, ShareModule):
            raise ValueError(
                "Model is not of required type ShareModule. Likely you forgot to build the model with the build_model function."
            )
        elif model._locked:
            raise ValueError("Cannot optimise a locked model.")

        # Initialise the optimisation state and save info
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.get_filter_spec = get_filter_spec_fn
        self.Δloss_criterion = Δloss_criterion

        # Initialise the optimisation state
        self._set_opt_state(self.model)

        # Initialise the optimisation history
        self.loss_history: list = []

        # Get the stepping function
        @filter_jit
        def make_step(
            model,
            optimiser,
            opt_state,
            filter_spec,
            loss_fn,
            *loss_args,
            **loss_kwargs,
        ):
            # Loss function
            @filter_value_and_grad
            def get_loss(vary_model, fixed_model, loss_fn, *inner_args, **inner_kwargs):
                model = combine(vary_model, fixed_model)
                return loss_fn(model, *inner_args, **inner_kwargs)

            # Split varying and constant parts of model
            vary_model, fixed_model = partition(model, filter_spec)
            # Calculate the loss and gradients
            loss, grad = get_loss(
                vary_model, fixed_model, loss_fn, *loss_args, **loss_kwargs
            )
            # Optimiser updates step
            updates, opt_state = optimiser.update(
                grad,
                opt_state,
                filter(vary_model, is_array),
                value=loss,
                grad=grad,
                value_fn=lambda _: get_loss(
                    vary_model,
                    fixed_model,
                    loss_fn,
                    *loss_args,
                    **loss_kwargs,
                )[0],
            )
            # Update the model
            model = apply_updates(model, updates)
            # Check convergence
            return loss, model, opt_state

        # Save the make step function we made
        self.make_step = make_step

    def run(self, n_steps, *loss_args, **loss_kwargs):
        # Verify loss function
        self._verify_loss_fn(*loss_args, **loss_kwargs)
        # Get the filter spec for the model
        filter_spec = self.get_filter_spec(self.model)
        # Grab current opt state and model
        opt_state_ = self.opt_state
        model_ = self.model
        # Perform optimisation by calling stepping function
        loss = []
        # Loop over number of steps
        pbar = tqdm(iterable=range(n_steps), desc="Optimising", unit="step")
        for i in pbar:
            loss_, model_, opt_state_ = self.make_step(
                model_,
                self.optimiser,
                opt_state_,
                filter_spec,
                self.loss_fn,
                *loss_args,
                **loss_kwargs,
            )
            loss.append(loss_)

            # Check for convergence
            if self.Δloss_criterion is not None:
                if i >= 100 and i % 50 == 0:
                    if self._check_convergence(
                        loss_history=loss,
                        Δloss=self.Δloss_criterion,
                        pbar=pbar,
                    ):
                        print(
                            f"Early exit based on Δloss_criterion of {self.Δloss_criterion:.2e} at step {i}."
                        )
                        break
        # Save results
        self.opt_state = opt_state_
        self.model = model_
        self.loss_history += loss
        # Return the model I guess?
        return self.model

    def _set_opt_state(self, model: ShareModule):
        self.model = model
        vary_model, _ = partition(self.model, self.get_filter_spec(model))
        self.opt_state = self.optimiser.init(filter(vary_model, is_array))

    def _verify_loss_fn(self, *loss_args, **loss_kwargs):
        # Check the loss function is callable
        if not callable(self.loss_fn):
            raise ValueError("Loss function is not callable.")
        # Check the loss function doesn't output nan or raise Exceptions
        try:
            loss_output = self.loss_fn(self.model, *loss_args, **loss_kwargs)
        except Exception as e:
            raise ValueError(
                "Evaluating provided loss function causes an Exception."
            ) from e
        if jnp.any(jnp.isnan(loss_output)):
            raise ValueError("Loss function outputs NaN.")

    @staticmethod
    def _check_convergence(
        loss_history: list[float],
        Δloss: float,
        pbar: tqdm = None,
    ) -> bool:
        trend = loss_history[-50] - loss_history[-1]
        if pbar is not None:
            pbar.set_description(f"Optimising (Δloss trend: {trend:.2e})")
        return jnp.abs(trend) < Δloss
