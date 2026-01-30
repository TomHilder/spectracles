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
from jax import lax
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
            raise TypeError(
                f"OptimiserFrame requires a ShareModule, but got {type(model).__name__}.\n\n"
                f"To fix this, wrap your model with build_model():\n\n"
                f"    # Instead of:\n"
                f"    model = {type(model).__name__}(...)\n\n"
                f"    # Use:\n"
                f"    model = build_model({type(model).__name__}, ...)\n\n"
                f"Or wrap an existing model:\n"
                f"    model = ShareModule(model)"
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


# class OptimiserFrame:
#     """Drop-in replacement version with scan acceleration.
#     Prints a banner when created so you know it's active.
#     """

#     def __init__(
#         self,
#         model,
#         loss_fn,
#         optimiser,
#         get_filter_spec_fn=get_opt_filter_spec,
#         Δloss_criterion=None,
#     ):
#         print(">>> OptimiserFrame[SCAN VERSION] ACTIVE <<<")

#         # --- original sanity checks ---------------------------------------
#         if not isinstance(model, ShareModule):
#             raise ValueError("Model is not a ShareModule")
#         if model._locked:
#             raise ValueError("Cannot optimise a locked model.")

#         # store
#         self.model = model
#         self.loss_fn = loss_fn
#         self.optimiser = optimiser
#         self.get_filter_spec = get_filter_spec_fn
#         self.Δloss_criterion = Δloss_criterion  # unused, but kept for API safety

#         # init opt state
#         self._set_opt_state(self.model)
#         self.loss_history = []

#         # ===============================================================
#         #   1. PURE STEP (not jitted)
#         # ===============================================================
#         def _step(model, opt_state, filter_spec, *loss_args, **loss_kwargs):
#             @filter_value_and_grad
#             def get_loss(vary_model, fixed_model, loss_fn, *inner_args, **inner_kwargs):
#                 combined = combine(vary_model, fixed_model)
#                 return loss_fn(combined, *inner_args, **inner_kwargs)

#             vary_model, fixed_model = partition(model, filter_spec)
#             loss, grad = get_loss(vary_model, fixed_model, loss_fn, *loss_args, **loss_kwargs)

#             params = filter(vary_model, is_array)
#             updates, new_state = optimiser.update(grad, opt_state, params)

#             new_model = apply_updates(model, updates)
#             return loss, new_model, new_state

#         self._step = _step

#         # ===============================================================
#         #   2. JITTED STEP (for compatibility)
#         # ===============================================================
#         @filter_jit
#         def make_step(model, opt_state, filter_spec, *loss_args, **loss_kwargs):
#             return _step(model, opt_state, filter_spec, *loss_args, **loss_kwargs)

#         self.make_step = make_step

#         # ===============================================================
#         #   3. JITTED SCAN LOOP (THIS IS THE NEW FAST PATH)
#         # ===============================================================
#         @filter_jit
#         def run_many(model, opt_state, filter_spec, n_steps, *loss_args, **loss_kwargs):
#             # Split once outside scan
#             vary0, fixed = partition(model, filter_spec)

#             def body_fun(carry, _):
#                 vary, opt_s = carry

#                 # combine for loss
#                 model_c = combine(vary, fixed)

#                 # do one optimisation step
#                 loss, new_model, new_opt = _step(
#                     model_c, opt_s, filter_spec, *loss_args, **loss_kwargs
#                 )

#                 # re-partition: extract new varying part only
#                 new_vary, _ = partition(new_model, filter_spec)

#                 return (new_vary, new_opt), loss

#             # run scan over steps
#             (final_vary, final_opt), losses = lax.scan(
#                 body_fun, (vary0, opt_state), xs=None, length=n_steps
#             )

#             # rebuild full model at end
#             final_model = combine(final_vary, fixed)

#             return final_model, final_opt, losses

#         self._run_many = run_many

#     # ===============================================================
#     #   API-COMPATIBLE RUN FUNCTION (old behaviour but fast)
#     # ===============================================================
#     def run(self, n_steps, *loss_args, **loss_kwargs):
#         """Same API as original run(), but using scan for speed."""
#         self._verify_loss_fn(*loss_args, **loss_kwargs)

#         filter_spec = self.get_filter_spec(self.model)
#         model_ = self.model
#         state_ = self.opt_state

#         # call fast scan
#         final_model, final_state, losses = self._run_many(
#             model_, state_, filter_spec, n_steps, *loss_args, **loss_kwargs
#         )

#         # update fields
#         self.model = final_model
#         self.opt_state = final_state
#         self.loss_history += list(losses)

#         return self.model

#     # ===============================================================
#     #   Unchanged support methods (copied exactly)
#     # ===============================================================
#     def _set_opt_state(self, model):
#         self.model = model
#         vary_model, _ = partition(self.model, self.get_filter_spec(model))
#         self.opt_state = self.optimiser.init(filter(vary_model, is_array))

#     def _verify_loss_fn(self, *loss_args, **loss_kwargs):
#         if not callable(self.loss_fn):
#             raise ValueError("Loss function is not callable.")
#         try:
#             loss_out = self.loss_fn(self.model, *loss_args, **loss_kwargs)
#         except Exception as e:
#             raise ValueError("Loss fn threw exception.") from e
#         if jnp.any(jnp.isnan(loss_out)):
#             raise ValueError("Loss fn produced NaN.")

#     @staticmethod
#     def _check_convergence(loss_history, Δloss, pbar=None):
#         # kept only for API completeness
#         trend = loss_history[-50] - loss_history[-1]
#         if pbar is not None:
#             pbar.set_description(f"Optimising (Δloss trend: {trend:.2e})")
#         return jnp.abs(trend) < Δloss
