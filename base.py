#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This package contains abstract base classes for different solvers.

Solvers are used to package models, criterions and optimizers and implement training
loops. When subclassing abstract solvers, in the simplest case only the
:py:meth:`Solver._inference` needs to be overridden.

For more complex use cases, the :py:meth:`Solver.step` and
:py:meth:`Solver.fit` method can be overridden to
implement larger changes to the training loop.
"""

import abc
import copy
import os
from typing import Callable, Dict, List, Literal, Optional

import literate_dataclasses as dataclasses
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
import torch

import cebra
import cebra.data
import cebra.io
import cebra.models
from cebra.solver.util import Meter
from cebra.solver.util import ProgressBar


class JacobianReg(torch.nn.Module):
    """Jacobian regularizer (Hoffman, Roberts & Yaida, 2019, arXiv:1908.02729).

    Computes (1/2) * trace(|dy/dx|^2) using random projections, exactly as in
    ``cebra.models.jacobian_regularizer.JacobianReg`` (used in the xCEBRA paper)
    and in the original implementation at
    https://github.com/facebookresearch/jacobian_regularizer.

    Adding this term to the training loss encourages the model to learn an
    encoder whose Jacobian (and therefore its attribution map, as computed by
    ``cebra.attribution``) is more sparse / identifiable.
    """

    def __init__(self, n: int = 1):
        super().__init__()
        assert n == -1 or n > 0
        self.n = n

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C = y.shape
        num_proj = C if self.n == -1 else self.n

        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                v = torch.zeros(B, C, device=y.device)
                v[:, ii] = 1
            else:
                v = self._random_vector(C=C, B=B).to(y.device)

            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 = J2 + C * torch.norm(Jv)**2 / (num_proj * B)

        return 0.5 * J2

    def _random_vector(self, C: int, B: int) -> torch.Tensor:
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        return v / vnorm

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
        return grad_x


def _l2_normalize(t: torch.Tensor, eps: float = 1e-12):
    """Per-sample L2 normalisation (zero vectors stay zero)."""
    flat = t.reshape(t.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return t / norm.view(-1, *([1] * (t.dim() - 1)))


def _rand_radius_like(t: torch.Tensor):
    """U(0,1) radius shaped like t but broadcastable (B,1,1,…) ."""
    return torch.rand([t.size(0)] + [1] * (t.dim() - 1), device=t.device)


def _proj_l2_ball(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
    """Project adv back to the closed L2 ball of radius ε around orig."""
    delta = adv - orig
    flat = delta.reshape(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = torch.where(norm > epsilon, norm / epsilon, torch.ones_like(norm))
    delta = delta / factor.view(-1, *([1] * (delta.dim() - 1)))
    return orig + delta


#

# positive_perm= shuffle_and_permute(batch.positive)
# anchor_perm= shuffle_and_permute(batch.reference)
# negative_perm= shuffle_and_permute(batch.negative)


# copied_data = copy.deepcopy(batch)#batch.clone().detach()
# copied_data.positive = torch.randn_like(batch.positive).cuda() #positive_perm
# copied_data.reference = torch.randn_like(batch.reference).cuda() #anchor_perm
# copied_data.negative = torch.randn_like(batch.negative).cuda() #negative_perm


# prediction_fix = self._inference(batch)

# prediction_perm= self._inference(copied_data)

# x_perm_ = torch.cat([prediction_perm.reference,prediction_perm.positive, prediction_perm.negative], dim=0)
# x_fix= torch.cat([prediction_fix.reference,prediction_fix.positive, prediction_fix.negative], dim=0)

# self.optimizer.zero_grad()

# cos_raw    = torch.nn.functional.cosine_similarity(x_perm_,x_fix)
# L_cos      = torch.log((2 + cos_raw) / 2).mean()  # log version
# # print(L_cos.item())
# L_cos.backward()
# self.optimizer.step()


# fake_data=torch.randn_like(batch.positive).cuda()


# self.optimizer.zero_grad()

# positive_perm= shuffle_and_permute(batch.positive)
# anchor_perm= shuffle_and_permute(batch.reference)
# fake_data=torch.randn_like(batch.positive).cuda()


# x_perm_ = torch.cat([anchor_perm,positive_perm, batch.negative], dim=0)

# idx = torch.randperm(x_perm_.size(0))[:positive_perm.size(0)]


# permed_data_negative   = x_perm_[idx]

# fake_batch = cebra.data.Batch(
#                 reference=batch.reference,
#                 positive=batch.positive,
#                 negative=batch.negative
#             )
# fake_prediction = self._inference(fake_batch)
# loss, align, uniform= self.criterion(fake_prediction.reference,
#                                       fake_prediction.positive,
#                                       fake_prediction.negative)
# loss.backward()
# self.optimizer.step()


def clone_batch(batch):
    # 1) shallow-copy the Batch container
    new_batch = copy.copy(batch)

    # 2) for each tensor attribute in the batch, clone & detach
    #    adjust this list to match the actual fields in your Batch
    tensor_fields = ['reference', 'positive', 'negative', 'anchor', 'other_tensor']
    for field in tensor_fields:
        tensor = getattr(batch, field, None)
        if isinstance(tensor, torch.Tensor):
            # clone and detach so it shares neither data nor grad
            setattr(new_batch, field, tensor.clone().detach())
        # if the field can be a list/tuple of tensors, you could also do:
        # elif (isinstance(tensor, (list, tuple)) and
        #       all(isinstance(t, torch.Tensor) for t in tensor)):
        #     cloned = [t.clone().detach() for t in tensor]
        #     setattr(new_batch, field, type(tensor)(cloned))

    return new_batch


def shuffle_and_permute(data: torch.Tensor) -> torch.Tensor:
    """
    Shuffle the second (M) and third (D) dimensions of a 3D tensor independently.

    Parameters:
        data (torch.Tensor): Input tensor of shape [N, M, D].

    Returns:
        torch.Tensor: Tensor with shuffled M and D dimensions.
    """
    N, M, D = data.size()

    # Create random permutations for the second and third dimensions
    perm_M = torch.randperm(M)
    # perm_D = torch.randperm(D)

    # Permute the second dimension (axis=1) first, then the third dimension (axis=2)
    shuffled_data = data[:, perm_M.cuda(), :]

    return shuffled_data


@dataclasses.dataclass
class Solver(abc.ABC, cebra.io.HasDevice):
    """Solver base class.

    A solver contains helper methods for bundling a model, criterion and optimizer.

    Attributes:
        model: The encoder for transforming reference, positive and negative samples.
        criterion: The criterion computed from the similarities between positive pairs
            and negative pairs. The criterion can have trainable parameters on its own.
        optimizer: A PyTorch optimizer for updating model and criterion parameters.
        history: Deprecated since 0.0.2. Use :py:attr:`log`.
        decode_history: Deprecated since 0.0.2. Use a hook during training for validation and
            decoding. See the arguments of :py:meth:`fit`.
        log: The logs recorded during training, typically contains the ``total`` loss as well
            as the logs for positive (``pos``) and negative (``neg``) pairs. For the standard
            criterions in CEBRA, also contains the value of the ``temperature``.
        tqdm_on: Use ``tqdm`` for showing a progress bar during training.
        training_mode: The training mode, either "clean" or "adversarial".
        adv_epsilon: The maximum perturbation allowed for adversarial training.
        adv_alpha: The step size for adversarial training.
        adv_steps: The number of steps for adversarial training.
        attack_norm: The norm used for adversarial training, either "l2" or "linf".
    """

    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    history: List = dataclasses.field(default_factory=list)
    decode_history: List = dataclasses.field(default_factory=list)
    log: Dict = dataclasses.field(default_factory=lambda: ({
        "pos": [],
        "neg": [],
        "total": [],
        "temperature": []
    }))
    tqdm_on: bool = True
    training_mode: str = "clean"
    adv_epsilon: float = 0.5
    adv_alpha: float = 0.1
    adv_steps: int = 10
    attack_norm: str = "l2"
    jacobian_weight: float = 0.0  # 0.0 = Jacobian regularizer off (backward-compatible default)
    adv_aggregate: bool = False  # False = only use the final PGD step (old behavior).
                                  # True = expose the model to every intermediate PGD
                                  # step (0..adv_steps) in the same batch, averaged into one loss.

    def __post_init__(self):
        cebra.io.HasDevice.__init__(self)
        self.best_loss = float("inf")
        self._jacobian_reg = JacobianReg(n=1)

    def state_dict(self) -> dict:
        """Return a dictionary fully describing the current solver state.

        Returns:
            State dictionary, including the state dictionary of the models and
            optimizer. Also contains the training history and the CEBRA version
            the model was trained with.
        """

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": torch.tensor(self.history),
            "decode": self.decode_history,
            "criterion": self.criterion.state_dict(),
            "version": cebra.__version__,
            "log": self.log,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Update the solver state with the given state_dict.

        Args:
            state_dict: Dictionary with parameters for the `model`, `optimizer`,
                and the past loss history for the solver.
            strict: Make sure all states can be loaded. Set to `False` to allow
                to partially load the state for all given keys.
        """

        def _contains(key):
            if key in state_dict:
                return True
            elif strict:
                raise KeyError(
                    f"Key {key} missing in state_dict. Contains: {list(state_dict.keys())}."
                )
            return False

        def _get(key):
            return state_dict.get(key)

        if _contains("model"):
            self.model.load_state_dict(_get("model"))
        if _contains("criterion"):
            self.criterion.load_state_dict(_get("criterion"))
        if _contains("optimizer"):
            self.optimizer.load_state_dict(_get("optimizer"))
        # TODO(stes): This will be deprecated at some point; the "log" attribute
        # holds the same information.
        if _contains("loss"):
            self.history = _get("loss").cpu().numpy().tolist()
        if _contains("decode"):
            self.decode_history = _get("decode")
        if _contains("log"):
            self.log = _get("log")

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the encoder and criterion."""
        return sum(p.numel() for p in self.parameters())

    def parameters(self):
        """Iterate over all parameters."""
        for parameter in self.model.parameters():
            yield parameter

        for parameter in self.criterion.parameters():
            yield parameter

    def _get_loader(self, loader):
        return ProgressBar(
            loader,
            "tqdm" if self.tqdm_on else "off",
        )

    def fit(
            self,
            loader: cebra.data.Loader,
            valid_loader: cebra.data.Loader = None,
            *,
            save_frequency: int = None,
            valid_frequency: int = None,
            decode: bool = False,
            logdir: str = None,
            save_hook: Callable[[int, "Solver"], None] = None,
    ):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            save_frequency: If not `None`, the frequency for automatically saving model checkpoints
                to `logdir`.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.

        TODO:
            * Refine the API here. Drop the validation entirely, and implement this via a hook?
        """

        self.to(loader.device)

        iterator = self._get_loader(loader)
        self.model.train()
        for num_steps, batch in iterator:
            stats = self.step(batch)
            iterator.set_description(stats)

            if save_frequency is None:
                continue
            save_model = num_steps % save_frequency == 0
            run_validation = (valid_loader
                              is not None) and (num_steps % valid_frequency
                                                == 0)
            if run_validation:
                validation_loss = self.validation(valid_loader)
                if self.best_loss is None or validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.save(logdir, "checkpoint_best.pth")
            if save_model:
                if decode:
                    self.decode_history.append(
                        self.decoding(loader, valid_loader))
                if save_hook is not None:
                    save_hook(num_steps, self)
                if logdir is not None:
                    self.save(logdir, f"checkpoint_{num_steps:#07d}.pth")

    # ----------------------------------------------------------------------
    # Drop-in replacement for Solver.step
    # ----------------------------------------------------------------------
    def step(self, batch: cebra.data.Batch) -> dict:
        # ------------------------------------------------------------
        # 1) ordinary contrastive update
        # ------------------------------------------------------------
        self.optimizer.zero_grad()

        # Needed for the Jacobian regularizer (no-op if jacobian_weight == 0)
        if self.jacobian_weight > 0:
            batch.reference.requires_grad_(True)

        pred = self._inference(batch)
        loss, align, uniform = self.criterion(
            pred.reference, pred.positive, pred.negative
        )

        # ---- Jacobian regularizer (xCEBRA-style identifiability term) ----
        if self.jacobian_weight > 0:
            jacobian_loss = self._jacobian_reg(batch.reference, pred.reference)
            loss = loss + self.jacobian_weight * jacobian_loss

        loss.backward()
        self.optimizer.step()

        self.history.append(loss.item())
        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        for k, v in stats.items():
            self.log[k].append(v)
        
        if self.training_mode == "adversarial":
            if self.attack_norm == "linf":
                self.optimizer.zero_grad()

                adv_epsilon = self.adv_epsilon
                adv_alpha = self.adv_alpha
                adv_steps = self.adv_steps

                self.optimizer.zero_grad()

                # Create adversarial examples
                # x_adv = batch.reference.clone().detach()

                x_adv = batch.reference.clone().detach() + torch.empty_like(batch.reference).uniform_(-adv_epsilon,
                                                                                                      adv_epsilon)
                x_adv.requires_grad_(True)

                for _ in range(adv_steps):
                    adv_batch = cebra.data.Batch(reference=x_adv,
                                                 positive=batch.positive,
                                                 negative=batch.negative)
                    adv_output = self._inference(adv_batch)
                    adv_loss = self.criterion(adv_output.reference,
                                              adv_output.positive,
                                              adv_output.negative)[0]

                    grad_x, = torch.autograd.grad(
                        adv_loss, x_adv,
                        retain_graph=False,
                        create_graph=False
                    )
                    with torch.no_grad():
                        x_adv = x_adv + adv_alpha * grad_x.sign()
                        x_adv = torch.max(torch.min(x_adv, batch.reference + adv_epsilon),
                                          batch.reference - adv_epsilon)
                        x_adv.requires_grad = True

                # Final forward pass with adversarial examples
                adv_batch = cebra.data.Batch(
                    reference=x_adv,
                    positive=batch.positive,
                    negative=batch.negative
                )
                output = self._inference(adv_batch)
                adv_loss, _, _ = self.criterion(output.reference,
                                            output.positive,
                                            output.negative)
                adv_loss.backward()
                self.optimizer.step()
            elif self.attack_norm == "l2":
                self.optimizer.zero_grad()

                adv_eps, adv_alpha, adv_steps = self.adv_epsilon, self.adv_alpha, self.adv_steps

                x_adv = batch.reference.clone().detach()
                noise = _l2_normalize(torch.randn_like(x_adv))
                noise *= _rand_radius_like(x_adv) * adv_eps
                x_adv = (batch.reference + noise).clone().detach().requires_grad_(True)

                if not self.adv_aggregate:
                    # ---- original behavior: only the final PGD step is trained on ----
                    for _ in range(adv_steps):
                        if x_adv.grad is not None:
                            x_adv.grad.zero_()

                        adv_b = cebra.data.Batch(reference=x_adv,
                                                 positive=batch.positive,
                                                 negative=batch.negative)
                        adv_out = self._inference(adv_b)
                        adv_loss, _, _ = self.criterion(adv_out.reference,
                                                  adv_out.positive,
                                                  adv_out.negative)

                        grad_x, = torch.autograd.grad(
                            adv_loss, x_adv,
                            retain_graph=False,
                            create_graph=False
                        )
                        with torch.no_grad():
                            x_adv += adv_alpha * _l2_normalize(grad_x)
                            x_adv = _proj_l2_ball(x_adv, batch.reference, adv_eps)
                            x_adv.requires_grad = True

                    adv_b = cebra.data.Batch(reference=x_adv,
                                             positive=batch.positive,
                                             negative=batch.negative)
                    out = self._inference(adv_b)
                    loss3, _, _ = self.criterion(out.reference, out.positive, out.negative)
                    loss3.backward()
                    self.optimizer.step()
                else:
                    # ---- aggregate mode: expose the model to every intermediate
                    # PGD sample (step 0 == clean reference, ..., step adv_steps ==
                    # final adversarial sample), and train on the averaged loss. ----
                    all_steps_loss = 0.0
                    for step in range(adv_steps + 1):
                        if step > 0:
                            adv_b_grad = cebra.data.Batch(reference=x_adv,
                                                          positive=batch.positive,
                                                          negative=batch.negative)
                            adv_out_grad = self._inference(adv_b_grad)
                            adv_loss_for_grad, _, _ = self.criterion(
                                adv_out_grad.reference,
                                adv_out_grad.positive,
                                adv_out_grad.negative,
                            )
                            grad_x, = torch.autograd.grad(
                                adv_loss_for_grad, x_adv,
                                retain_graph=False,
                                create_graph=False,
                            )
                            with torch.no_grad():
                                # NOTE: must be out-of-place (x_adv = x_adv + ...), not x_adv += ...
                                # In aggregate mode, x_adv from earlier steps is still referenced
                                # by the autograd graph of earlier step_loss terms (since we sum
                                # them all into all_steps_loss and backward() only at the end).
                                # An in-place update here would corrupt those earlier graphs and
                                # raise "modified by an inplace operation" on backward().
                                x_adv_new = x_adv + adv_alpha * _l2_normalize(grad_x)
                                x_adv_new = _proj_l2_ball(x_adv_new, batch.reference, adv_eps)
                            x_adv = x_adv_new.detach().requires_grad_(True)

                        adv_b_loss = cebra.data.Batch(reference=x_adv,
                                                      positive=batch.positive,
                                                      negative=batch.negative)
                        out_step = self._inference(adv_b_loss)
                        step_loss, _, _ = self.criterion(
                            out_step.reference, out_step.positive, out_step.negative)
                        all_steps_loss = all_steps_loss + step_loss

                    final_adv_loss = all_steps_loss / (adv_steps + 1)
                    final_adv_loss.backward()
                    self.optimizer.step()

        
        return stats

    # ------------------------------------------------------------
    # 6) logging
    # ------------------------------------------------------------

    def validation(self,
                   loader: cebra.data.Loader,
                   session_id: Optional[int] = None):
        """Compute score of the model on data.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            session_id: The session ID, an integer between 0 and the number of sessions in the
                multisession model, set to None for single session.

        Returns:
            Loss averaged over iterations on data batch.
        """
        assert (session_id is None) or (session_id == 0)
        iterator = self._get_loader(loader)
        total_loss = Meter()
        self.model.eval()
        for _, batch in iterator:
            prediction = self._inference(batch)
            loss, _, _ = self.criterion(prediction.reference,
                                        prediction.positive,
                                        prediction.negative)
            total_loss.add(loss.item())
        return total_loss.average

    @torch.no_grad()
    def decoding(self, train_loader, valid_loader):
        """Deprecated since 0.0.2."""
        train_x = self.transform(train_loader.dataset[torch.arange(
            len(train_loader.dataset.neural))])
        train_y = train_loader.dataset.index
        valid_x = self.transform(valid_loader.dataset[torch.arange(
            len(valid_loader.dataset.neural))])
        valid_y = valid_loader.dataset.index
        decode_metric = train_loader.dataset.decode(
            train_x.cpu().numpy(),
            train_y.cpu().numpy(),
            valid_x.cpu().numpy(),
            valid_y.cpu().numpy(),
        )
        return decode_metric

    @torch.no_grad()
    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the embedding.

        This function by default only applies the ``forward`` function
        of the given model, after switching it into eval mode.

        Args:
            inputs: The input signal

        Returns:
            The output embedding.

        TODO:
            * Remove eval mode
        """

        self.model.eval()
        return self.model(inputs)

    @abc.abstractmethod
    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, return the model outputs.

        TODO: make this a public function?

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        raise NotImplementedError

    def load(self, logdir, filename="checkpoint.pth"):
        """Load the experiment from its checkpoint file.

        Args:
            filename (str): Checkpoint name for loading the experiment.
        """

        savepath = os.path.join(logdir, filename)
        if not os.path.exists(savepath):
            print("Did not find a previous experiment. Starting from scratch.")
            return
        checkpoint = torch.load(savepath, map_location=self.device)
        self.load_state_dict(checkpoint, strict=True)

    def save(self, logdir, filename="checkpoint_last.pth"):
        """Save the model and optimizer params.

        Args:
            logdir: Logging directory for this model.
            filename: Checkpoint name for saving the experiment.
        """
        if not os.path.exists(os.path.dirname(logdir)):
            os.makedirs(logdir)
        savepath = os.path.join(logdir, filename)
        torch.save(
            self.state_dict(),
            savepath,
        )


@dataclasses.dataclass
class MultiobjectiveSolver(Solver):
    """Train models to satisfy multiple learning objectives.

    This variant of the standard :py:class:`cebra.solver.base.Solver` implements multi-objective
    or "hybrid" training.

    Attributes:
        model: A multi-objective CEBRA model
        optimizer: The optimizer used for training.
        num_behavior_features: The feature dimension for the features dedicated
            to satisfy the behavior contrastive objective. The remainder is used
            for time contrastive learning.
        renormalize_features: If ``True``, normalize the behavior and time
            contrastive features individually before computing similarity scores.
    """

    num_behavior_features: int = 3
    renormalize_features: bool = False
    output_mode: Literal["overlapping", "separate"] = "overlapping"

    @property
    def num_time_features(self):
        return self.num_total_features - self.num_behavior_features

    @property
    def num_total_features(self):
        return self.model.num_output

    def __post_init__(self):
        super().__post_init__()
        self._check_dimensions()
        self.model = cebra.models.MultiobjectiveModel(
            self.model,
            dimensions=(self.num_behavior_features, self.model.num_output),
            renormalize=self.renormalize_features,
            output_mode=self.output_mode,
        )

    def _check_dimensions(self):
        """Check the feature dimensions for behavior/time contrastive learning.

        Raises:
            ValueError: If feature dimensions are larger than the model features,
                or not sufficiently large for renormalization.
        """
        if self.output_mode == "separate":
            if self.num_behavior_features >= self.num_total_features:
                raise ValueError(
                    "For multi-objective training, the number of features for "
                    f"behavior contrastive learning ({self.num_behavior_features}) cannot be as large or larger "
                    f"than the total feature dimension ({self.num_total_features})."
                )
            if self.num_time_features >= self.num_total_features:
                raise ValueError(
                    "For multi-objective training, the number of features for "
                    f"time contrastive learning ({self.num_time_features}) cannot be as large or larger "
                    f"than the total feature dimension ({self.num_total_features})."
                )
        if self.renormalize_features:
            if self.num_behavior_features < 2:
                raise ValueError(
                    "When renormalizing the features, the feature dimension needs "
                    "to be at least 2 for behavior. "
                    "Check the values of 'renormalize_features' and 'num_behavior_features'."
                )
            if self.num_time_features < 2:
                raise ValueError(
                    "When renormalizing the features, the feature dimension needs "
                    "to be at least 2 for behavior. "
                    "Check the values of 'renormalize_features' and 'num_time_features'."
                )

    def step(self, batch: cebra.data.Batch) -> dict:
        """Perform a single gradient update with multiple objectives.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics.
        """
        self.optimizer.zero_grad()

        # Needed for the Jacobian regularizer (no-op if jacobian_weight == 0,
        # since requires_grad on a leaf tensor that's never used in backward
        # through autograd.grad has no effect on the existing computation).
        if self.jacobian_weight > 0:
            batch.reference.requires_grad_(True)

        prediction_behavior, prediction_time = self._inference(batch)

        behavior_loss, behavior_align, behavior_uniform = self.criterion(
            prediction_behavior.reference,
            prediction_behavior.positive,
            prediction_behavior.negative,
        )

        time_loss, time_align, time_uniform = self.criterion(
            prediction_time.reference,
            prediction_time.positive,
            prediction_time.negative,
        )

        loss = behavior_loss + time_loss

        # ---- Jacobian regularizer (xCEBRA-style identifiability term) ----
        if self.jacobian_weight > 0:
            jacobian_loss = self._jacobian_reg(
                batch.reference, prediction_behavior.reference)
            loss = loss + self.jacobian_weight * jacobian_loss

        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())

        # 2) ---------------- adversarial branch (optional) ----------------------
        if self.training_mode == "adversarial":
            if self.attack_norm == "linf":
                adv_eps, adv_alpha, adv_steps = (
                    self.adv_epsilon,
                    self.adv_alpha,
                    self.adv_steps,
                )

                # Initialize adversarial examples
                x_adv = batch.reference.detach() + torch.empty_like(batch.reference).uniform_(-adv_eps, adv_eps)
                x_adv = x_adv.clamp(0, 1).requires_grad_(True)

                for _ in range(adv_steps):
                    # Compute gradient wrt input only (not model parameters)
                    adv_batch = cebra.data.Batch(
                        reference=x_adv,
                        positive=batch.positive,
                        negative=batch.negative,
                    )

                    adv_pred_beh, adv_pred_time = self._inference(adv_batch)
                    adv_beh_loss, _, _ = self.criterion(
                        adv_pred_beh.reference,
                        adv_pred_beh.positive,
                        adv_pred_beh.negative,
                    )
                    adv_time_loss, _, _ = self.criterion(
                        adv_pred_time.reference,
                        adv_pred_time.positive,
                        adv_pred_time.negative,
                    )
                    adv_loss = adv_beh_loss + adv_time_loss

                    grad_x, = torch.autograd.grad(
                        adv_loss,
                        x_adv,
                        retain_graph=False,
                        create_graph=False,
                    )

                    # PGD update + projection to ε-ball
                    with torch.no_grad():
                        x_adv.add_(adv_alpha * grad_x.sign())
                        x_adv.clamp_(batch.reference - adv_eps, batch.reference + adv_eps)
                        x_adv.requires_grad_(True)

                # Final adversarial training step
                self.optimizer.zero_grad()
                adv_batch = cebra.data.Batch(
                    reference=x_adv.detach(),
                    positive=batch.positive,
                    negative=batch.negative,
                )
                adv_pred_beh, adv_pred_time = self._inference(adv_batch)
                adv_beh_loss, _, _ = self.criterion(
                    adv_pred_beh.reference,
                    adv_pred_beh.positive,
                    adv_pred_beh.negative,
                )
                adv_time_loss, _, _ = self.criterion(
                    adv_pred_time.reference,
                    adv_pred_time.positive,
                    adv_pred_time.negative,
                )
                adv_total_loss = adv_beh_loss + adv_time_loss
                adv_total_loss.backward()
                self.optimizer.step()
            elif self.attack_norm == "l2":
                adv_eps, adv_alpha, adv_steps = (
                    self.adv_epsilon,
                    self.adv_alpha,
                    self.adv_steps,
                )

                x_adv = batch.reference.clone().detach()
                noise = _l2_normalize(torch.randn_like(x_adv))
                noise *= _rand_radius_like(x_adv) * adv_eps
                x_adv = (batch.reference + noise).clone().detach().requires_grad_(True)

                for _ in range(adv_steps):
                    adv_b = cebra.data.Batch(
                        reference=x_adv,
                        positive=batch.positive,
                        negative=batch.negative,
                    )

                    adv_out, _ = self._inference(adv_b)
                    adv_loss, _, _ = self.criterion(
                        adv_out.reference, adv_out.positive, adv_out.negative
                    )

                    grad_x, = torch.autograd.grad(
                        adv_loss,
                        x_adv,
                        retain_graph=False,
                        create_graph=False,
                    )

                    with torch.no_grad():
                        x_adv += adv_alpha * _l2_normalize(grad_x)
                        x_adv = _proj_l2_ball(x_adv, batch.reference, adv_eps)
                        x_adv.requires_grad_(True)

                self.optimizer.zero_grad()
                adv_b = cebra.data.Batch(
                    reference=x_adv.detach(),
                    positive=batch.positive,
                    negative=batch.negative,
                )
                out, _ = self._inference(adv_b)
                loss3, _, _ = self.criterion(out.reference, out.positive, out.negative)
                loss3.backward()
                self.optimizer.step()

        self.history.append(loss.item())
        return dict(
            behavior_pos=behavior_align.item(),
            behavior_neg=behavior_uniform.item(),
            behavior_total=behavior_loss.item(),
            time_pos=time_align.item(),
            time_neg=time_uniform.item(),
            time_total=time_loss.item(),
        )
