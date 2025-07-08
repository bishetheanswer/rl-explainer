import copy
from enum import StrEnum, auto
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor
from torch import nn
from torch.nn.modules.conv import Conv2d


class BaseSaliency:
    def __init__(self, model: DQN) -> None:
        self.model = model
        self.device = model.device

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Converts the observation to a tensor and permutes the dimensions from (N, H, W, C) to (N, C, H, W).
        """
        return obs_as_tensor(obs, self.device).float().permute(0, 3, 1, 2)

    def _normalize_map(self, activation_map: np.ndarray) -> np.ndarray:
        """
        Normalizes the activation map to the 0-255 range for colorization.
        We add 1e-8 to avoid division by zero.
        """
        return (activation_map - np.min(activation_map)) / (
            np.max(activation_map) - np.min(activation_map) + 1e-8
        )

    def get_saliency(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def generate_superimposed_map(
        self, obs: np.ndarray, action: np.ndarray, alpha: float = 0.4, **kwargs
    ) -> np.ndarray:
        saliency_map = self.get_saliency(obs, action, **kwargs)

        # process the saliency map
        heatmap_normalized = self._normalize_map(saliency_map)
        heatmap_uint8 = np.uint8(255 * heatmap_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # process the observation
        last_frame = obs[0, :, :, -1]
        last_frame_colored = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        last_frame_colored = cv2.cvtColor(last_frame_colored, cv2.COLOR_BGR2RGB)

        superimposed_img = (heatmap_colored * alpha) + (
            last_frame_colored * (1 - alpha)
        )
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img


class VanillaGrad(BaseSaliency):
    def __init__(self, model: DQN) -> None:
        self.model = model
        self.device = model.device

    def get_saliency(
        self, obs: np.ndarray, action: np.ndarray, method: str = "max"
    ) -> np.ndarray:
        self.model.q_net.eval()

        obs_tensor = self._obs_to_tensor(obs)
        obs_tensor.requires_grad = True

        q_values = self.model.q_net(obs_tensor)
        self.model.q_net.zero_grad()
        saliency_q_value = q_values[0, action[0]]

        saliency_q_value.backward()

        match method:
            case "all":
                # visualize all 4 frames stacked
                saliency_map = obs_tensor.grad.abs().cpu().numpy()
            case "last":
                # visualize the last frame
                saliency_map = obs_tensor.grad.abs().cpu().numpy()[0, -1, :, :]
            case "max":
                # visualize the max among all
                saliency_map = obs_tensor.grad.abs().squeeze(0).cpu().numpy()
                saliency_map = np.max(saliency_map, axis=0)
            case _:
                raise ValueError(f'Method "{method}" not supported.')

        return saliency_map

    def generate_superimposed_map(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        alpha: float = 0.4,
        method: str = "max",
    ) -> np.ndarray:
        return super().generate_superimposed_map(obs, action, alpha, method=method)


class ScoreCAM(BaseSaliency):
    def __init__(self, model: DQN, target_layer: int) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = model.device

    def _upsample_map(self, activation_map: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            activation_map, size=[84, 84], mode="bilinear", align_corners=False
        )

    def _normalize_map(self, activation_map: torch.Tensor) -> torch.Tensor:
        return (activation_map - torch.min(activation_map)) / (
            torch.max(activation_map) - torch.min(activation_map)
        )

    def _get_q_value(self, x: torch.Tensor, action_idx: int) -> float:
        q_values = self.model.q_net(x)
        return q_values[:, action_idx].item()

    def get_saliency(self, obs: np.ndarray, action_idx: int) -> np.ndarray:
        obs_tensor = self._obs_to_tensor(obs)

        conv_output = None
        x = obs_tensor.clone()
        with torch.no_grad():
            for i, layer in enumerate(self.model.q_net.features_extractor.cnn):
                x = layer(x)
                if i == self.target_layer:
                    conv_output = x
                    break

        weights = []

        # iterate over the channels (each activation map)
        with torch.no_grad():
            for i in range(conv_output.shape[1]):
                activation_map = conv_output[:, i, :, :].unsqueeze(
                    1
                )  # without unsqueeze, the shape is (1, 7, 7) and we need (1, 1, 7, 7)
                upsampled_map = self._upsample_map(activation_map)
                norm_upsampled_map = self._normalize_map(upsampled_map)
                masked_input = obs_tensor.clone() * norm_upsampled_map
                score = self._get_q_value(masked_input, action_idx)

                weights.append(score)

            weights_tensor = torch.tensor(weights).reshape(1, -1, 1, 1)
            # multiply each activation map by its weight and sum them up
            scorecam_map = torch.sum(weights_tensor * conv_output, dim=1, keepdim=True)
            scorecam_map = F.relu(scorecam_map)
            final_heatmap = self._upsample_map(scorecam_map)
            final_heatmap = self._normalize_map(final_heatmap)

            return final_heatmap.squeeze().cpu().numpy()


class GuidedBackprop(BaseSaliency):
    def __init__(self, model: DQN) -> None:
        self.model = model
        self.device = model.device
        self.hooks = []
        self._register_hooks()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_in: Tuple[torch.Tensor, ...],
        grad_out: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        new_grads = []
        for grad in grad_in:
            if isinstance(grad, torch.Tensor):
                # apply the guided backprop rule: clamp negative gradients to zero
                new_grads.append(torch.clamp(grad, min=0.0))
            else:
                new_grads.append(grad)

        return tuple(new_grads)

    def _register_hooks(self) -> None:
        for module in self.model.q_net.features_extractor.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(
                    module.register_full_backward_hook(self._backward_hook)
                )

    def _remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def get_saliency(
        self, obs: np.ndarray, action: np.ndarray, method: str = "max"
    ) -> np.ndarray:
        self.model.q_net.eval()

        obs_tensor = self._obs_to_tensor(obs)
        obs_tensor.requires_grad = True

        q_values = self.model.q_net(obs_tensor)
        self.model.q_net.zero_grad()
        saliency_q_value = q_values[0, action[0]]

        saliency_q_value.backward()

        saliency_map = obs_tensor.grad.abs()

        self._remove_hooks()

        match method:
            case "all":
                # visualize all 4 frames stacked
                saliency_map = obs_tensor.grad.abs().cpu().numpy()
            case "last":
                # visualize the last frame
                saliency_map = obs_tensor.grad.abs().cpu().numpy()[0, -1, :, :]
            case "max":
                # visualize the max among all
                saliency_map = obs_tensor.grad.abs().squeeze(0).cpu().numpy()
                saliency_map = np.max(saliency_map, axis=0)
            case _:
                raise ValueError(f'Method "{method}" not supported.')

        return saliency_map

    def generate_superimposed_map(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        alpha: float = 0.4,
        method: str = "max",
    ) -> np.ndarray:
        return super().generate_superimposed_map(obs, action, alpha, method=method)


class GradCAM(BaseSaliency):
    def __init__(self, model: DQN, target_layer: Conv2d) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = model.device

        self.activations = None
        self.gradients = None
        self.hooks = []

    def _forward_hook(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.activations = output

    def _backward_hook(
        self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        self.gradients = grad_output[0]

    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_saliency(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.model.q_net.eval()

        # register hooks
        forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        backward_hook = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )
        self.hooks.extend([forward_hook, backward_hook])

        obs_tensor = self._obs_to_tensor(obs)

        # forward pass
        q_values = self.model.q_net(obs_tensor)

        # backward pass
        self.model.q_net.zero_grad()
        target_q_value = q_values[0, action[0]]
        target_q_value.backward()

        self._remove_hooks()

        gradients = self.gradients.squeeze(0)
        activations = self.activations.squeeze(0)

        weights = torch.mean(gradients, dim=[1, 2])

        # create heatmap
        heatmap = torch.zeros(activations.shape[1:]).to(self.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i, :, :]

        heatmap = F.relu(heatmap)

        # upsample to original size
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        )

        return heatmap.squeeze().detach().cpu().numpy()


class GuidedGradCAM(BaseSaliency):
    def __init__(self, model: DQN, target_layer: Conv2d) -> None:
        self.model = model
        self.device = model.device

        self.grad_cam = GradCAM(model, target_layer)
        # need to pass a deepcopy of the model to avoid issues with the hooks
        self.guided_backprop = GuidedBackprop(copy.deepcopy(model))

    def get_saliency(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        grad_cam_map = self.grad_cam.get_saliency(obs, action)
        guided_backprop_map = self.guided_backprop.get_saliency(
            copy.deepcopy(obs), action
        )

        return np.multiply(grad_cam_map, guided_backprop_map)


class SaliencyMethod(StrEnum):
    VANILLA = auto()
    SMOOTH = auto()
    SCORE_CAM = auto()
    GUIDED_BACKPROP = auto()
    GRAD_CAM = auto()
    GUIDED_GRAD_CAM = auto()


def initialize_saliency_method(
    saliency_method: SaliencyMethod, model: DQN, target_layer: int | Conv2d
) -> BaseSaliency:
    match saliency_method:
        case SaliencyMethod.VANILLA:
            return VanillaGrad(model)
        case SaliencyMethod.SCORE_CAM:
            if not isinstance(target_layer, int):
                raise ValueError("target_layer needs to be an int")
            return ScoreCAM(model, target_layer)
        case SaliencyMethod.GUIDED_BACKPROP:
            return GuidedBackprop(model)
        case SaliencyMethod.GRAD_CAM:
            if not isinstance(target_layer, Conv2d):
                raise ValueError("target layer needs to be a Conv2d layer")
            return GradCAM(model, target_layer)
        case SaliencyMethod.GUIDED_GRAD_CAM:
            if not isinstance(target_layer, Conv2d):
                raise ValueError("target layer needs to be a Conv2d layer")
            return GuidedGradCAM(model, target_layer)
        case _:
            raise ValueError(f"Saliency method {saliency_method} not supported")


class SmoothGrad(BaseSaliency):
    def __init__(
        self,
        base_saliency_generator: BaseSaliency,
        n_samples: int = 25,
        noise_std_dev: float = 0.15,
    ) -> None:
        self.base_generator = base_saliency_generator
        self.n_samples = n_samples
        self.noise_std_dev = noise_std_dev
        self.device = self.base_generator.device

    def get_saliency(
        self, obs: np.ndarray, action: np.ndarray, method: str = "max"
    ) -> np.ndarray:
        normalized_obs = obs / 255
        normalized_obs = torch.from_numpy(normalized_obs).to(self.device)

        total_saliency = None
        for _ in range(self.n_samples):
            noise = torch.randn_like(normalized_obs) * self.noise_std_dev
            noisy_normalized_obs = normalized_obs + noise
            noisy_denormalized_obs = noisy_normalized_obs.cpu().numpy() * 255

            saliency_map = self.base_generator.get_saliency(
                noisy_denormalized_obs, action, method
            )

            if total_saliency is None:
                total_saliency = saliency_map
            else:
                total_saliency += saliency_map

        return total_saliency / self.n_samples

    def generate_superimposed_map(
        self, obs: np.ndarray, action: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generates a smoothed saliency map and superimposes it on the original observation.
        """
        saliency_map = self.get_saliency(obs, action, method="max")

        heatmap_normalized = (saliency_map - np.min(saliency_map)) / (
            np.max(saliency_map) - np.min(saliency_map) + 1e-8
        )
        heatmap_uint8 = np.uint8(255 * heatmap_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        last_frame = obs[0, :, :, -1]
        last_frame_colored = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        last_frame_colored = cv2.cvtColor(last_frame_colored, cv2.COLOR_BGR2RGB)

        superimposed_img = (heatmap_colored * alpha) + (
            last_frame_colored * (1 - alpha)
        )
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img
