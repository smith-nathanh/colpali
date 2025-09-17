"""ColFastVLM modeling wrapper.

Provides ColBERT-style multi-vector embeddings over FastVLM (LLaVA-Qwen2 based) models.
"""

from typing import ClassVar, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class ColFastVLM(nn.Module):
    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(
        self,
        pretrained_model_name_or_path: str = "apple/FastVLM-0.5B",
        mask_non_image_embeddings: bool = False,
        dim: int = 128,
        fuse_in_decoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self.config = self.model.config
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.dim = dim
        self.fuse_in_decoder = fuse_in_decoder
        self.custom_text_proj = nn.Linear(self.config.hidden_size, dim)
        nn.init.normal_(self.custom_text_proj.weight, std=0.02)
        nn.init.zeros_(self.custom_text_proj.bias)
        # Move custom_text_proj to the same device and dtype as the model
        if hasattr(self.model, "device"):
            self.custom_text_proj = self.custom_text_proj.to(device=self.model.device, dtype=self.model.dtype)
        else:
            # Get device and dtype from model parameters
            param = next(self.model.parameters())
            self.custom_text_proj = self.custom_text_proj.to(device=param.device, dtype=param.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Remove unused arguments to avoid conflicts
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        # Convert images to correct dtype if provided
        if images is not None:
            images = images.to(dtype=self.model.dtype)

        use_fused = images is not None and self.fuse_in_decoder
        visual_embeddings = None

        # Debug: Check embeddings before normalization
        debug = torch.rand(1).item() < 0.05  # 5% chance to debug

        if use_fused:
            # When using fused processing, the LlavaQwen2 model expects images
            # to be passed and will handle the fusion internally
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    images=images,
                    image_sizes=image_sizes,
                    return_dict=True,
                    **kwargs,
                )
                hidden = outputs.hidden_states[-1]
                attn_mask = attention_mask
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Fused decoder path failed with error: {e}\n"
                    f"Falling back to non-fused processing.\n"
                    f"This may impact performance. Consider setting fuse_in_decoder=False."
                )
                # Fall back to non-fused processing
                use_fused = False

        if use_fused:
            # Store the position where vision features start for masking
            if hasattr(outputs, "vision_feature_positions"):
                self._fused_visual_start = outputs.vision_feature_positions
            elif input_ids is not None:
                # Fallback: assume vision features come after text
                self._fused_visual_start = input_ids.size(1)

            # Debug fused path
            if debug:
                print(f"  Fused path - input_ids: {input_ids.size(1) if input_ids is not None else None}")
                print(f"  Fused path - hidden: {hidden.shape}")
                print(f"  Images provided: {images is not None}")
                if images is not None:
                    print(f"  Image shape: {images.shape}")

            # Detect appended visual tokens (visual patches injected during multimodal prep)
            if input_ids is not None and hidden.size(1) > input_ids.size(1):
                extra = hidden.size(1) - input_ids.size(1)
                if debug:
                    print(f"  Visual tokens detected: {extra}")
                if attention_mask is not None:
                    extra_mask = torch.ones(
                        attention_mask.size(0),
                        extra,
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attn_mask = torch.cat([attention_mask, extra_mask], dim=1)
                self._fused_visual_start = hidden.size(1) - extra
                self._fused_visual_count = extra
            else:
                if debug:
                    print("  No visual tokens detected in fused path!")
                    print("  This suggests FastVLM isn't processing images correctly")

        if not use_fused:
            if images is not None:
                try:
                    # Access the vision tower and projector correctly for LLaVA-Qwen2 architecture
                    vision_tower = self.model.model.vision_tower
                    mm_projector = self.model.model.mm_projector

                    # Process images through vision tower
                    # vision_tower expects images and returns features
                    vision_outputs = vision_tower(images)

                    # The vision tower output needs to be projected
                    # mm_projector is a Sequential module (3072->896->896)
                    visual_embeddings = mm_projector(vision_outputs)

                    # Ensure correct shape (batch_size, num_patches, hidden_dim)
                    if visual_embeddings.dim() == 2:
                        # If we get (batch*patches, dim), reshape to (batch, patches, dim)
                        bsz = images.size(0)
                        npatch = visual_embeddings.size(0) // bsz
                        visual_embeddings = visual_embeddings.reshape(bsz, npatch, -1)
                except (AttributeError, RuntimeError) as e:
                    import warnings

                    warnings.warn(
                        f"Failed to process visual embeddings: {e}\n"
                        f"Vision tower available: {hasattr(self.model.model, 'vision_tower')}\n"
                        f"MM projector available: {hasattr(self.model.model, 'mm_projector')}\n"
                        "Falling back to text-only mode."
                    )
                    visual_embeddings = None
            text_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                images=None,
                image_sizes=None,
                return_dict=True,
                **kwargs,
            )
            text_hidden = text_out.hidden_states[-1]
            if visual_embeddings is not None:
                hidden = torch.cat([text_hidden, visual_embeddings], dim=1)
                if attention_mask is not None:
                    vmask = torch.ones(
                        visual_embeddings.size(0),
                        visual_embeddings.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attn_mask = torch.cat([attention_mask, vmask], dim=1)
                else:
                    attn_mask = None
            else:
                hidden = text_hidden
                attn_mask = attention_mask

        proj = self.custom_text_proj(hidden)

        if debug:
            print(f"FastVLM debug - proj_shape: {proj.shape}")
            print(f"  proj_mean: {proj.mean():.6f}, proj_std: {proj.std():.6f}")
            print(f"  proj_norm_mean: {proj.norm(dim=-1).mean():.6f}")

        proj = proj / proj.norm(dim=-1, keepdim=True)

        if attn_mask is not None:
            proj = proj * attn_mask.unsqueeze(-1)
            if debug:
                print(f"  After masking - proj_mean: {proj.mean():.6f}")
                print(f"  Non-zero tokens: {(attn_mask.sum(dim=1).float().mean()):.1f}")
                print(f"  Images provided: {images is not None}")
                if images is not None:
                    print(f"  Visual masking will apply: {self.mask_non_image_embeddings}")
                    if hasattr(self, "_fused_visual_start"):
                        print(f"  Fused visual start: {self._fused_visual_start}, count: {self._fused_visual_count}")
                    elif input_ids is not None:
                        print(f"  Text len: {input_ids.size(1)}, proj len: {proj.size(1)}")

        if (
            self.mask_non_image_embeddings
            and images is not None
            and input_ids is not None
            and proj.size(1) > input_ids.size(1)
        ):
            if use_fused and hasattr(self, "_fused_visual_start"):
                start = self._fused_visual_start
                mask_vec = torch.zeros_like(proj[:, :, 0])
                mask_vec[:, start:] = 1.0
                proj = proj * mask_vec.unsqueeze(-1)
                if debug:
                    print(
                        f"  Applied fused visual mask: start={start}, kept {mask_vec.sum() / mask_vec.numel():.2%} tokens"
                    )
            else:
                text_len = input_ids.size(1)
                mask_vec = torch.zeros_like(proj[:, :, 0])
                mask_vec[:, text_len:] = 1.0
                proj = proj * mask_vec.unsqueeze(-1)
                if debug:
                    print(
                        f"  Applied visual mask: text_len={text_len}, kept {mask_vec.sum() / mask_vec.numel():.2%} tokens"
                    )
                    visual_tokens = proj.size(1) - text_len
                    print(f"  Visual tokens: {visual_tokens}")
                    print(
                        f"  Final proj stats: mean={proj.mean():.6f}, nonzero_frac={proj.abs().sum() / (proj.numel()):.2%}"
                    )

        return proj

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        import os

        torch.save(self.custom_text_proj.state_dict(), os.path.join(save_directory, "custom_text_proj.pt"))

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    @property
    def patch_size(self) -> int:
        # FastVLM/MobileCLIP uses 64x64 patches
        if hasattr(self.model.model, "vision_tower") and hasattr(self.model.model.vision_tower, "config"):
            vision_config = self.model.model.vision_tower.config
            if isinstance(vision_config, dict) and "image_cfg" in vision_config:
                return vision_config["image_cfg"].get("patch_size", 64)
        return 64

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
