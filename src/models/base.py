"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias
    
    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, LayerNorm):  
        module_output = DynamicTanh(module.weight.shape)
    else:
        for name, child in module.named_children():
            module_output.add_module(name, convert_ln_to_dyt(child))
    return module_output




class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)




class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config, exp_factor=1.0):
        super().__init__()
        self.dim_exp_factor = exp_factor * 4

        self.c_fc = nn.Linear(
            config.n_embd, int(self.dim_exp_factor * config.n_embd), bias=config.bias
        )
        self.c_proj = nn.Linear(
            int(self.dim_exp_factor * config.n_embd), config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.parallel = config.parallel_block
        if not self.parallel:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, *args, **kwargs):
        if self.parallel:
            # from GPT-J 6B https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L299
            x_ln = self.ln_1(x, *args, **kwargs)
            x_attn = self.attn(x_ln)
            x_ffn = self.mlp(x_ln)
            x = x + x_attn + x_ffn
        else:
            x = x + self.attn(self.ln_1(x, *args, **kwargs))
            x_ = self.mlp(self.ln_2(x, *args, **kwargs))
            x = x + x_
        return x


class GPTBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if self.config.use_dynamic_tanh:
            self.transformer = convert_ln_to_dyt(self.transformer)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.init_std / math.sqrt(2 * config.n_layer),
                )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # forward pass through all the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {
            "logits": logits,
            "loss": loss,
        }

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:sequence_length]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :sequence_length, :sequence_length]

    def from_pretrained(
        self,
        model_path,
    ):
        paths = model_path.split(",")
        if len(paths) == 1:
            # TODO: with distributed?
            loaded_state = torch.load(
                str(model_path + "/ckpt.pt"),
                map_location=torch.device(self.config.device),
            )
            state_to_load = loaded_state["model"]

            # load the sparse model
            state_to_load = {
                ".".join(k.split(".")[1:]): v  # drop _orig_mod from keys
                for k, v in state_to_load.items()
            }

    def get_parameter_group_specs(self):
        """
        This function separates parameters into those that will experience weight decay for regularization
        and those that won't (e.g., biases, LayerNorm, and DynamicTanh parameters). The returned specs
        are then used to set up optimizer parameter groups.
        """

        # Separate parameters for decay/no_decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        
        # Need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        # Iterate over all modules and their parameters
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # Full param name

                # Handle DynamicTanh modules: All parameters in DynamicTanh should not have weight decay
                if isinstance(m, DynamicTanh):
                    no_decay.add(fpn)  # Add all parameters of DynamicTanh to no_decay

                # Handle LayerNorm modules: Bias and weight of LayerNorm should not have weight decay
                elif isinstance(m, torch.nn.LayerNorm):
                    if pn == "weight" or pn == "bias":
                        no_decay.add(fpn)  # Bias and weight in LayerNorm should not decay

                # Handle biases: All biases should not have weight decay
                elif pn.endswith("bias"):
                    no_decay.add(fpn)

                # Handle weights of whitelist modules: These should have weight decay
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                # Handle weights of blacklist modules: These should not have weight decay
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    no_decay.add(fpn)

        # Remove lm_head.weight from decay set as it's tied to transformer.wte.weight
        decay.remove("lm_head.weight")

        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # Return parameter groups for optimizer
        return [
            {"params": sorted(list(decay))},  # Parameters with weight decay
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},  # Parameters without weight decay
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = (
            torch.tensor(
                self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
            )
            .view(1, -1)
            .to(self.lm_head.weight.device)
        )
        out_idx = (
            self.generate(idx, max_new_tokens, temperature, top_k)
            .view(-1)
            .to("cpu")
            .numpy()
        )
        return self.tokenizer.decode(out_idx)
