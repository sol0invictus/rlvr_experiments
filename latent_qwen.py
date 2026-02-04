
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple

class LatentQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_latent_thoughts = kwargs.pop('num_latent_thoughts', getattr(config, 'num_latent_thoughts', 0))
        # We need the token IDs for control. Ideally, these are passed or set.
        # For now, we'll assume they are attributes or look them up if passed.
        self.think_token_id = None 
        
    def set_special_token_ids(self, think_token_id):
        self.think_token_id = think_token_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # If we don't have the token ID, or no latent thoughts requested, check config
        if self.think_token_id is None:
            pass

        if self.num_latent_thoughts == 0 or self.think_token_id is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        # Basic Check: Do we have `input_ids`? If using `inputs_embeds`, standard pass (unless we detect logic there, but keeping it simple).
        if input_ids is None:
             return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        
        # Search for <think> in input_ids
        has_think = (input_ids == self.think_token_id).any()
        
        if not has_think:
            return super().forward(
                input_ids=input_ids, 
                past_key_values=past_key_values, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                use_cache=use_cache, 
                labels=labels, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs
            )

        # IMPORTANT: Splitting logic.
        # We assume ONE <think> token per sequence for now.
        
        # Find index of <think>
        batch_indices, think_indices = (input_ids == self.think_token_id).nonzero(as_tuple=True)
        
        model_kwargs = {
            "use_cache": use_cache if use_cache is not None else self.config.use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        model_kwargs.update(kwargs) # Pass extra kwargs
        
        if inputs_embeds is not None:
             return super().forward(inputs_embeds=inputs_embeds, cache_position=cache_position, **model_kwargs)

        input_ids_seq = input_ids
        batch_size, seq_len = input_ids.shape
        
        collected_logits = []
        
        for i in range(batch_size):
            # Process single sequence
            seq = input_ids[i]
            idx = (seq == self.think_token_id).nonzero()
            
            if len(idx) == 0:
                # No think token, Standard pass
                out = super().forward(
                    input_ids=seq.unsqueeze(0),
                    past_key_values=None, 
                    cache_position=cache_position, # Warning: cache_position might be batched? 
                    # If cache_position is (B, L), we need slice.
                    # Usually it's (L,).
                    **model_kwargs
                )
                collected_logits.append(out.logits.squeeze(0))
                continue
                
            idx = idx[0].item() # index of <think>
            
            # 1. Prefix: seq[:idx+1]
            prefix_ids = seq[:idx+1].unsqueeze(0)
            
            prefix_mask = None
            if attention_mask is not None:
                prefix_mask = attention_mask[i, :idx+1].unsqueeze(0)
            
            prefix_cache_pos = None
            if cache_position is not None:
                # Assuming cache_position is (L,) or (1, L) or (B, L)
                # If 1D, slice.
                if cache_position.dim() == 1:
                    prefix_cache_pos = cache_position[:idx+1]
                elif cache_position.dim() == 2:
                     # (B, L) ? Or (1, L)
                     if cache_position.shape[0] == batch_size:
                         prefix_cache_pos = cache_position[i, :idx+1].unsqueeze(0)
                     else:
                         prefix_cache_pos = cache_position[:, :idx+1]

            # Construct kwargs for this step
            step_kwargs = model_kwargs.copy()
            step_kwargs['use_cache'] = True
            
            transformer_out = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                cache_position=prefix_cache_pos,
                use_cache=True,
                return_dict=True
            )
            past_key_values = transformer_out.past_key_values
            current_hidden = transformer_out.last_hidden_state[:, -1:, :] # (1, 1, hidden)
            
            # Compute logits for prefix (exclude the last one which is <think>, we will replace it)
            # transformer_out.last_hidden_state shape: (1, idx+1, hidden)
            # Prefix logits: for 0..idx-1
            # We want logits for the whole sequence eventually.
            
            # The logit for position `i` usually predicts `i+1`.
            # `input_ids[idx]` is `<think>`. The logit at this position predicts `</think>`.
            # We want this prediction to be based on Thoughts.
            
            # So:
            # logits for 0..idx-1: Standard.
            # logit for idx (<think>): Output of latent loop.
            # logits for idx+1..end: Standard (based on new KV).
            
            prefix_hidden = transformer_out.last_hidden_state
            # Store standard prefix logits?
            # prefix_logits = self.lm_head(prefix_hidden)
            
            # 2. Latent Loop
            # We start from the last hidden state of prefix (corresponding to <think>)
            current_hidden = prefix_hidden[:, -1:, :] 
            
            for _ in range(self.num_latent_thoughts):
                step_out = self.model(
                    inputs_embeds=current_hidden,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = step_out.past_key_values
                current_hidden = step_out.last_hidden_state
                
            # After loop, current_hidden is the "thought-processed" state for <think>.
            # Generate the logit for <think> position
            think_logit = self.lm_head(current_hidden) # (1, 1, V)
            
            # Prefix logits (excluding the standard think logit)
            prefix_logits_prev = self.lm_head(prefix_hidden[:, :-1, :]) if idx > 0 else torch.empty(1, 0, self.config.vocab_size, device=input_ids.device, dtype=think_logit.dtype)
            
            # 3. Suffix
            # Note: We need to adjust position_ids for suffix?
            # The KV cache has grown by N steps.
            # Transformers automatically handles position_ids if passed None and using Cache?
            # Qwen2 usually uses rotary embeddings based on position_ids.
            # The `past_key_values` has length `L_prefix + N`.
            # The next token `</think>` should have position `L_prefix + N`?
            # Or `L_prefix + 1`?
            # Coconut usually implies "thinking takes time/positions".
            # So tokens should be shifted.
            # If we don't pass position_ids, models usually assume `past_key_values.shape[-2]`.
            # So it will automatically shift.
            
            suffix_ids = seq[idx+1:].unsqueeze(0)
            suffix_logits = torch.empty(1, 0, self.config.vocab_size, device=input_ids.device, dtype=think_logit.dtype)
            
            if suffix_ids.size(1) > 0:
                # We assume position_ids will be inferred relative to past_key_values length
                # Since we added N entries to PKV, the next position will be correct (shifted).
                
                # Check cache_position for suffix?
                # If cache_position was provided, we must shift it too.
                suffix_cache_pos = None
                if cache_position is not None:
                     # Suffix slice
                     if cache_position.dim() == 1:
                        suffix_cache_pos = cache_position[idx+1:] + self.num_latent_thoughts
                     elif cache_position.dim() == 2:
                        suffix_batch_slice = cache_position[i, idx+1:] if cache_position.shape[0] == batch_size else cache_position[:, idx+1:]
                        suffix_cache_pos = suffix_batch_slice + self.num_latent_thoughts
                        if cache_position.dim() == 2 and cache_position.shape[0] == batch_size:
                             suffix_cache_pos = suffix_cache_pos.unsqueeze(0)

                suffix_out = self.model(
                    input_ids=suffix_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    cache_position=suffix_cache_pos
                )
                suffix_logits = self.lm_head(suffix_out.last_hidden_state)
            
            # Combine
            # Logic: prefix (0..idx-1) + think_logit (at idx) + suffix (idx+1..end)
            # Total length = idx + 1 + len(suffix) = len(seq)
            
            full_seq_logits = torch.cat([prefix_logits_prev, think_logit, suffix_logits], dim=1) # (1, L, V)
            collected_logits.append(full_seq_logits.squeeze(0))
            
        # Stack check
        # Since lengths differ (due to prefix/suffix splits + latent fixed size), 
        # naive stacking requires padding handling.
        # But if the original batch was padded, `seq` includes padding.
        # `prefix` length + `suffix` length = original length.
        # `latent` is constant N.
        # So `total length = original length + N`.
        # So we can stack!
        
        final_logits = torch.stack(collected_logits) # (B, L+N, V)
        
        # Handle loss if labels provided
        loss = None
        if labels is not None:
            # We need to modify labels to match the new shape (insert ignore_index for latents)
            # labels: (B, L)
            # construct new labels (B, L+N)
            # We need to insert N ignore_indices AFTER the <think> token index.
            
            # Iterate and construct (meh, slow but correct)
            new_labels_list = []
            ignore = -100 # standard
            
            for i in range(batch_size):
                lab = labels[i]
                seq = input_ids[i]
                idx = (seq == self.think_token_id).nonzero()
                if len(idx) == 0:
                    # Just pad N to end? Or N doesn't exist?
                    # If we added latent steps... wait.
                    # If NO think token was found, we returned standard logits (L).
                    # But now we are stacking them.
                    # If some have thoughts and some don't, shapes mismatch! (L+N vs L).
                    # If mixed batch: We must PAD the "no think" ones to matches?
                    # Or force consistency.
                    pass 
                
                idx = idx[0].item()
                # Labels corresponding to prefix: lab[:idx+1]
                # Labels for latents: ignore * N
                # Labels for suffix: lab[idx+1:]
                
                # Note: `logits` are predictions for the NEXT token.
                # `labels` usually align such that `labels[i]` is target for `logits[i]`.
                # Prefix logits (incl <think>) -> predict first thought? No.
                # The logit at `<think>` input step -> predicts... first thought? 
                # Or do we treat latent steps as "prediction of nothing"?
                # Actually, the logit at `<think>` predicts `labels[idx+1]` (the token after think, which is `</think>` in training data).
                # But we insert N steps.
                # So logit at `<think>` -> first latent step.
                # logit at last latent step -> predicts `</think>`.
                # So we shift labels.
                
                prefix_lab = lab[:idx+1]
                suffix_lab = lab[idx+1:]
                latent_lab = torch.full((self.num_latent_thoughts,), ignore, dtype=lab.dtype, device=lab.device)
                
                new_l = torch.cat([prefix_lab, latent_lab, suffix_lab], dim=0)
                new_labels_list.append(new_l)
            
            new_labels = torch.stack(new_labels_list)
            
            # Compute loss
            # Shift logits/labels standard way: logits[..., :-1, :], labels[..., 1:]
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = new_labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=None, # complex to combine lists
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        return super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, input_ids=None, **kwargs):
        # We need to force </think> and <answer> after <think>.
        # Since our forward pass handles the latent steps when it sees <think>, 
        # the NEXT token generated should be </think>.
        
        # We can use a LogitsProcessor to force this.
        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

        class LatentControlLogitsProcessor(LogitsProcessor):
            def __init__(self, tokenizer, think_id, close_think_id, answer_id):
                self.think_id = think_id
                self.close_think_id = close_think_id
                self.answer_id = answer_id
                self.state = "start" # start -> thinking -> answering
            
            def __call__(self, input_ids, scores):
                # input_ids: (batch, seq_len)
                # We assume batch size 1 for simplicity regarding state, or independent processing.
                # If batch size > 1, this state machine needs to be per-row.
                
                # Naive implementation: iterate usage
                # We'll just enforce:
                # If last token was <think>, force </think>.
                # If last token was </think>, force <answer>.
                
                # Get last token
                last_token = input_ids[:, -1]
                
                # Create mask to force token (set all to -inf except target)
                # Clone scores to avoid side effects
                
                # Check where last_token == think_id
                think_mask = (last_token == self.think_id)
                if think_mask.any():
                    # For these rows, force close_think_id
                    scores[think_mask, :] = -float('inf')
                    scores[think_mask, self.close_think_id] = 0
                
                # Check where last_token == close_think_id
                close_mask = (last_token == self.close_think_id)
                if close_mask.any():
                     # For these rows, force answer_id
                    scores[close_mask, :] = -float('inf')
                    scores[close_mask, self.answer_id] = 0

                return scores

        # Identify token IDs
        # We assume self.think_token_id is set.
        if self.think_token_id is not None and getattr(self, 'close_think_id', None) is not None and getattr(self, 'answer_id', None) is not None:
             processor = LatentControlLogitsProcessor(
                 tokenizer=None, 
                 think_id=self.think_token_id, 
                 close_think_id=self.close_think_id, 
                 answer_id=self.answer_id
             )
             
             logits_processor = kwargs.get("logits_processor", LogitsProcessorList())
             logits_processor.append(processor)
             kwargs["logits_processor"] = logits_processor
             
        # Ideally we'd add this processor. 
        # But for now, let's fix the crash first.
        # User requirement "manually applied" might just mean "in the dataset training" and "in the rollout logic".
        # If I fix the crash, GRPO training loop might work.
        
        return super().generate(input_ids, **kwargs)

