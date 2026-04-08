# Wan Animate `ulysses_fsdp` Walkthrough

This note documents the current Wan Animate inference path in MaxDiffusion when we run:

- `attention=ulysses_fsdp`
- `ici_fsdp_parallelism=-1`
- `ici_context_parallelism=1`

The goal of this mode is to reuse a single physical mesh axis, `fsdp`, for both:

- model sharding of weights
- sequence sharding inside attention

## 1. Mental Model: Ideal vs Current Code

The intended mental model is:

```text
attention input:
(b, s / n, n_heads, d_head)
  -> all_to_all over fsdp
  -> (b, s, n_heads / n, d_head)
  -> local full-sequence splash attention
  -> all_to_all over fsdp
  -> (b, s / n, n_heads, d_head)
```

That part now matches the code in `attention_flax.py`.

For linear / FFN layers, the ideal model is:

```text
activations: seq-sharded
weights: fsdp-sharded
compiler: insert the needed gather / reduce-scatter / reshard ops
```

Important caveat:

- We do not currently hand-write explicit col-parallel / row-parallel FFN kernels.
- We rely on JAX / XLA SPMD to legalize the mixed layout between sequence-sharded activations and fsdp-sharded parameters.
- So the attention path now explicitly matches the Ulysses mental model.
- The linear path is still compiler-driven rather than an explicit two-kernel parallel schedule.

That means:

- `q / k / v` are no longer replicated for the local attention kernel path.
- The attention collective pattern is the two `all_to_all`s you expect.
- Linear and FFN layers may still reshard through compiler-inserted collectives when the activation and parameter layouts disagree.

## 2. Effective Logical Axes

With `attention=ulysses_fsdp`, the important effective logical mappings are:

- `activation_length -> fsdp`
- `activation_kv_length -> fsdp`
- `activation_self_attn_q_length -> fsdp`
- `activation_self_attn_kv_length -> fsdp`
- `activation_cross_attn_q_length -> fsdp`
- `activation_cross_attn_kv_length -> fsdp`
- `embed -> fsdp`
- `conv_out -> fsdp`
- `batch -> data`
- `conv_batch -> data`

So in this mode:

- sequence is sharded on `fsdp`
- attention heads are not explicitly mesh-sharded in the splash kernel itself
- Wan VAE / patch / projection weights use `fsdp`, not `context`

## 3. Exact Runtime Shape Math for 720x1280, 121 Frames, Prev=5, Steps=40

Assume:

- `height=720`
- `width=1280`
- `num_frames=121`
- `segment_frame_length=77`
- `prev_segment_conditioning_frames=5`
- `num_inference_steps=40`

Current Wan Animate pipeline uses:

- VAE spatial scale factor = `8`
- VAE temporal scale factor = `4`
- transformer patch size = `(1, 2, 2)`
- Wan latent channel count `z_dim = 16`

Derived latent sizes:

- latent height = `720 / 8 = 90`
- latent width = `1280 / 8 = 160`
- latent frames per segment = `(77 - 1) / 4 + 1 = 20`
- transformer temporal slots per segment = `20 + 1 = 21`

Segment arithmetic:

- effective new frames per segment = `77 - 5 = 72`
- `last_frames = (121 - 5) % 72 = 44`
- pad frames needed = `72 - 44 = 28`
- target frames after padding = `121 + 28 = 149`
- number of segments = `149 / 72 = 2`

So this run becomes:

- segment 0: pixel frames `0..76`
- segment 1: pixel frames `72..148`
- segment 1 reuses frames `72..76` as overlap conditioning
- final decoded video length before trim = `149`
- final output length after trim = `121`

## 4. End-to-End Pipeline Flow

## 4.1 Inputs

`generate_wan_animate.py` loads:

- reference image
- pose video
- face video

For the standard animate mode:

- `reference_image_path` gives the subject identity
- `pose_video_path` gives body motion
- `face_video_path` gives face motion

## 4.2 Prompt Encoding

The pipeline encodes:

- text prompt with UMT5
- reference image with CLIP vision encoder

The transformer later receives:

- `encoder_hidden_states`: text embeddings
- `encoder_hidden_states_image`: image embeddings

Inside the transformer condition embedder:

- timestep embedding is projected
- text embedding is projected to model width
- image embedding is projected and concatenated ahead of text tokens when present

## 4.3 Reference Image -> Reference Latents

The reference image is resized and VAE-encoded once:

- input pixel shape: `(B, C, H, W)`
- promoted to one-frame video: `(B, C, 1, H, W)`
- VAE latent shape after encode: `(B, 1, 90, 160, 16)`

Then the pipeline prepends an I2V conditioning mask:

- mask shape: `(B, 1, 90, 160, 4)`
- reference latent package: `(B, 1, 90, 160, 20)`

This is the "identity anchor" for every segment.

## 4.4 Pose Video -> Pose Latents

Each segment slice of the pose video is VAE-encoded:

- pixel segment shape: `(B, 3, 77, 720, 1280)`
- latent shape: `(B, 20, 90, 160, 16)`

These latents are fed to the transformer's pose patch embedding path.

## 4.5 Previous Segment Conditioning

For segment 0:

- there is no previous decoded segment
- the pipeline builds zero conditioning frames for animate mode

For segment 1:

- the last `5` decoded pixel frames from segment 0 are reused
- they are re-encoded through the VAE
- an I2V mask marks those overlap frames as conditioned

The result per segment is:

- previous-segment conditioning latent package: `(B, 20, 90, 160, 20)`

Then the pipeline concatenates:

- reference image latent package: `(B, 1, 90, 160, 20)`
- previous-segment latent package: `(B, 20, 90, 160, 20)`

Final `reference_latents` per segment:

- `(B, 21, 90, 160, 20)`

## 4.6 Noisy Latents

Each segment samples fresh noisy latents:

- `seg_latents`: `(B, 21, 90, 160, 16)`

The `+1` is the dedicated reference slot.

## 4.7 Transformer Input Assembly

The single denoising step builds:

- `latents`: `(B, 21, 90, 160, 16)`
- `reference_latents`: `(B, 21, 90, 160, 20)`

These are concatenated on channels:

- `latent_model_input`: `(B, 21, 90, 160, 36)`

Then transposed for the transformer:

- `(B, 36, 21, 90, 160)`

Why `36`:

- `16` noisy latent channels
- `16` reference / prev-segment latent channels
- `4` I2V mask channels

## 5. Wan Animate Transformer: Layer by Layer

## 5.1 Rotary Embedding

`WanRotaryPosEmbed` computes spatial-temporal rotary features from the latent grid.

Input grid:

- `(B, 21, 90, 160, 36)` after channel-last transpose inside `_embed_inputs`

It produces rotary embeddings used by self-attention.

## 5.2 Patch Embedding

`patch_embedding` is a 3D conv with:

- kernel size `(1, 2, 2)`
- stride `(1, 2, 2)`

So it:

- keeps temporal length `21`
- halves latent height `90 -> 45`
- halves latent width `160 -> 80`
- projects channels `36 -> inner_dim`

For Wan Animate 27B:

- `inner_dim = num_attention_heads * attention_head_dim = 40 * 128 = 5120`

Patch-embedded hidden grid:

- `(B, 21, 45, 80, 5120)`

Flattened token sequence:

- token count = `21 * 45 * 80 = 75,600`
- hidden states = `(B, 75600, 5120)`

## 5.3 Pose Patch Embedding

Pose latents go through a second patch conv:

- input: `(B, 16, 20, 90, 160)` channel-first before transpose
- patch output: `(B, 20, 45, 80, 5120)`

The code then prepends one all-zero temporal slice so pose tokens align with the transformer's `21` temporal slots:

- padded pose grid: `(B, 21, 45, 80, 5120)`

Then it adds pose features into the main hidden grid before flattening.

## 5.4 Time / Text / Image Conditioning

`WanTimeTextImageEmbedding` produces:

- `temb`: global timestep conditioning
- `timestep_proj`: `(B, 6, 5120)` AdaLN modulation input
- text tokens projected to model width
- optional image tokens projected to model width

If image embeddings are present:

- image tokens are concatenated in front of text tokens

These become `encoder_hidden_states` for every transformer block.

## 5.5 Motion Encoder

The face video segment is not VAE-encoded for this path.

Instead it goes through:

- `WanAnimateMotionEncoder`
- `WanAnimateFaceEncoder`

This produces `motion_vec`, which is used by face-adapter injections.

The motion encoder runs on the raw face crop sequence resized to the model's configured `motion_encoder_size`.

## 5.6 Transformer Block

Each `WanTransformerBlock` applies:

1. AdaLN modulation from timestep projection
2. self-attention
3. cross-attention to text/image conditioning
4. feed-forward

Residual gates are applied around the self-attention and FFN paths.

## 5.7 Self-Attention in `ulysses_fsdp`

This is the key part.

The block calls `FlaxWanAttention`, which reaches `_tpu_flash_attention` in `attention_flax.py`.

In `ulysses_fsdp` mode:

- sequence logical axes resolve to `fsdp`
- q / k / v are sequence-sharded on the single `fsdp` axis
- before splash attention, the code performs `all_to_all`
- this swaps sequence sharding for head sharding
- local splash attention then runs on full local sequence and local head shard
- after attention, another `all_to_all` restores sequence sharding

So the implementation now matches this attention mental model:

```text
(b, s / n, h, d)
  -> all_to_all(fsdp)
  -> (b, s, h / n, d)
  -> local splash attention
  -> all_to_all(fsdp)
  -> (b, s / n, h, d)
```

Two implementation details matter:

- head count is padded internally when `num_heads % num_fsdp_shards != 0`
- sequence is padded before `shard_map` when the local length would otherwise fail divisibility

That second fix is important for short or awkward sequence lengths.

## 5.8 Cross-Attention

Cross-attention uses the same attention kernel plumbing.

The query comes from video tokens.
The key / value come from the projected text + image conditioning tokens.

Because conditioning sequence lengths are often not divisible by shard count, the current implementation pads the sequence before the `shard_map` boundary when needed.

## 5.9 Feed-Forward

The FFN stays in token space:

- input: `(B, 75600, 5120)`
- expansion to `ffn_dim`
- activation
- projection back to `5120`

Current state:

- parameter axes are still annotated for `fsdp`
- activations stay logically sequence-sharded
- JAX / XLA decides the exact reshards / collectives needed

So this is close to the intended single-axis model, but still compiler-managed rather than a fully explicit custom FFN parallel schedule.

## 5.10 Face Adapter Injection

After certain blocks, the model injects face motion conditioning through `WanAnimateFaceBlockCrossAttention`.

By default this happens every `inject_face_latents_blocks = 5` blocks.

For a 40-layer model, that means face adapter injections at block indices:

- `0`
- `5`
- `10`
- `15`
- `20`
- `25`
- `30`
- `35`

This is where identity / face-performance information is reintroduced through the transformer depth.

## 5.11 Final Projection

After the last block:

- `norm_out`
- scale / shift modulation
- `proj_out`

This projects token states back to latent-patch outputs.

The transformer output is reshaped back to:

- `(B, 16, 21, 90, 160)` channel-first

Then the pipeline transposes it to:

- `(B, 21, 90, 160, 16)`

That is the predicted noise used by the scheduler.

## 6. Denoising and Decoding

Each segment runs:

- `40` scheduler steps

At the end of a segment:

- the reference slot is dropped: `seg_latents[:, 1:]`
- decoded latent shape: `(B, 20, 90, 160, 16)`
- decoded pixel frames: `(B, 3, 77, 720, 1280)`

For segment 1:

- the first `5` decoded frames are dropped because they were only overlap conditioning

Final assembly:

- segment 0 contributes `77` frames
- segment 1 contributes `72` new frames
- total before trim = `149`
- final trim back to requested length = `121`

## 7. What Is True Today

The following statements are true for the current code:

- attention uses the `fsdp` axis as the sequence-parallel axis in `ulysses_fsdp`
- attention performs `all_to_all -> local splash attention -> all_to_all back`
- q / k / v are not left replicated for the local splash kernel path
- Wan and Wan Animate configs keep model sharding on `fsdp` instead of `context`
- Wan VAE conv output-channel sharding prefers `fsdp`

The following is still only partially true:

- "zero extra communication for linear layers"

Reason:

- MaxDiffusion does not currently implement explicit custom row-parallel / col-parallel FFN kernels for this path
- linear / FFN communication remains whatever XLA inserts to satisfy the mixed activation / parameter layouts

So the correct summary is:

- attention now follows the intended single-axis Ulysses scheme
- linear and FFN layers rely on compiler-managed resharding on that same axis

## 8. Command To Run

This command uses the existing local sample assets from `assets/wan_animate/` and runs Wan Animate with:

- `720x1280`
- `121` frames
- `5` previous conditioning frames
- `40` denoising steps
- `ulysses_fsdp`

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
python src/maxdiffusion/generate_wan_animate.py \
src/maxdiffusion/configs/base_wan_animate_27b.yml \
attention=ulysses_fsdp \
ici_data_parallelism=1 \
ici_fsdp_parallelism=-1 \
ici_context_parallelism=1 \
height=720 \
width=1280 \
num_frames=121 \
segment_frame_length=77 \
prev_segment_conditioning_frames=5 \
num_inference_steps=40 \
run_name=wan-animate-ulysses-fsdp-720p-121f \
output_dir=/tmp/wan_animate_ulysses_fsdp_out
```

If you want to override the local sample assets too:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
python src/maxdiffusion/generate_wan_animate.py \
src/maxdiffusion/configs/base_wan_animate_27b.yml \
attention=ulysses_fsdp \
ici_data_parallelism=1 \
ici_fsdp_parallelism=-1 \
ici_context_parallelism=1 \
height=720 \
width=1280 \
num_frames=121 \
segment_frame_length=77 \
prev_segment_conditioning_frames=5 \
num_inference_steps=40 \
reference_image_path=/path/to/ref.png \
pose_video_path=/path/to/pose.mp4 \
face_video_path=/path/to/face.mp4 \
run_name=wan-animate-ulysses-fsdp-720p-121f \
output_dir=/tmp/wan_animate_ulysses_fsdp_out
```

The output video is written under:

- `/tmp/wan_animate_ulysses_fsdp_out/animate_wan_output_<seed>_0.mp4`
