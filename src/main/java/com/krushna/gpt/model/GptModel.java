package com.krushna.gpt.model;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.util.MathUtils;

import java.util.Arrays;

/**
 * GptModel – the GPT forward and backward passes.
 *
 * <p>This class is a direct Java translation of the core computational graph
 * in {@code microgpt.c}.  It operates on the weight arrays stored in a
 * {@link ModelParams} instance and writes intermediate activations into a
 * pre-allocated array of {@link PosActivations} objects.
 *
 * <p>Architecture per token position:
 * <ol>
 *   <li>Token + positional embedding lookup → {@code x}</li>
 *   <li>RMSNorm of {@code x}</li>
 *   <li>For each transformer layer:
 *     <ol>
 *       <li>RMSNorm → multi-head causal self-attention (with KV-cache) → residual add</li>
 *       <li>RMSNorm → squared-ReLU MLP → residual add</li>
 *     </ol>
 *   </li>
 *   <li>Linear language-model head → logits over vocabulary</li>
 * </ol>
 */
public class GptModel {

    // -----------------------------------------------------------------------
    //  Saved activations (allocated once, reused every step)
    // -----------------------------------------------------------------------

    /** Per-position saved activations for the backward pass. */
    private final PosActivations[] saved;

    // -----------------------------------------------------------------------
    //  KV-cache and gradient accumulators (cleared at the start of each step)
    // -----------------------------------------------------------------------

    /** Key vectors cached per layer per position: [N_LAYER][BLOCK_SIZE][N_EMBD]. */
    private final float[][][] kvKeys;

    /** Value vectors cached per layer per position: [N_LAYER][BLOCK_SIZE][N_EMBD]. */
    private final float[][][] kvVals;

    /**
     * Accumulated key gradients: [N_LAYER][BLOCK_SIZE][N_EMBD].
     * Because every future query position contributes a gradient to each
     * past key, we accumulate before calling the weight backward.
     */
    private final float[][][] dkAccum;

    /** Accumulated value gradients: [N_LAYER][BLOCK_SIZE][N_EMBD]. */
    private final float[][][] dvAccum;

    // -----------------------------------------------------------------------
    //  Constructor
    // -----------------------------------------------------------------------

    /**
     * Allocates all activation and cache buffers.
     * Call once after the model parameters have been initialised.
     */
    public GptModel() {
        saved = new PosActivations[GptConfig.BLOCK_SIZE];
        for (int i = 0; i < GptConfig.BLOCK_SIZE; i++) {
            saved[i] = new PosActivations();
        }
        kvKeys  = new float[GptConfig.N_LAYER][GptConfig.BLOCK_SIZE][GptConfig.N_EMBD];
        kvVals  = new float[GptConfig.N_LAYER][GptConfig.BLOCK_SIZE][GptConfig.N_EMBD];
        dkAccum = new float[GptConfig.N_LAYER][GptConfig.BLOCK_SIZE][GptConfig.N_EMBD];
        dvAccum = new float[GptConfig.N_LAYER][GptConfig.BLOCK_SIZE][GptConfig.N_EMBD];
    }

    // -----------------------------------------------------------------------
    //  Forward pass (one token per call)
    // -----------------------------------------------------------------------

    /**
     * Runs the forward pass for a single token at the given sequence position.
     *
     * <p>Mirrors {@code gpt_forward()} in microgpt.c.
     *
     * @param p         the model parameters
     * @param tokenId   the input token id
     * @param posId     the position in the sequence (0-based)
     * @param logitsOut output buffer for raw logits; must have length ≥ vocabSize
     * @param act       the {@link PosActivations} instance to fill with saved values
     */
    public void forward(ModelParams p, int tokenId, int posId,
                        float[] logitsOut, PosActivations act) {

        int E = GptConfig.N_EMBD;
        int V = p.vocabSize;

        // --- 1. Embedding lookup: x = wte[token] + wpe[pos] ---
        float[] x = new float[E];
        for (int i = 0; i < E; i++) {
            x[i] = p.wte[tokenId * E + i] + p.wpe[posId * E + i];
        }
        MathUtils.copy(x, act.xEmbed);

        // --- 2. Initial RMSNorm ---
        act.rmsScaleInit = MathUtils.rmsnormFwd(x, E, x);

        // --- 3. Transformer layers ---
        float[] tmp    = new float[Math.max(GptConfig.MLP_DIM, E)];
        float[] xn     = new float[E];
        float[] q      = new float[E];
        float[] k      = new float[E];
        float[] v      = new float[E];
        float[] ao     = new float[E];
        float[] h1     = new float[GptConfig.MLP_DIM];
        float[] h2     = new float[GptConfig.MLP_DIM];

        float scale = 1.0f / (float) Math.sqrt((double) E / GptConfig.N_HEAD);

        for (int li = 0; li < GptConfig.N_LAYER; li++) {
            MathUtils.copy(x, act.xIn[li]);

            // ----- Attention sub-layer -----
            act.rmsScaleAttn[li] = MathUtils.rmsnormFwd(x, E, xn);
            MathUtils.copy(xn, act.xnAttn[li]);

            MathUtils.linearFwd(xn, p.attnWq[li], E, E, q);
            MathUtils.linearFwd(xn, p.attnWk[li], E, E, k);
            MathUtils.linearFwd(xn, p.attnWv[li], E, E, v);
            MathUtils.copy(q, act.q[li]);

            // Cache keys and values for this position
            System.arraycopy(k, 0, kvKeys[li][posId], 0, E);
            System.arraycopy(v, 0, kvVals[li][posId], 0, E);

            int seqLen = posId + 1;

            // Multi-head causal self-attention (attend to all past + current pos)
            for (int h = 0; h < GptConfig.N_HEAD; h++) {
                int hs = h * GptConfig.HEAD_DIM;
                float[] al = new float[GptConfig.BLOCK_SIZE];

                // Compute attention logits
                for (int tt = 0; tt < seqLen; tt++) {
                    float dot = 0f;
                    for (int j = 0; j < GptConfig.HEAD_DIM; j++) {
                        dot += q[hs + j] * kvKeys[li][tt][hs + j];
                    }
                    al[tt] = dot * scale;
                }

                // Softmax over the seqLen positions
                float mx = al[0];
                for (int tt = 1; tt < seqLen; tt++) if (al[tt] > mx) mx = al[tt];
                float sm = 0f;
                for (int tt = 0; tt < seqLen; tt++) { al[tt] = (float) Math.exp(al[tt] - mx); sm += al[tt]; }
                float inv = 1.0f / sm;
                for (int tt = 0; tt < seqLen; tt++) al[tt] *= inv;

                // Save attention weights for backward
                System.arraycopy(al, 0, act.aw[li][h], 0, seqLen);

                // Weighted sum of values
                for (int j = 0; j < GptConfig.HEAD_DIM; j++) {
                    float s = 0f;
                    for (int tt = 0; tt < seqLen; tt++) {
                        s += al[tt] * kvVals[li][tt][hs + j];
                    }
                    ao[hs + j] = s;
                }
            }
            MathUtils.copy(ao, act.attnOut[li]);

            // O-projection + residual
            MathUtils.linearFwd(ao, p.attnWo[li], E, E, tmp);
            for (int i = 0; i < E; i++) x[i] = tmp[i] + act.xIn[li][i];
            MathUtils.copy(x, act.xMid[li]);

            // ----- MLP sub-layer -----
            act.rmsScaleMlp[li] = MathUtils.rmsnormFwd(x, E, xn);
            MathUtils.copy(xn, act.xnMlp[li]);

            MathUtils.linearFwd(xn, p.mlpFc1[li], GptConfig.MLP_DIM, E, h1);
            System.arraycopy(h1, 0, act.mlpPre[li], 0, GptConfig.MLP_DIM);

            MathUtils.sqReluFwd(h1, GptConfig.MLP_DIM, h2);
            System.arraycopy(h2, 0, act.mlpPost[li], 0, GptConfig.MLP_DIM);

            // Note: mlpFc2 is stored as [N_EMBD × MLP_DIM]; output dimension = N_EMBD
            MathUtils.linearFwd(h2, p.mlpFc2[li], E, GptConfig.MLP_DIM, tmp);
            for (int i = 0; i < E; i++) x[i] = tmp[i] + act.xMid[li][i];
        }

        MathUtils.copy(x, act.xOut);

        // --- 4. Language-model head ---
        MathUtils.linearFwd(x, p.lmHead, V, E, logitsOut);
    }

    // -----------------------------------------------------------------------
    //  Full sequence forward (training)
    // -----------------------------------------------------------------------

    /**
     * Runs the forward pass for all {@code n} positions in a sequence and
     * computes the average cross-entropy loss.
     *
     * @param p       model parameters
     * @param tokens  input token ids (length ≥ n+1; position {@code i} uses {@code tokens[i]})
     * @param targets target token ids at each position (length ≥ n)
     * @param n       number of positions to process (≤ BLOCK_SIZE)
     * @return average cross-entropy loss over all positions
     */
    public float forwardSequence(ModelParams p, int[] tokens, int[] targets, int n) {
        // Clear the KV cache for a fresh forward pass
        for (float[][] layer : kvKeys) for (float[] row : layer) Arrays.fill(row, 0f);
        for (float[][] layer : kvVals) for (float[] row : layer) Arrays.fill(row, 0f);

        float[] logits = new float[p.vocabSize];
        float totalLoss = 0f;

        for (int pos = 0; pos < n; pos++) {
            forward(p, tokens[pos], pos, logits, saved[pos]);
            MathUtils.softmaxFwd(logits, p.vocabSize, saved[pos].probs);
            totalLoss += -(float) Math.log(saved[pos].probs[targets[pos]] + 1e-30f);
        }
        return totalLoss / n;
    }

    // -----------------------------------------------------------------------
    //  Backward pass (all positions)
    // -----------------------------------------------------------------------

    /**
     * Runs the backward pass over all {@code n} positions after a forward pass.
     *
     * <p>Gradients are <em>accumulated</em> into the {@code d*} arrays of
     * {@link ModelParams}; call {@link ModelParams#zeroGradients()} before this
     * method if you want fresh gradients for a new step.
     *
     * <p>Mirrors {@code gpt_backward()} in microgpt.c.
     *
     * @param p       model parameters (gradients accumulated here)
     * @param tokens  input tokens used in the forward pass
     * @param targets target tokens used in the forward pass
     * @param n       number of positions (must match the forward call)
     */
    public void backward(ModelParams p, int[] tokens, int[] targets, int n) {
        // Clear accumulated key/value gradients
        for (float[][] layer : dkAccum) for (float[] row : layer) Arrays.fill(row, 0f);
        for (float[][] layer : dvAccum) for (float[] row : layer) Arrays.fill(row, 0f);

        int E = GptConfig.N_EMBD;
        float invN = 1.0f / n;
        float scaleFwd = 1.0f / (float) Math.sqrt((double) E / GptConfig.N_HEAD);

        // Iterate positions in reverse (standard backprop through time)
        for (int pos = n - 1; pos >= 0; pos--) {
            PosActivations act = saved[pos];
            int seqLen = pos + 1;

            // ---- Softmax cross-entropy gradient: dl[i] = (prob[i] - 1{i==target}) / n ----
            float[] dl = new float[p.vocabSize];
            for (int i = 0; i < p.vocabSize; i++) {
                dl[i] = (act.probs[i] - (i == targets[pos] ? 1.0f : 0.0f)) * invN;
            }

            // ---- LM head backward ----
            float[] dx = new float[E];
            MathUtils.linearBwdX(p.lmHead, dl, p.vocabSize, E, dx);
            MathUtils.linearBwdW(act.xOut, dl, p.vocabSize, E, p.dLmHead);

            // ---- Transformer layers (in reverse) ----
            float[] dH2   = new float[GptConfig.MLP_DIM];
            float[] dH1   = new float[GptConfig.MLP_DIM];
            float[] dXnMlp = new float[E];
            float[] dXMid  = new float[E];
            float[] dAo    = new float[E];
            float[] dQ     = new float[E];
            float[] dXn    = new float[E];
            float[] dXIn   = new float[E];
            float[] dEmbed = new float[E];

            for (int li = GptConfig.N_LAYER - 1; li >= 0; li--) {

                // ===== MLP backward =====
                Arrays.fill(dH2, 0f);
                MathUtils.linearBwdX(p.mlpFc2[li], dx, E, GptConfig.MLP_DIM, dH2);
                MathUtils.linearBwdW(act.mlpPost[li], dx, E, GptConfig.MLP_DIM, p.dMlpFc2[li]);

                MathUtils.sqReluBwd(act.mlpPre[li], dH2, GptConfig.MLP_DIM, dH1);

                Arrays.fill(dXnMlp, 0f);
                MathUtils.linearBwdX(p.mlpFc1[li], dH1, GptConfig.MLP_DIM, E, dXnMlp);
                MathUtils.linearBwdW(act.xnMlp[li], dH1, GptConfig.MLP_DIM, E, p.dMlpFc1[li]);

                Arrays.fill(dXMid, 0f);
                MathUtils.rmsnormBwd(act.xMid[li], act.rmsScaleMlp[li], dXnMlp, E, dXMid);
                for (int i = 0; i < E; i++) dx[i] += dXMid[i];

                // ===== Attention backward =====
                Arrays.fill(dAo, 0f);
                MathUtils.linearBwdX(p.attnWo[li], dx, E, E, dAo);
                MathUtils.linearBwdW(act.attnOut[li], dx, E, E, p.dAttnWo[li]);

                Arrays.fill(dQ, 0f);

                for (int h = 0; h < GptConfig.N_HEAD; h++) {
                    int hs = h * GptConfig.HEAD_DIM;
                    float[] dAw = new float[GptConfig.BLOCK_SIZE];

                    // gradient through weighted-sum of values
                    for (int j = 0; j < GptConfig.HEAD_DIM; j++) {
                        for (int tt = 0; tt < seqLen; tt++) {
                            dAw[tt] += dAo[hs + j] * kvVals[li][tt][hs + j];
                            dvAccum[li][tt][hs + j] += act.aw[li][h][tt] * dAo[hs + j];
                        }
                    }

                    // gradient through softmax: d_al = aw * (d_aw - dot(d_aw, aw))
                    float dot = 0f;
                    for (int tt = 0; tt < seqLen; tt++) dot += dAw[tt] * act.aw[li][h][tt];
                    float[] dAl = new float[seqLen];
                    for (int tt = 0; tt < seqLen; tt++) {
                        dAl[tt] = act.aw[li][h][tt] * (dAw[tt] - dot);
                    }

                    // gradient into Q and K through dot-product attention
                    for (int tt = 0; tt < seqLen; tt++) {
                        for (int j = 0; j < GptConfig.HEAD_DIM; j++) {
                            dQ[hs + j] += dAl[tt] * kvKeys[li][tt][hs + j] * scaleFwd;
                            dkAccum[li][tt][hs + j] += dAl[tt] * act.q[li][hs + j] * scaleFwd;
                        }
                    }
                }

                // Backprop through Q, K, V projections
                Arrays.fill(dXn, 0f);
                MathUtils.linearBwdX(p.attnWq[li], dQ, E, E, dXn);
                MathUtils.linearBwdW(act.xnAttn[li], dQ, E, E, p.dAttnWq[li]);

                MathUtils.linearBwdX(p.attnWk[li], dkAccum[li][pos], E, E, dXn);
                MathUtils.linearBwdW(act.xnAttn[li], dkAccum[li][pos], E, E, p.dAttnWk[li]);

                MathUtils.linearBwdX(p.attnWv[li], dvAccum[li][pos], E, E, dXn);
                MathUtils.linearBwdW(act.xnAttn[li], dvAccum[li][pos], E, E, p.dAttnWv[li]);

                // Backprop through attention RMSNorm + residual
                Arrays.fill(dXIn, 0f);
                MathUtils.rmsnormBwd(act.xIn[li], act.rmsScaleAttn[li], dXn, E, dXIn);
                for (int i = 0; i < E; i++) dx[i] = dx[i] + dXIn[i];
            }

            // ---- Backprop through initial RMSNorm and into embeddings ----
            Arrays.fill(dEmbed, 0f);
            MathUtils.rmsnormBwd(act.xEmbed, act.rmsScaleInit, dx, E, dEmbed);

            int tok = tokens[pos];
            for (int i = 0; i < E; i++) {
                p.dWte[tok * E + i] += dEmbed[i];
                p.dWpe[pos * E + i] += dEmbed[i];
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Inference (single-step forward for generation)
    // -----------------------------------------------------------------------

    /**
     * Runs a single forward step for inference and returns the logits.
     *
     * <p>Callers are responsible for managing the KV cache (clear it between
     * generated sequences by calling {@link #clearKvCache()}).
     *
     * @param p       model parameters
     * @param tokenId the current input token
     * @param posId   the current position in the generated sequence
     * @param logits  output buffer (must have length ≥ vocabSize)
     */
    public void inferenceStep(ModelParams p, int tokenId, int posId, float[] logits) {
        PosActivations tmpAct = new PosActivations();
        forward(p, tokenId, posId, logits, tmpAct);
    }

    /**
     * Clears the KV cache.  Call between generated sequences during inference
     * to prevent a sequence from attending to the previous one.
     */
    public void clearKvCache() {
        for (float[][] layer : kvKeys) for (float[] row : layer) Arrays.fill(row, 0f);
        for (float[][] layer : kvVals) for (float[] row : layer) Arrays.fill(row, 0f);
    }
}
