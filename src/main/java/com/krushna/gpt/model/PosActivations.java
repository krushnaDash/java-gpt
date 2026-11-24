package com.krushna.gpt.model;

import com.krushna.gpt.GptConfig;

/**
 * PosActivations – stores intermediate activations for a single token position
 * so that the backward pass can reuse them without recomputing.
 *
 * <p>Mirrors the {@code PosActs} struct and the global {@code saved[]} /
 * {@code saved_probs[]} / {@code kv_*} arrays in {@code microgpt.c}.
 *
 * <p>One {@code PosActivations} instance is allocated per position in the
 * context window (up to {@link GptConfig#BLOCK_SIZE}); the GptModel class
 * holds an array of them.
 */
public class PosActivations {

    // -----------------------------------------------------------------------
    //  Initial embedding
    // -----------------------------------------------------------------------

    /** Combined token + positional embedding before the first RMSNorm. */
    public final float[] xEmbed = new float[GptConfig.N_EMBD];

    /** RMSNorm scale factor saved from the very first normalisation. */
    public float rmsScaleInit;

    // -----------------------------------------------------------------------
    //  Per-layer activations
    // -----------------------------------------------------------------------

    /** Input to each transformer block (before attention sub-layer RMSNorm). */
    public final float[][] xIn = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /** Normalised input fed into the attention projections. */
    public final float[][] xnAttn = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /** RMSNorm scale saved for the attention sub-layer of each block. */
    public final float[] rmsScaleAttn = new float[GptConfig.N_LAYER];

    /** Query vector after the Q-projection for each layer. */
    public final float[][] q = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /**
     * Softmax attention weights for each layer and head.
     * Shape: [N_LAYER][N_HEAD][BLOCK_SIZE].
     */
    public final float[][][] aw =
            new float[GptConfig.N_LAYER][GptConfig.N_HEAD][GptConfig.BLOCK_SIZE];

    /** Output of the attention aggregation (before the O-projection). */
    public final float[][] attnOut = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /**
     * Residual stream after the attention sub-layer
     * (input to MLP sub-layer RMSNorm).
     */
    public final float[][] xMid = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /** Normalised input fed into the MLP. */
    public final float[][] xnMlp = new float[GptConfig.N_LAYER][GptConfig.N_EMBD];

    /** RMSNorm scale saved for the MLP sub-layer of each block. */
    public final float[] rmsScaleMlp = new float[GptConfig.N_LAYER];

    /** Pre-activation values of the MLP hidden layer (before squared-ReLU). */
    public final float[][] mlpPre = new float[GptConfig.N_LAYER][GptConfig.MLP_DIM];

    /** Post-activation values of the MLP hidden layer (after squared-ReLU). */
    public final float[][] mlpPost = new float[GptConfig.N_LAYER][GptConfig.MLP_DIM];

    // -----------------------------------------------------------------------
    //  Final output embedding
    // -----------------------------------------------------------------------

    /** Residual stream value after the last transformer block. */
    public final float[] xOut = new float[GptConfig.N_EMBD];

    // -----------------------------------------------------------------------
    //  Softmax probabilities over the vocabulary
    // -----------------------------------------------------------------------

    /**
     * Softmax probability distribution produced at this position.
     * Length = vocabSize (set dynamically; allocated to MAX_CHARS+1 to be safe).
     */
    public final float[] probs = new float[GptConfig.MAX_CHARS + 1];
}
