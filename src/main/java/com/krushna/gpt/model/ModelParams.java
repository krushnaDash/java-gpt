package com.krushna.gpt.model;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.util.MathUtils;
import com.krushna.gpt.util.RandomUtils;

/**
 * ModelParams – holds all trainable parameters and their gradient buffers.
 *
 * <p>Mirrors the global parameter arrays in {@code microgpt.c}:
 * <ul>
 *   <li>{@code wte}      – token embedding table [vocabSize × N_EMBD]</li>
 *   <li>{@code wpe}      – positional embedding  [BLOCK_SIZE × N_EMBD]</li>
 *   <li>{@code lmHead}   – language-model head   [vocabSize × N_EMBD]</li>
 *   <li>{@code attnWq/k/v/o[layer]} – attention projections [N_EMBD × N_EMBD]</li>
 *   <li>{@code mlpFc1[layer]} – MLP up-projection   [MLP_DIM × N_EMBD]</li>
 *   <li>{@code mlpFc2[layer]} – MLP down-projection [N_EMBD × MLP_DIM]</li>
 * </ul>
 *
 * <p>Each parameter array has a companion gradient buffer (prefixed with
 * {@code d}) and Adam moment buffers ({@code m_} and {@code v_}).
 *
 * <p>Call {@link #init(int, RandomUtils)} once after the vocabulary size is
 * known to allocate and randomly initialise everything.
 */
public class ModelParams {

    // -----------------------------------------------------------------------
    //  Embedding & output head
    // -----------------------------------------------------------------------

    /** Token embedding table: shape [vocabSize × N_EMBD]. */
    public float[] wte, dWte;
    // flat one dimensional array beteer from a performance point of view as
    // compare to 2D array, the flat logic is ROW*COLS + col
    // look up for a token can be done like wte[token_id*32, token_id*32+32]
    // start index token_id*32, end index token_id*32+32

    /** Positional embedding: shape [BLOCK_SIZE × N_EMBD]. */
    public float[] wpe, dWpe;

    /** Language-model head (weight-tied with wte): shape [vocabSize × N_EMBD]. */
    public float[] lmHead, dLmHead;

    // -----------------------------------------------------------------------
    //  Per-layer attention weights
    // -----------------------------------------------------------------------

    /** Query projection per layer: shape [N_EMBD × N_EMBD]. */
    public float[][] attnWq, dAttnWq;

    /** Key projection per layer. */
    public float[][] attnWk, dAttnWk;

    /** Value projection per layer. */
    public float[][] attnWv, dAttnWv;

    /** Output projection per layer. */
    public float[][] attnWo, dAttnWo;

    // -----------------------------------------------------------------------
    //  Per-layer MLP weights
    // -----------------------------------------------------------------------

    /**
     * MLP first (up-projection) weight per layer: shape [MLP_DIM × N_EMBD].
     */
    public float[][] mlpFc1, dMlpFc1;

    /**
     * MLP second (down-projection) weight per layer: shape [N_EMBD × MLP_DIM].
     * Note: stored as [MLP_DIM × N_EMBD] in the C code (transpose layout) –
     * here we store it as [N_EMBD × MLP_DIM] for clarity.
     */
    public float[][] mlpFc2, dMlpFc2;

    // -----------------------------------------------------------------------
    //  Adam moment buffers (one per parameter tensor)
    // -----------------------------------------------------------------------

    public float[] mWte, vWte;
    public float[] mWpe, vWpe;
    public float[] mLmHead, vLmHead;
    public float[][] mWq, vWq;
    public float[][] mWk, vWk;
    public float[][] mWv, vWv;
    public float[][] mWo, vWo;
    public float[][] mFc1, vFc1;
    public float[][] mFc2, vFc2;

    // -----------------------------------------------------------------------
    //  Metadata
    // -----------------------------------------------------------------------

    /** Total number of trainable scalar parameters (informational). */
    public int numParams;

    /** Vocabulary size – stored so serialisation knows the embedding dims. */
    public int vocabSize;

    // -----------------------------------------------------------------------
    //  Initialisation
    // -----------------------------------------------------------------------

    /**
     * Allocates and randomly initialises all parameter + gradient arrays.
     *
     * <p>Mirrors {@code init_params()} in microgpt.c.
     *
     * @param vocabSize number of tokens in the vocabulary (set after tokeniser build)
     * @param rng       random number generator for weight initialisation
     */
    public void init(int vocabSize, RandomUtils rng) {
        this.vocabSize = vocabSize;
        numParams = 0;

        int embSize = vocabSize * GptConfig.N_EMBD;         // wte, lmHead
        int posSize = GptConfig.BLOCK_SIZE * GptConfig.N_EMBD; // wpe
        int attSize = GptConfig.N_EMBD * GptConfig.N_EMBD;   // Q/K/V/O
        int mlpSize = GptConfig.MLP_DIM * GptConfig.N_EMBD;   // fc1, fc2

        // ----- Embeddings -----
        // wte : word token embeddings
        wte     = makeParam(embSize, GptConfig.INIT_STD_NORMAL, rng);
        dWte    = MathUtils.zeros(embSize);
        mWte    = MathUtils.zeros(embSize);
        vWte    = MathUtils.zeros(embSize);

        wpe     = makeParam(posSize, GptConfig.INIT_STD_NORMAL, rng);
        dWpe    = MathUtils.zeros(posSize);
        mWpe    = MathUtils.zeros(posSize);
        vWpe    = MathUtils.zeros(posSize);

        lmHead  = makeParam(embSize, GptConfig.INIT_STD_NORMAL, rng);
        dLmHead = MathUtils.zeros(embSize);
        mLmHead = MathUtils.zeros(embSize);
        vLmHead = MathUtils.zeros(embSize);

        // ----- Per-layer arrays -----
        attnWq  = new float[GptConfig.N_LAYER][];
        dAttnWq = new float[GptConfig.N_LAYER][];
        attnWk  = new float[GptConfig.N_LAYER][];
        dAttnWk = new float[GptConfig.N_LAYER][];
        attnWv  = new float[GptConfig.N_LAYER][];
        dAttnWv = new float[GptConfig.N_LAYER][];
        attnWo  = new float[GptConfig.N_LAYER][];
        dAttnWo = new float[GptConfig.N_LAYER][];
        mlpFc1  = new float[GptConfig.N_LAYER][];
        dMlpFc1 = new float[GptConfig.N_LAYER][];
        mlpFc2  = new float[GptConfig.N_LAYER][];
        dMlpFc2 = new float[GptConfig.N_LAYER][];

        mWq  = new float[GptConfig.N_LAYER][];
        vWq  = new float[GptConfig.N_LAYER][];
        mWk  = new float[GptConfig.N_LAYER][];
        vWk  = new float[GptConfig.N_LAYER][];
        mWv  = new float[GptConfig.N_LAYER][];
        vWv  = new float[GptConfig.N_LAYER][];
        mWo  = new float[GptConfig.N_LAYER][];
        vWo  = new float[GptConfig.N_LAYER][];
        mFc1 = new float[GptConfig.N_LAYER][];
        vFc1 = new float[GptConfig.N_LAYER][];
        mFc2 = new float[GptConfig.N_LAYER][];
        vFc2 = new float[GptConfig.N_LAYER][];

        for (int li = 0; li < GptConfig.N_LAYER; li++) {
            attnWq[li]  = makeParam(attSize, GptConfig.INIT_STD_NORMAL, rng);
            dAttnWq[li] = MathUtils.zeros(attSize);
            mWq[li]     = MathUtils.zeros(attSize);
            vWq[li]     = MathUtils.zeros(attSize);

            attnWk[li]  = makeParam(attSize, GptConfig.INIT_STD_NORMAL, rng);
            dAttnWk[li] = MathUtils.zeros(attSize);
            mWk[li]     = MathUtils.zeros(attSize);
            vWk[li]     = MathUtils.zeros(attSize);

            attnWv[li]  = makeParam(attSize, GptConfig.INIT_STD_NORMAL, rng);
            dAttnWv[li] = MathUtils.zeros(attSize);
            mWv[li]     = MathUtils.zeros(attSize);
            vWv[li]     = MathUtils.zeros(attSize);

            // Output projection zero-initialised for stable residual (same as C)
            attnWo[li]  = makeParam(attSize, GptConfig.INIT_STD_ZERO, rng);
            dAttnWo[li] = MathUtils.zeros(attSize);
            mWo[li]     = MathUtils.zeros(attSize);
            vWo[li]     = MathUtils.zeros(attSize);

            mlpFc1[li]  = makeParam(mlpSize, GptConfig.INIT_STD_NORMAL, rng);
            dMlpFc1[li] = MathUtils.zeros(mlpSize);
            mFc1[li]    = MathUtils.zeros(mlpSize);
            vFc1[li]    = MathUtils.zeros(mlpSize);

            // Second MLP layer zero-initialised (same as C)
            mlpFc2[li]  = makeParam(mlpSize, GptConfig.INIT_STD_ZERO, rng);
            dMlpFc2[li] = MathUtils.zeros(mlpSize);
            mFc2[li]    = MathUtils.zeros(mlpSize);
            vFc2[li]    = MathUtils.zeros(mlpSize);
        }

        System.out.printf("num params: %d%n", numParams);
    }

    // -----------------------------------------------------------------------
    //  Helper
    // -----------------------------------------------------------------------

    /**
     * Allocates a parameter array of {@code size} floats and fills it with
     * Gaussian-distributed random values (mean = 0, std = {@code std}).
     * If {@code std == 0} the array is zero-initialised.
     */
    private float[] makeParam(int size, float std, RandomUtils rng) {
        float[] p = new float[size];
        if (std > 0f) {
            for (int i = 0; i < size; i++) {
                p[i] = rng.nextGaussian(0f, std);
            }
        }
        numParams += size;
        return p;
    }

    // -----------------------------------------------------------------------
    //  Gradient reset (called before each backward pass)
    // -----------------------------------------------------------------------

    /**
     * Zeros all gradient buffers.  Must be called before each backward pass
     * to prevent gradient accumulation across steps.
     */
    public void zeroGradients() {
        MathUtils.zero(dWte);
        MathUtils.zero(dWpe);
        MathUtils.zero(dLmHead);
        for (int li = 0; li < GptConfig.N_LAYER; li++) {
            MathUtils.zero(dAttnWq[li]);
            MathUtils.zero(dAttnWk[li]);
            MathUtils.zero(dAttnWv[li]);
            MathUtils.zero(dAttnWo[li]);
            MathUtils.zero(dMlpFc1[li]);
            MathUtils.zero(dMlpFc2[li]);
        }
    }
}
