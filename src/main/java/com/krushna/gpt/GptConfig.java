package com.krushna.gpt;

/**
 * GptConfig – central place for all model hyper-parameters and dataset limits.
 *
 * <p>Mirrors the {@code #define} constants at the top of {@code microgpt.c}.
 * Change these values to experiment with different model sizes.
 *
 * <ul>
 *   <li>{@link #N_EMBD}    – embedding / hidden dimension</li>
 *   <li>{@link #N_HEAD}    – number of attention heads</li>
 *   <li>{@link #N_LAYER}   – number of transformer blocks</li>
 *   <li>{@link #BLOCK_SIZE}– context window (sequence length)</li>
 * </ul>
 */
public final class GptConfig {

    // -----------------------------------------------------------------------
    //  Model architecture
    // -----------------------------------------------------------------------

    /** Embedding dimension (must be divisible by N_HEAD). */
    public static final int N_EMBD = 32;

    /** Number of attention heads. */
    public static final int N_HEAD = 4;

    /** Number of transformer layers (blocks). */
    public static final int N_LAYER = 1;

    /** Context window – maximum number of tokens the model sees at once. */
    public static final int BLOCK_SIZE = 8;

    /** Per-head dimension = N_EMBD / N_HEAD. */
    public static final int HEAD_DIM = N_EMBD / N_HEAD;

    /** Feed-forward hidden dimension = 4 * N_EMBD. */
    public static final int MLP_DIM = 4 * N_EMBD;

    // -----------------------------------------------------------------------
    //  Dataset limits
    // -----------------------------------------------------------------------

    /** Maximum number of documents (lines) loaded from the training file. */
    public static final int MAX_DOCS = 85_000;

    /** Maximum characters per document (line). */
    public static final int MAX_DOC_LEN = 512;

    /** Maximum distinct characters in the vocabulary (before BOS token). */
    public static final int MAX_CHARS = 128;

    // -----------------------------------------------------------------------
    //  Training hyper-parameters (defaults – can be overridden at runtime)
    // -----------------------------------------------------------------------

    /** Default initial learning rate. */
    public static final float DEFAULT_LR = 1e-2f;

    /** Adam β₁ (first moment decay). */
    public static final float ADAM_BETA1 = 0.9f;

    /** Adam β₂ (second moment decay). */
    public static final float ADAM_BETA2 = 0.95f;

    /** Adam numerical stability ε. */
    public static final float ADAM_EPS = 1e-8f;

    /** Default number of training steps. */
    public static final int DEFAULT_NUM_STEPS = 5000;

    // -----------------------------------------------------------------------
    //  Inference defaults
    // -----------------------------------------------------------------------

    /** Softmax temperature used during sampling (lower = more deterministic). */
    public static final float DEFAULT_TEMPERATURE = 0.5f;

    /** Number of sample sequences generated after training. */
    public static final int DEFAULT_NUM_SAMPLES = 20;

    // -----------------------------------------------------------------------
    //  Initialisation standard deviations (match microgpt.c)
    // -----------------------------------------------------------------------

    /** Std-dev for most weight matrices. */
    public static final float INIT_STD_NORMAL = 0.02f;

    /** Std-dev for output projection weights (zero-init keeps residual stable). */
    public static final float INIT_STD_ZERO = 0.0f;

    // prevent instantiation
    private GptConfig() {}
}
