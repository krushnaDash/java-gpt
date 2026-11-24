package com.krushna.gpt.training;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.model.ModelParams;

/**
 * AdamOptimizer – the Adam optimiser with cosine learning-rate schedule.
 *
 * <p>Mirrors {@code adam_update()} and the learning-rate annealing in
 * {@code main()} of {@code microgpt.c}.
 *
 * <p>Algorithm (for each parameter scalar {@code p}):
 * <pre>
 *   m = β₁ * m + (1 - β₁) * g
 *   v = β₂ * v + (1 - β₂) * g²
 *   p -= lr * (m / (1 - β₁^{t+1})) / (sqrt(v / (1 - β₂^{t+1})) + ε)
 *   g = 0   ← gradient reset after update
 * </pre>
 *
 * <p>Learning rate follows a cosine decay schedule:
 * {@code lr_t = lr * 0.5 * (1 + cos(π * step / numSteps))}
 */
public class AdamOptimizer {

    // -----------------------------------------------------------------------
    //  Hyper-parameters
    // -----------------------------------------------------------------------

    /** Initial (peak) learning rate. */
    private final float lr;

    /** β₁ – first moment (momentum) decay. */
    private final float beta1;

    /** β₂ – second moment (RMS) decay. */
    private final float beta2;

    /** ε – numerical stability constant. */
    private final float eps;

    /** Total number of training steps (used for cosine decay). */
    private final int numSteps;

    // -----------------------------------------------------------------------
    //  Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates an Adam optimiser with explicitly given hyper-parameters.
     *
     * @param lr       initial learning rate
     * @param beta1    first-moment decay (e.g. 0.9)
     * @param beta2    second-moment decay (e.g. 0.95)
     * @param eps      numerical stability (e.g. 1e-8)
     * @param numSteps total number of training steps for the cosine schedule
     */
    public AdamOptimizer(float lr, float beta1, float beta2, float eps, int numSteps) {
        this.lr       = lr;
        this.beta1    = beta1;
        this.beta2    = beta2;
        this.eps      = eps;
        this.numSteps = numSteps;
    }

    /**
     * Creates an Adam optimiser using default hyper-parameters from
     * {@link GptConfig}.
     */
    public AdamOptimizer() {
        this(GptConfig.DEFAULT_LR, GptConfig.ADAM_BETA1,
             GptConfig.ADAM_BETA2, GptConfig.ADAM_EPS,
             GptConfig.DEFAULT_NUM_STEPS);
    }

    // -----------------------------------------------------------------------
    //  Update
    // -----------------------------------------------------------------------

    /**
     * Applies one Adam update step to all model parameters.
     *
     * <p>Computes the cosine-annealed learning rate for this step, then
     * iterates over every parameter tensor.
     *
     * @param p    the model parameters (weights and gradients updated in-place)
     * @param step the current training step (0-based)
     */
    public void step(ModelParams p, int step) {
        // Cosine learning-rate schedule
        float lrT = lr * 0.5f * (1.0f + (float) Math.cos(Math.PI * step / numSteps));

        int E = GptConfig.N_EMBD;
        int V = p.vocabSize;

        int embSize = V * E;
        int posSize = GptConfig.BLOCK_SIZE * E;
        int attSize = E * E;
        int mlpSize = GptConfig.MLP_DIM * E;

        // ---- Embeddings and LM head ----
        adamUpdate(p.wte,    p.dWte,    p.mWte,    p.vWte,    embSize, lrT, step);
        adamUpdate(p.wpe,    p.dWpe,    p.mWpe,    p.vWpe,    posSize, lrT, step);
        adamUpdate(p.lmHead, p.dLmHead, p.mLmHead, p.vLmHead, embSize, lrT, step);

        // ---- Per-layer parameters ----
        for (int li = 0; li < GptConfig.N_LAYER; li++) {
            adamUpdate(p.attnWq[li], p.dAttnWq[li], p.mWq[li], p.vWq[li], attSize, lrT, step);
            adamUpdate(p.attnWk[li], p.dAttnWk[li], p.mWk[li], p.vWk[li], attSize, lrT, step);
            adamUpdate(p.attnWv[li], p.dAttnWv[li], p.mWv[li], p.vWv[li], attSize, lrT, step);
            adamUpdate(p.attnWo[li], p.dAttnWo[li], p.mWo[li], p.vWo[li], attSize, lrT, step);
            adamUpdate(p.mlpFc1[li], p.dMlpFc1[li], p.mFc1[li], p.vFc1[li], mlpSize, lrT, step);
            adamUpdate(p.mlpFc2[li], p.dMlpFc2[li], p.mFc2[li], p.vFc2[li], mlpSize, lrT, step);
        }
    }

    // -----------------------------------------------------------------------
    //  Per-tensor Adam update helper
    // -----------------------------------------------------------------------

    /**
     * Applies Adam to a single parameter array.
     *
     * <p>Bias-correction factors use {@code step + 1} (matching the C code
     * which passes {@code step} to {@code powf(b1, step+1)}).
     *
     * @param param the parameter values (updated in-place)
     * @param grad  the gradient values (zeroed after update)
     * @param m     first-moment buffer (updated in-place)
     * @param v     second-moment buffer (updated in-place)
     * @param size  number of elements
     * @param lrT   the current (annealed) learning rate
     * @param step  the current 0-based training step
     */
    private void adamUpdate(float[] param, float[] grad,
                            float[] m,     float[] v,
                            int size,      float lrT, int step) {

        float b1c = 1.0f - (float) Math.pow(beta1, step + 1); // bias correction
        float b2c = 1.0f - (float) Math.pow(beta2, step + 1);

        for (int i = 0; i < size; i++) {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1f - beta1) * g;
            v[i] = beta2 * v[i] + (1f - beta2) * g * g;
            param[i] -= lrT * (m[i] / b1c) / ((float) Math.sqrt(v[i] / b2c) + eps);
            grad[i] = 0f; // reset gradient
        }
    }
}
