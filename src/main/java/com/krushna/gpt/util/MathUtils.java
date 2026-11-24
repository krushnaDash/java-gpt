package com.krushna.gpt.util;

/**
 * MathUtils – pure-Java implementations of the neural-network building blocks
 * used throughout the GPT forward and backward passes.
 *
 * <p>All methods are {@code static} so callers never need to instantiate this
 * class.  Arrays are passed in as plain {@code float[]} to keep allocation
 * out of the hot path; callers are responsible for allocating output buffers.
 *
 * <p>Naming and semantics match the inlined helpers in {@code microgpt.c}:
 * <ul>
 *   <li>{@link #linearFwd}   – matrix–vector multiply (forward)</li>
 *   <li>{@link #linearBwdX}  – gradient w.r.t. input x</li>
 *   <li>{@link #linearBwdW}  – gradient w.r.t. weight matrix W</li>
 *   <li>{@link #rmsnormFwd}  – RMS layer normalisation (forward)</li>
 *   <li>{@link #rmsnormBwd}  – RMS layer normalisation (backward)</li>
 *   <li>{@link #softmaxFwd}  – numerically stable softmax</li>
 * </ul>
 */
public final class MathUtils {

    // prevent instantiation
    private MathUtils() {}

    // -----------------------------------------------------------------------
    //  Linear layer
    // -----------------------------------------------------------------------

    /**
     * Forward pass of a linear (fully-connected) layer: {@code out = W * x}.
     *
     * <p>Weight matrix {@code w} is stored row-major with shape
     * {@code [nOut][nIn]}.
     *
     * @param x    input vector of length {@code nIn}
     * @param w    weight matrix of length {@code nOut * nIn} (row-major)
     * @param nOut number of output features
     * @param nIn  number of input features
     * @param out  output buffer of length {@code nOut} (overwritten)
     */
    public static void linearFwd(float[] x, float[] w, int nOut, int nIn, float[] out) {
        for (int r = 0; r < nOut; r++) {
            float s = 0f;
            int rowOffset = r * nIn;
            for (int c = 0; c < nIn; c++) {
                s += w[rowOffset + c] * x[c];
            }
            out[r] = s;
        }
    }

    /**
     * Backward pass of a linear layer: gradient w.r.t. input {@code x}.
     *
     * <p>{@code dx[c] += Σ_r  dOut[r] * W[r, c]}  (accumulates into dx)
     *
     * @param w    weight matrix of length {@code nOut * nIn} (row-major)
     * @param dOut upstream gradient of length {@code nOut}
     * @param nOut number of output features
     * @param nIn  number of input features
     * @param dx   gradient accumulator of length {@code nIn} (added to)
     */
    public static void linearBwdX(float[] w, float[] dOut, int nOut, int nIn, float[] dx) {
        for (int c = 0; c < nIn; c++) {
            float s = 0f;
            for (int r = 0; r < nOut; r++) {
                s += dOut[r] * w[r * nIn + c];
            }
            dx[c] += s;
        }
    }

    /**
     * Backward pass of a linear layer: gradient w.r.t. weight matrix {@code W}.
     *
     * <p>{@code dW[r, c] += dOut[r] * x[c]}  (accumulates into dW)
     *
     * @param x    input vector of length {@code nIn}
     * @param dOut upstream gradient of length {@code nOut}
     * @param nOut number of output features
     * @param nIn  number of input features
     * @param dw   gradient accumulator of length {@code nOut * nIn} (added to)
     */
    public static void linearBwdW(float[] x, float[] dOut, int nOut, int nIn, float[] dw) {
        for (int r = 0; r < nOut; r++) {
            float dr = dOut[r];
            int rowOffset = r * nIn;
            for (int c = 0; c < nIn; c++) {
                dw[rowOffset + c] += dr * x[c];
            }
        }
    }

    // -----------------------------------------------------------------------
    //  RMS normalisation
    // -----------------------------------------------------------------------

    /**
     * Forward pass of RMS layer normalisation.
     *
     * <p>Computes {@code out[i] = x[i] / RMS(x)} where
     * {@code RMS(x) = sqrt(mean(x²) + ε)}.
     *
     * @param x   input vector of length {@code n}
     * @param n   vector length
     * @param out output buffer of length {@code n} (overwritten)
     * @return    the scale factor {@code 1 / RMS(x)} saved for the backward pass
     */
    public static float rmsnormFwd(float[] x, int n, float[] out) {
        float ms = 0f;
        for (int i = 0; i < n; i++) {
            ms += x[i] * x[i];
        }
        ms /= n;
        float scale = 1.0f / (float) Math.sqrt(ms + 1e-5f);
        for (int i = 0; i < n; i++) {
            out[i] = x[i] * scale;
        }
        return scale;
    }

    /**
     * Backward pass of RMS layer normalisation.
     *
     * <p>Accumulates gradient into {@code dx}:
     * {@code dx[i] += scale * dOut[i]  -  coeff * x[i] * dot(dOut, x)}
     * where {@code coeff = scale³ / n}.
     *
     * @param x     original input to the normalisation layer
     * @param scale the scale factor returned by {@link #rmsnormFwd}
     * @param dOut  upstream gradient of length {@code n}
     * @param n     vector length
     * @param dx    gradient accumulator of length {@code n} (added to)
     */
    public static void rmsnormBwd(float[] x, float scale, float[] dOut, int n, float[] dx) {
        float dot = 0f;
        for (int i = 0; i < n; i++) {
            dot += dOut[i] * x[i];
        }
        float coeff = scale * scale * scale / n;
        for (int i = 0; i < n; i++) {
            dx[i] += scale * dOut[i] - coeff * x[i] * dot;
        }
    }

    // -----------------------------------------------------------------------
    //  Softmax
    // -----------------------------------------------------------------------

    /**
     * Numerically stable softmax in-place.
     *
     * <p>Subtracts the maximum logit before exponentiation to avoid overflow.
     * Mirrors {@code softmax_fwd()} in microgpt.c.
     *
     * @param logits input logits of length {@code n}
     * @param n      vector length
     * @param probs  output probabilities of length {@code n} (overwritten)
     */
    public static void softmaxFwd(float[] logits, int n, float[] probs) {
        // find max for numerical stability
        float mx = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > mx) mx = logits[i];
        }
        // exponentiate and sum
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            probs[i] = (float) Math.exp(logits[i] - mx);
            sum += probs[i];
        }
        // normalise
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            probs[i] *= inv;
        }
    }

    // -----------------------------------------------------------------------
    //  Activation – squared-ReLU (used in the MLP, matching microgpt.c)
    // -----------------------------------------------------------------------

    /**
     * Squared-ReLU forward pass: {@code out[i] = max(0, x[i])²}.
     *
     * @param pre  pre-activation values of length {@code n}
     * @param n    vector length
     * @param post output buffer of length {@code n} (overwritten)
     */
    public static void sqReluFwd(float[] pre, int n, float[] post) {
        for (int i = 0; i < n; i++) {
            float v = pre[i];
            post[i] = v > 0f ? v * v : 0f;
        }
    }

    /**
     * Squared-ReLU backward: gradient w.r.t. {@code pre}.
     *
     * <p>{@code dPre[i] = (pre[i] > 0) ? 2 * pre[i] * dPost[i] : 0}
     *
     * @param pre   pre-activation values saved during the forward pass
     * @param dPost upstream gradient of length {@code n}
     * @param n     vector length
     * @param dPre  output gradient buffer of length {@code n} (overwritten)
     */
    public static void sqReluBwd(float[] pre, float[] dPost, int n, float[] dPre) {
        for (int i = 0; i < n; i++) {
            dPre[i] = pre[i] > 0f ? 2.0f * pre[i] * dPost[i] : 0f;
        }
    }

    // -----------------------------------------------------------------------
    //  Array utilities
    // -----------------------------------------------------------------------

    /**
     * Allocates and returns a float array of {@code size} zeros.
     *
     * @param size desired length
     * @return a new zero-filled float array
     */
    public static float[] zeros(int size) {
        return new float[size]; // Java default-initialises to 0.0f
    }

    /**
     * Fills every element of {@code arr} with {@code 0.0f}.
     *
     * @param arr the array to zero-fill (modified in-place)
     */
    public static void zero(float[] arr) {
        java.util.Arrays.fill(arr, 0f);
    }

    /**
     * Copies {@code src} into {@code dst} from index 0.
     * Equivalent to {@code System.arraycopy(src, 0, dst, 0, src.length)}.
     *
     * @param src source array
     * @param dst destination array (must be at least as long as src)
     */
    public static void copy(float[] src, float[] dst) {
        System.arraycopy(src, 0, dst, 0, src.length);
    }
}
