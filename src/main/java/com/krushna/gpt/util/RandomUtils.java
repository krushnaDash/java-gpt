package com.krushna.gpt.util;

/**
 * RandomUtils – deterministic pseudo-random number generation.
 *
 * <p>Implements the same <em>xorshift64</em> PRNG used in {@code microgpt.c}
 * so that experiments can be reproduced across C and Java runs when seeded
 * with the same value (default seed = 42).
 *
 * <p>Thread-safety: this class is <em>not</em> thread-safe; use one instance
 * per thread if parallel training is needed.
 */
public final class RandomUtils {

    /** Current state of the 64-bit xorshift PRNG. */
    private long rngState;

    /**
     * Constructs a new RandomUtils with the default seed (42),
     * matching the C implementation.
     */
    public RandomUtils() {
        this(42L);
    }

    /**
     * Constructs a new RandomUtils with an explicit seed.
     *
     * @param seed the initial PRNG state (must be non-zero)
     */
    public RandomUtils(long seed) {
        this.rngState = seed == 0 ? 1L : seed; // xorshift must not be 0
    }

    // -----------------------------------------------------------------------
    //  Core PRNG
    // -----------------------------------------------------------------------

    /**
     * Advances the xorshift64 state and returns the next 64-bit integer.
     * Mirrors {@code rng_next()} in microgpt.c.
     *
     * @return a pseudo-random unsigned 64-bit value (as signed Java long)
     */
    private long nextLong() {
        rngState ^= (rngState << 13);
        rngState ^= (rngState >>> 7);   // logical (unsigned) right shift
        rngState ^= (rngState << 17);
        return rngState;
    }

    /**
     * Returns a uniform double in [0, 1).
     * Mirrors {@code rng_uniform()} in microgpt.c.
     *
     * @return a value in [0.0, 1.0)
     */
    public double nextUniform() {
        // Use top 53 bits (unsigned), same as C: (rng_next() >> 11) * (1.0 / 2^53)
        long raw = nextLong() >>> 11;           // logical shift → positive
        return raw * (1.0 / 9_007_199_254_740_992.0); // 2^53
    }

    /**
     * Returns a Gaussian (normal) distributed float with the given mean and
     * standard deviation using the Box–Muller transform.
     * Mirrors {@code rng_gauss()} in microgpt.c.
     *
     * @param mean the desired mean
     * @param std  the desired standard deviation
     * @return a Gaussian-distributed float
     */
    public float nextGaussian(float mean, float std) {
        double u1 = nextUniform();
        double u2 = nextUniform();
        if (u1 < 1e-30) {
            u1 = 1e-30;
        }
        double gauss = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return mean + std * (float) gauss;
    }

    // -----------------------------------------------------------------------
    //  Utility
    // -----------------------------------------------------------------------

    /**
     * Fisher-Yates (Knuth) shuffle of an integer array.
     * Mirrors {@code shuffle_ints()} in microgpt.c.
     *
     * @param arr the array to shuffle in-place
     */
    public void shuffleInts(int[] arr) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = (int) (nextUniform() * (i + 1));
            // swap arr[i] and arr[j]
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    /**
     * Weighted random selection: picks index {@code i} with probability
     * proportional to {@code weights[i]}.
     * Mirrors {@code weighted_choice()} in microgpt.c.
     *
     * @param weights array of non-negative weights (need not sum to 1)
     * @return the chosen index
     */
    public int weightedChoice(float[] weights) {
        float total = 0f;
        for (float w : weights) {
            total += w;
        }
        float r = (float) nextUniform() * total;
        float cum = 0f;
        for (int i = 0; i < weights.length; i++) {
            cum += weights[i];
            if (r < cum) {
                return i;
            }
        }
        return weights.length - 1; // fallback for floating-point rounding
    }
}
