package com.krushna.gpt.training;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.model.GptModel;
import com.krushna.gpt.model.ModelParams;
import com.krushna.gpt.tokenizer.Tokenizer;
import com.krushna.gpt.util.MathUtils;
import com.krushna.gpt.util.RandomUtils;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * GptTrainer – orchestrates the full training loop.
 *
 * <p>Mirrors the {@code main()} training loop in {@code microgpt.c}:
 * <ol>
 *   <li>For each training step pick a document (cycling through the corpus).</li>
 *   <li>Tokenise it (prepend BOS, clip/pad to BLOCK_SIZE).</li>
 *   <li>Run the forward pass and compute the cross-entropy loss.</li>
 *   <li>Run the backward pass to accumulate gradients.</li>
 *   <li>Apply the Adam update (with cosine LR decay).</li>
 *   <li>Log the loss every step.</li>
 * </ol>
 *
 * <p>After training, {@link #runInference(ModelParams, Tokenizer, int, float)}
 * generates sample sequences using the learned model.
 */
public class GptTrainer {

    // -----------------------------------------------------------------------
    //  Dependencies (injected via constructor)
    // -----------------------------------------------------------------------

    /** The GPT model (forward / backward logic). */
    private final GptModel model;

    /** The Adam optimiser. */
    private final AdamOptimizer optimizer;

    /** The dataset (shuffled documents). */
    private final Dataset dataset;

    /** The character-level tokeniser. */
    private final Tokenizer tokenizer;

    /** Source of pseudo-randomness (for inference sampling). */
    private final RandomUtils rng;

    // -----------------------------------------------------------------------
    //  Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates a {@code GptTrainer} with the provided components.
     *
     * @param model     the GPT model instance
     * @param optimizer the Adam optimiser
     * @param dataset   the pre-loaded and shuffled dataset
     * @param tokenizer the built tokeniser
     * @param rng       a PRNG instance (used for inference sampling)
     */
    public GptTrainer(GptModel model, AdamOptimizer optimizer,
                      Dataset dataset, Tokenizer tokenizer, RandomUtils rng) {
        this.model     = model;
        this.optimizer = optimizer;
        this.dataset   = dataset;
        this.tokenizer = tokenizer;
        this.rng       = rng;
    }

    // -----------------------------------------------------------------------
    //  Training loop
    // -----------------------------------------------------------------------

    /**
     * Runs the training loop for {@code numSteps} steps.
     *
     * <p>Progress is printed to stdout after every step in the format:
     * {@code step XXXX / XXXX | loss X.XXXX}
     *
     * @param params   the model parameters (updated in-place)
     * @param numSteps number of gradient descent steps to run
     */
    public void train(ModelParams params, int numSteps) {
        System.out.printf("Starting training for %d steps on %d documents.%n",
                          numSteps, dataset.size());

        // Reusable token buffers (max capacity = BLOCK_SIZE + 2 to include BOS + target)
        int[] tokens  = new int[GptConfig.MAX_DOC_LEN + 2];
        int[] targets = new int[GptConfig.BLOCK_SIZE];

        for (int step = 0; step < numSteps; step++) {
            // Pick the next document (cycle through corpus like the C code)
            String doc    = dataset.getDoc(step % dataset.size());
            int    docLen = doc.length();

            // Build token sequence: [BOS, c0, c1, ..., cn-1, BOS]
            tokens[0] = tokenizer.getBosTokenId();
            for (int i = 0; i < docLen; i++) {
                tokens[i + 1] = tokenizer.charToId(doc.charAt(i));
            }
            tokens[docLen + 1] = tokenizer.getBosTokenId();

            // Clip to BLOCK_SIZE (context window)
            int n = Math.min(GptConfig.BLOCK_SIZE, docLen + 1);

            // Targets: predict the next token at each position
            for (int pos = 0; pos < n; pos++) {
                targets[pos] = tokens[pos + 1];
            }

            // ---- Forward + loss ----
            float loss = model.forwardSequence(params, tokens, targets, n);

            // ---- Backward ----
            model.backward(params, tokens, targets, n);

            // ---- Optimizer step ----
            optimizer.step(params, step);

            // ---- Log ----
            System.out.printf("step %4d / %4d | loss %.4f%n", step + 1, numSteps, loss);
        }

        System.out.println("Training complete.");
    }

    // -----------------------------------------------------------------------
    //  Inference
    // -----------------------------------------------------------------------

    /**
     * Generates {@code numSamples} text sequences using the trained model.
     *
     * <p>Each sequence starts from the BOS token and samples up to
     * {@link GptConfig#BLOCK_SIZE} characters.  Sampling stops early if
     * the BOS token is generated (acts as an end-of-sequence marker).
     *
     * <p>Mirrors the inference section of {@code main()} in microgpt.c.
     *
     * @param params      the trained model parameters
     * @param tokenizer   the tokeniser used to decode sampled token ids
     * @param numSamples  number of sequences to generate
     * @param temperature softmax temperature (lower = more greedy)
     */
    public void runInference(ModelParams params, Tokenizer tokenizer,
                             int numSamples, float temperature) {
        System.out.println("\n--- inference ---");

        float[] logits = new float[params.vocabSize];
        float[] probs  = new float[params.vocabSize];

        for (int si = 0; si < numSamples; si++) {
            model.clearKvCache();           // isolate each generated sequence
            StringBuilder sample = new StringBuilder();
            int tokenId = tokenizer.getBosTokenId(); // start from BOS

            for (int pos = 0; pos < GptConfig.BLOCK_SIZE; pos++) {
                model.inferenceStep(params, tokenId, pos, logits);

                // Apply temperature scaling
                float invT = 1.0f / temperature;
                for (int i = 0; i < params.vocabSize; i++) {
                    logits[i] *= invT;
                }
                MathUtils.softmaxFwd(logits, params.vocabSize, probs);

                // Sample from the distribution
                tokenId = rng.weightedChoice(probs);

                if (tokenId == tokenizer.getBosTokenId()) break; // end-of-sequence
                char c = tokenizer.idToChar(tokenId);
                if (c != '\0') sample.append(c);
            }

            System.out.printf("sample %2d: %s%n", si + 1, sample.toString());
        }
    }
}
