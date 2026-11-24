package com.krushna.gpt.io;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.model.ModelParams;
import com.krushna.gpt.tokenizer.Tokenizer;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * ModelSerializer – saves and loads a trained GPT model to/from disk.
 *
 * <p>The on-disk format is a simple binary file:
 * <pre>
 *   [4 bytes] magic number 0x474C4D42 ("GLMB")
 *   [4 bytes] format version (1)
 *   [4 bytes] vocabSize  (int)
 *   [4 bytes] nEmbd      (int)
 *   [4 bytes] nHead      (int)
 *   [4 bytes] nLayer     (int)
 *   [4 bytes] blockSize  (int)
 *   [4 bytes] numUniqueChars (int)
 *   [numUniqueChars * 2 bytes] the unique characters (UTF-16 chars)
 *   [numParams * 4 bytes] all float parameters in the order they
 *                         appear in ModelParams
 * </pre>
 *
 * <p>The Adam moment buffers are intentionally <em>not</em> saved (they can
 * be reconstructed by restarting Adam from zero, which is acceptable for
 * fine-tuning or inference scenarios).
 */
public class ModelSerializer {

    /** Magic number identifies the file as a JavaGPT model. */
    private static final int MAGIC = 0x474C4D42;  // "GLMB"

    /** Binary format version. */
    private static final int VERSION = 1;

    // -----------------------------------------------------------------------
    //  Save
    // -----------------------------------------------------------------------

    /**
     * Saves the model parameters and tokeniser vocabulary to a binary file.
     *
     * @param params    the trained model parameters
     * @param tokenizer the tokeniser used during training
     * @param path      destination file path
     * @throws IOException if the file cannot be written
     */
    public static void save(ModelParams params, Tokenizer tokenizer, Path path) throws IOException {
        try (DataOutputStream dos =
                 new DataOutputStream(new BufferedOutputStream(
                     new FileOutputStream(path.toFile())))) {

            // ---- Header ----
            dos.writeInt(MAGIC);
            dos.writeInt(VERSION);
            dos.writeInt(params.vocabSize);
            dos.writeInt(GptConfig.N_EMBD);
            dos.writeInt(GptConfig.N_HEAD);
            dos.writeInt(GptConfig.N_LAYER);
            dos.writeInt(GptConfig.BLOCK_SIZE);
            dos.writeInt(tokenizer.getNumUniqueChars());

            // ---- Vocabulary characters ----
            // We reconstruct the unique chars from the tokeniser via encode/decode
            for (int i = 0; i < tokenizer.getNumUniqueChars(); i++) {
                dos.writeChar(tokenizer.idToChar(i));
            }

            // ---- Parameters ----
            int E = GptConfig.N_EMBD;
            int V = params.vocabSize;
            writeFloats(dos, params.wte,    V * E);
            writeFloats(dos, params.wpe,    GptConfig.BLOCK_SIZE * E);
            writeFloats(dos, params.lmHead, V * E);

            for (int li = 0; li < GptConfig.N_LAYER; li++) {
                writeFloats(dos, params.attnWq[li], E * E);
                writeFloats(dos, params.attnWk[li], E * E);
                writeFloats(dos, params.attnWv[li], E * E);
                writeFloats(dos, params.attnWo[li], E * E);
                writeFloats(dos, params.mlpFc1[li],  GptConfig.MLP_DIM * E);
                writeFloats(dos, params.mlpFc2[li],  GptConfig.MLP_DIM * E);
            }
        }

        System.out.printf("Model saved to %s%n", path);
    }

    // -----------------------------------------------------------------------
    //  Load
    // -----------------------------------------------------------------------

    /**
     * Loads model parameters from a binary file previously created by
     * {@link #save(ModelParams, Tokenizer, Path)}.
     *
     * <p>The returned {@link ModelParams} contains all weights but has zeroed
     * gradient and Adam buffers (suitable for inference or further training).
     *
     * @param path      source file path
     * @param tokenizer the tokeniser to populate with the saved vocabulary
     * @return a fully populated {@link ModelParams} ready for inference
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if the file format is invalid
     */
    public static ModelParams load(Path path, Tokenizer tokenizer) throws IOException {
        try (DataInputStream dis =
                 new DataInputStream(new BufferedInputStream(
                     new FileInputStream(path.toFile())))) {

            // ---- Header validation ----
            int magic = dis.readInt();
            if (magic != MAGIC) {
                throw new IllegalArgumentException(
                    "Invalid model file – magic number mismatch (expected 0x474C4D42, got 0x"
                    + Integer.toHexString(magic) + ")");
            }
            int version = dis.readInt();
            if (version != VERSION) {
                throw new IllegalArgumentException(
                    "Unsupported model file version: " + version);
            }

            int vocabSize    = dis.readInt();
            int nEmbd        = dis.readInt();
            int nHead        = dis.readInt();
            int nLayer       = dis.readInt();
            int blockSize    = dis.readInt();
            int numUniqueChars = dis.readInt();

            // Sanity checks against compile-time config
            assertConfig("N_EMBD",     nEmbd,     GptConfig.N_EMBD);
            assertConfig("N_HEAD",     nHead,     GptConfig.N_HEAD);
            assertConfig("N_LAYER",    nLayer,    GptConfig.N_LAYER);
            assertConfig("BLOCK_SIZE", blockSize, GptConfig.BLOCK_SIZE);

            // ---- Vocabulary ----
            char[] chars = new char[numUniqueChars];
            for (int i = 0; i < numUniqueChars; i++) {
                chars[i] = dis.readChar();
            }
            // Re-build tokeniser from the saved chars
            java.util.List<String> fakeDoc = new java.util.ArrayList<>();
            fakeDoc.add(new String(chars));
            tokenizer.build(fakeDoc);

            // ---- Parameters ----
            ModelParams params = new ModelParams();
            // Allocate without randomisation (std = 0 means zeros, we overwrite anyway)
            params.vocabSize = vocabSize;
            int E = GptConfig.N_EMBD;
            int V = vocabSize;

            params.wte     = readFloats(dis, V * E);
            params.dWte    = new float[V * E];
            params.mWte    = new float[V * E];
            params.vWte    = new float[V * E];

            params.wpe     = readFloats(dis, GptConfig.BLOCK_SIZE * E);
            params.dWpe    = new float[GptConfig.BLOCK_SIZE * E];
            params.mWpe    = new float[GptConfig.BLOCK_SIZE * E];
            params.vWpe    = new float[GptConfig.BLOCK_SIZE * E];

            params.lmHead  = readFloats(dis, V * E);
            params.dLmHead = new float[V * E];
            params.mLmHead = new float[V * E];
            params.vLmHead = new float[V * E];

            params.attnWq  = new float[GptConfig.N_LAYER][];
            params.dAttnWq = new float[GptConfig.N_LAYER][];
            params.attnWk  = new float[GptConfig.N_LAYER][];
            params.dAttnWk = new float[GptConfig.N_LAYER][];
            params.attnWv  = new float[GptConfig.N_LAYER][];
            params.dAttnWv = new float[GptConfig.N_LAYER][];
            params.attnWo  = new float[GptConfig.N_LAYER][];
            params.dAttnWo = new float[GptConfig.N_LAYER][];
            params.mlpFc1  = new float[GptConfig.N_LAYER][];
            params.dMlpFc1 = new float[GptConfig.N_LAYER][];
            params.mlpFc2  = new float[GptConfig.N_LAYER][];
            params.dMlpFc2 = new float[GptConfig.N_LAYER][];

            // Adam buffers (not stored, zero-initialised)
            params.mWq  = new float[GptConfig.N_LAYER][];
            params.vWq  = new float[GptConfig.N_LAYER][];
            params.mWk  = new float[GptConfig.N_LAYER][];
            params.vWk  = new float[GptConfig.N_LAYER][];
            params.mWv  = new float[GptConfig.N_LAYER][];
            params.vWv  = new float[GptConfig.N_LAYER][];
            params.mWo  = new float[GptConfig.N_LAYER][];
            params.vWo  = new float[GptConfig.N_LAYER][];
            params.mFc1 = new float[GptConfig.N_LAYER][];
            params.vFc1 = new float[GptConfig.N_LAYER][];
            params.mFc2 = new float[GptConfig.N_LAYER][];
            params.vFc2 = new float[GptConfig.N_LAYER][];

            for (int li = 0; li < GptConfig.N_LAYER; li++) {
                params.attnWq[li]  = readFloats(dis, E * E);
                params.dAttnWq[li] = new float[E * E];
                params.mWq[li]     = new float[E * E];
                params.vWq[li]     = new float[E * E];

                params.attnWk[li]  = readFloats(dis, E * E);
                params.dAttnWk[li] = new float[E * E];
                params.mWk[li]     = new float[E * E];
                params.vWk[li]     = new float[E * E];

                params.attnWv[li]  = readFloats(dis, E * E);
                params.dAttnWv[li] = new float[E * E];
                params.mWv[li]     = new float[E * E];
                params.vWv[li]     = new float[E * E];

                params.attnWo[li]  = readFloats(dis, E * E);
                params.dAttnWo[li] = new float[E * E];
                params.mWo[li]     = new float[E * E];
                params.vWo[li]     = new float[E * E];

                params.mlpFc1[li]  = readFloats(dis, GptConfig.MLP_DIM * E);
                params.dMlpFc1[li] = new float[GptConfig.MLP_DIM * E];
                params.mFc1[li]    = new float[GptConfig.MLP_DIM * E];
                params.vFc1[li]    = new float[GptConfig.MLP_DIM * E];

                params.mlpFc2[li]  = readFloats(dis, GptConfig.MLP_DIM * E);
                params.dMlpFc2[li] = new float[GptConfig.MLP_DIM * E];
                params.mFc2[li]    = new float[GptConfig.MLP_DIM * E];
                params.vFc2[li]    = new float[GptConfig.MLP_DIM * E];
            }

            System.out.printf("Model loaded from %s (vocabSize=%d)%n", path, vocabSize);
            return params;
        }
    }

    // -----------------------------------------------------------------------
    //  I/O helpers
    // -----------------------------------------------------------------------

    /** Writes {@code count} floats from {@code arr} to the stream. */
    private static void writeFloats(DataOutputStream dos, float[] arr, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            dos.writeFloat(arr[i]);
        }
    }

    /** Reads exactly {@code count} floats from the stream into a new array. */
    private static float[] readFloats(DataInputStream dis, int count) throws IOException {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++) {
            arr[i] = dis.readFloat();
        }
        return arr;
    }

    /** Throws if the saved config value differs from the current compile-time value. */
    private static void assertConfig(String name, int saved, int current) {
        if (saved != current) {
            throw new IllegalArgumentException(
                "Model file was trained with " + name + "=" + saved
                + " but the current code uses " + name + "=" + current
                + ". Rebuild the code to match.");
        }
    }
}
