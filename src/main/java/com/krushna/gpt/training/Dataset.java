package com.krushna.gpt.training;

import com.krushna.gpt.GptConfig;
import com.krushna.gpt.util.RandomUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Dataset – loads text documents from a file and provides shuffled access.
 *
 * <p>Each non-empty line in the source file is treated as one "document".
 * Behaviour mirrors {@code load_dataset()} and the shuffle logic in
 * {@code main()} of {@code microgpt.c}.
 *
 * <p>After calling {@link #load(Path)}, use {@link #shuffle(RandomUtils)} to
 * randomise the order before training, then fetch documents via
 * {@link #getDoc(int)}.
 */
public class Dataset {

    // -----------------------------------------------------------------------
    //  Internal state
    // -----------------------------------------------------------------------

    /** The loaded documents (lines). */
    private final List<String> lines = new ArrayList<>();

    // -----------------------------------------------------------------------
    //  Loading
    // -----------------------------------------------------------------------

    /**
     * Loads documents from a plain-text file (one document per line).
     *
     * <p>Empty lines are skipped. Each line is truncated to
     * {@link GptConfig#MAX_DOC_LEN} characters if necessary.
     * At most {@link GptConfig#MAX_DOCS} documents are retained.
     *
     * @param path path to the training text file
     * @throws IOException if the file cannot be read
     */
    public void load(Path path) throws IOException {
        lines.clear();
        try (BufferedReader br = new BufferedReader(new FileReader(path.toFile()))) {
            String line;
            while ((line = br.readLine()) != null && lines.size() < GptConfig.MAX_DOCS) {
                // strip trailing whitespace / carriage returns
                line = line.stripTrailing();
                if (!line.isEmpty()) {
                    // truncate to MAX_DOC_LEN
                    if (line.length() > GptConfig.MAX_DOC_LEN) {
                        line = line.substring(0, GptConfig.MAX_DOC_LEN);
                    }
                    lines.add(line);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Shuffle
    // -----------------------------------------------------------------------

    /**
     * Shuffles the document order in-place using the supplied PRNG.
     *
     * <p>Uses a Fisher-Yates shuffle, mirroring the C implementation's
     * document shuffling step in {@code main()}.
     *
     * @param rng the random number generator to use
     */
    public void shuffle(RandomUtils rng) {
        // Build an index permutation and apply it
        int n = lines.size();
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        rng.shuffleInts(order);

        // Reorder docs according to the permutation
        List<String> tmp = new ArrayList<>(lines);
        for (int i = 0; i < n; i++) {
            lines.set(i, tmp.get(order[i]));
        }
    }

    // -----------------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------------

    /** @return the number of documents loaded. */
    public int size() { return lines.size(); }

    /**
     * Returns the document at the given index.
     *
     * @param index the document index (0-based)
     * @return the document string
     */
    public String getDoc(int index) { return lines.get(index); }

    /** @return an unmodifiable view of all loaded documents. */
    public List<String> getLines() { return java.util.Collections.unmodifiableList(lines); }
}
