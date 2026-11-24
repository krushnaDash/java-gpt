package com.krushna.gpt.tokenizer;

import com.krushna.gpt.GptConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tokenizer – character-level tokeniser for the GPT model.
 *
 * <p>Mirrors the tokeniser built inside {@code microgpt.c}:
 * <ol>
 *   <li>Scan all documents and collect every unique character.</li>
 *   <li>Sort the unique characters so that the mapping is deterministic.</li>
 *   <li>Assign token id {@code vocabSize - 1} to the special BOS/EOS token.</li>
 * </ol>
 *
 * <p>Usage:
 * <pre>{@code
 *   Tokenizer tok = new Tokenizer();
 *   tok.build(listOfDocuments);
 *   int[] ids = tok.encode("hello");
 *   String text = tok.decode(ids);
 * }</pre>
 */
public class Tokenizer {

    // -----------------------------------------------------------------------
    //  Internal state
    // -----------------------------------------------------------------------

    /** Sorted array of unique characters seen in training data. */
    private char[] uniqueChars;

    /** Total vocabulary size = uniqueChars.length + 1 (BOS token). */
    private int vocabSize;

    /** Token id used as the Begin-Of-Sequence / End-Of-Sequence marker. */
    private int bosTokenId;

    // -----------------------------------------------------------------------
    //  Build
    // -----------------------------------------------------------------------

    /**
     * Builds the vocabulary from a list of documents (training lines).
     *
     * <p>Mirrors {@code build_tokenizer()} in microgpt.c.
     *
     * @param docs the list of text documents to scan
     */
    public void build(List<String> docs) {
        // Collect every unique byte (char) that appears in the corpus
        boolean[] seen = new boolean[256];
        for (String doc : docs) {
            for (int i = 0; i < doc.length(); i++) {
                int c = doc.charAt(i) & 0xFF; // treat as unsigned byte
                if (c < 256) seen[c] = true;
            }
        }

        // Gather unique chars and sort them for a deterministic mapping
        List<Character> charList = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (seen[i]) charList.add((char) i);
        }
        char[] chars = new char[charList.size()];
        for (int i = 0; i < chars.length; i++) chars[i] = charList.get(i);
        Arrays.sort(chars);

        this.uniqueChars = chars;
        this.bosTokenId  = chars.length;           // BOS is the last id
        this.vocabSize   = chars.length + 1;       // +1 for the BOS token

        if (vocabSize > GptConfig.MAX_CHARS + 1) {
            throw new IllegalStateException(
                "Vocabulary size " + vocabSize + " exceeds MAX_CHARS+1=" + (GptConfig.MAX_CHARS + 1));
        }
    }

    // -----------------------------------------------------------------------
    //  Encode / decode
    // -----------------------------------------------------------------------

    /**
     * Encodes a single character to its token id.
     *
     * @param c the character to look up
     * @return the token id, or {@code -1} if the character is not in the vocabulary
     */
    public int charToId(char c) {
        // Linear scan – vocabulary is small (≤ 128 chars) so O(n) is fine
        for (int i = 0; i < uniqueChars.length; i++) {
            if (uniqueChars[i] == c) return i;
        }
        return -1; // unknown character
    }

    /**
     * Decodes a single token id back to its character.
     *
     * @param id the token id
     * @return the corresponding character, or {@code '\0'} for the BOS token
     */
    public char idToChar(int id) {
        if (id < 0 || id >= uniqueChars.length) return '\0'; // BOS or invalid
        return uniqueChars[id];
    }

    /**
     * Encodes a string into an array of token ids (no BOS prepended).
     *
     * @param text the text to encode
     * @return an array of token ids (unknown chars mapped to {@code -1})
     */
    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            ids[i] = charToId(text.charAt(i));
        }
        return ids;
    }

    /**
     * Decodes an array of token ids back to a string, skipping BOS/unknown tokens.
     *
     * @param ids array of token ids
     * @return the decoded string
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder(ids.length);
        for (int id : ids) {
            char c = idToChar(id);
            if (c != '\0') sb.append(c);
        }
        return sb.toString();
    }

    // -----------------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------------

    /** @return the total vocabulary size (characters + BOS token). */
    public int getVocabSize() { return vocabSize; }

    /** @return the token id of the BOS/EOS special token. */
    public int getBosTokenId() { return bosTokenId; }

    /** @return the number of printable/unique characters in the vocabulary. */
    public int getNumUniqueChars() { return uniqueChars.length; }
}
