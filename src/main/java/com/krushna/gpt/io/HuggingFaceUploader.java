package com.krushna.gpt.io;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

/**
 * HuggingFaceUploader – uploads a trained model binary and a model card to
 * a Hugging Face Hub repository using the Hub REST API.
 *
 * <p>Prerequisites:
 * <ol>
 *   <li>A Hugging Face account and a write-access API token.</li>
 *   <li>The target repository already exists (or
 *       {@link #createRepository(String, String, String, boolean)} is called
 *       first to create it).</li>
 * </ol>
 *
 * <p>Usage:
 * <pre>{@code
 *   HuggingFaceUploader uploader = new HuggingFaceUploader(hfToken);
 *   uploader.createRepository(owner, repoName, "A tiny GPT trained in Java", false);
 *   uploader.uploadFile(owner, repoName, modelPath, "model.bin");
 *   uploader.uploadModelCard(owner, repoName, "My Java GPT", "This model ...");
 * }</pre>
 *
 * <p>The implementation uses only the JDK's built-in {@link HttpURLConnection}
 * so that no extra HTTP client dependency is required.
 */
public class HuggingFaceUploader {

    // -----------------------------------------------------------------------
    //  Constants
    // -----------------------------------------------------------------------

    /** Hugging Face Hub API base URL. */
    private static final String HF_API_BASE = "https://huggingface.co/api";

    // -----------------------------------------------------------------------
    //  State
    // -----------------------------------------------------------------------

    /** Hugging Face write-access API token (starts with "hf_…"). */
    private final String apiToken;

    // -----------------------------------------------------------------------
    //  Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates an uploader with the given Hugging Face API token.
     *
     * @param apiToken a write-access Hugging Face API token (e.g. "hf_xxxx")
     */
    public HuggingFaceUploader(String apiToken) {
        if (apiToken == null || apiToken.isBlank()) {
            throw new IllegalArgumentException("Hugging Face API token must not be blank.");
        }
        this.apiToken = apiToken;
    }

    // -----------------------------------------------------------------------
    //  Repository management
    // -----------------------------------------------------------------------

    /**
     * Creates a new model repository on Hugging Face Hub.
     *
     * <p>If the repository already exists the API returns 409 Conflict,
     * which this method treats as a non-fatal condition (logs a warning and
     * continues).
     *
     * @param owner       the HF username or organisation that will own the repo
     * @param repoName    the desired repository name (slug-formatted)
     * @param description a short description shown on the model card
     * @param isPrivate   {@code true} to create a private repository
     * @throws IOException if the HTTP request fails
     */
    public void createRepository(String owner, String repoName,
                                 String description, boolean isPrivate) throws IOException {
        String url = HF_API_BASE + "/repos/create";
        String payload = String.format(
            "{\"type\":\"model\",\"name\":\"%s\",\"private\":%b,\"sdk\":null}",
            repoName, isPrivate);

        int responseCode = post(url, payload);
        if (responseCode == 409) {
            System.out.printf("[HuggingFace] Repository %s/%s already exists – continuing.%n",
                              owner, repoName);
        } else if (responseCode < 200 || responseCode >= 300) {
            throw new IOException("Failed to create repository – HTTP " + responseCode);
        } else {
            System.out.printf("[HuggingFace] Repository %s/%s created.%n", owner, repoName);
        }
    }

    // -----------------------------------------------------------------------
    //  File upload
    // -----------------------------------------------------------------------

    /**
     * Uploads a local file to a Hugging Face Hub repository.
     *
     * <p>Uses the LFS-bypass upload endpoint
     * {@code PUT /api/repos/{owner}/{repo}/upload/{filePath}}.
     *
     * @param owner        the HF username or organisation
     * @param repoName     the repository name
     * @param localFile    path to the local file to upload
     * @param remotePath   the target path within the repository (e.g. {@code "model.bin"})
     * @throws IOException if the file cannot be read or the upload fails
     */
    public void uploadFile(String owner, String repoName,
                           Path localFile, String remotePath) throws IOException {
        byte[] data = Files.readAllBytes(localFile);
        String url  = HF_API_BASE + "/repos/" + owner + "/" + repoName
                      + "/upload/" + remotePath;

        HttpURLConnection conn = openConnection(url, "PUT");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/octet-stream");
        conn.setRequestProperty("Content-Length", String.valueOf(data.length));

        try (OutputStream os = conn.getOutputStream()) {
            os.write(data);
        }

        int code = conn.getResponseCode();
        if (code < 200 || code >= 300) {
            throw new IOException("File upload failed – HTTP " + code
                                  + " for " + remotePath);
        }
        System.out.printf("[HuggingFace] Uploaded %s → %s/%s/%s%n",
                          localFile.getFileName(), owner, repoName, remotePath);
    }

    // -----------------------------------------------------------------------
    //  Model card
    // -----------------------------------------------------------------------

    /**
     * Creates or replaces the {@code README.md} (model card) in the repository.
     *
     * @param owner       the HF username or organisation
     * @param repoName    the repository name
     * @param modelTitle  title shown at the top of the model card
     * @param description longer description of the model
     * @throws IOException if the upload fails
     */
    public void uploadModelCard(String owner, String repoName,
                                String modelTitle, String description) throws IOException {
        String cardContent = buildModelCard(modelTitle, description);
        byte[] data = cardContent.getBytes(StandardCharsets.UTF_8);

        String url = HF_API_BASE + "/repos/" + owner + "/" + repoName
                     + "/upload/README.md";

        HttpURLConnection conn = openConnection(url, "PUT");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "text/plain; charset=utf-8");
        conn.setRequestProperty("Content-Length", String.valueOf(data.length));

        try (OutputStream os = conn.getOutputStream()) {
            os.write(data);
        }

        int code = conn.getResponseCode();
        if (code < 200 || code >= 300) {
            throw new IOException("Model card upload failed – HTTP " + code);
        }
        System.out.printf("[HuggingFace] Model card uploaded to %s/%s%n", owner, repoName);
    }

    // -----------------------------------------------------------------------
    //  Private helpers
    // -----------------------------------------------------------------------

    /**
     * Opens an {@link HttpURLConnection} with the Authorization header set.
     *
     * @param url    the full URL
     * @param method the HTTP method ("GET", "POST", "PUT", …)
     */
    private HttpURLConnection openConnection(String url, String method) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod(method);
        conn.setRequestProperty("Authorization", "Bearer " + apiToken);
        conn.setRequestProperty("Accept", "application/json");
        conn.setConnectTimeout(30_000);
        conn.setReadTimeout(120_000);
        return conn;
    }

    /**
     * Posts a JSON payload to the given URL and returns the HTTP response code.
     *
     * @param url     the target URL
     * @param payload the JSON string body
     */
    private int post(String url, String payload) throws IOException {
        byte[] body = payload.getBytes(StandardCharsets.UTF_8);
        HttpURLConnection conn = openConnection(url, "POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        conn.setRequestProperty("Content-Length", String.valueOf(body.length));
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body);
        }
        return conn.getResponseCode();
    }

    /**
     * Builds a minimal Hugging Face model card (Markdown + YAML front-matter).
     *
     * @param title       model title
     * @param description model description
     * @return the README.md content
     */
    private String buildModelCard(String title, String description) {
        return "---\n"
             + "language:\n- en\n"
             + "license: mit\n"
             + "tags:\n- gpt\n- java\n- character-level\n- from-scratch\n"
             + "---\n\n"
             + "# " + title + "\n\n"
             + description + "\n\n"
             + "## About\n\n"
             + "This model was trained from scratch in pure Java using the "
             + "[java-gpt](https://github.com/krushna/java-gpt) project, "
             + "which is a Java port of [microgpt-c](https://github.com/krushna/microgpt-c).\n\n"
             + "## Architecture\n\n"
             + "| Parameter       | Value |\n"
             + "|-----------------|-------|\n"
             + "| Embedding dim   | 32    |\n"
             + "| Attention heads | 4     |\n"
             + "| Layers          | 1     |\n"
             + "| Context window  | 8     |\n"
             + "| Activation      | Squared ReLU |\n"
             + "| Normalisation   | RMSNorm |\n";
    }
}
