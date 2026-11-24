package com.krushna.gpt;

import com.krushna.gpt.io.HuggingFaceUploader;
import com.krushna.gpt.io.ModelSerializer;
import com.krushna.gpt.model.GptModel;
import com.krushna.gpt.model.ModelParams;
import com.krushna.gpt.tokenizer.Tokenizer;
import com.krushna.gpt.training.AdamOptimizer;
import com.krushna.gpt.training.Dataset;
import com.krushna.gpt.training.GptTrainer;
import com.krushna.gpt.util.RandomUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Main – entry point for the Java GPT application.
 *
 * <p>Supported run modes (first command-line argument):
 * <ul>
 *   <li>{@code train}    – train the model on a text file, then save it.</li>
 *   <li>{@code infer}    – load a saved model and generate text samples.</li>
 *   <li>{@code upload}   – upload a saved model to Hugging Face Hub.</li>
 * </ul>
 *
 * <h2>Usage examples</h2>
 * <pre>
 *   # Train from scratch and save
 *   java -jar java-gpt.jar train input.txt model.bin
 *
 *   # Load and generate
 *   java -jar java-gpt.jar infer model.bin 20 0.5
 *
 *   # Upload to Hugging Face
 *   java -jar java-gpt.jar upload model.bin &lt;hf_token&gt; &lt;owner&gt; &lt;repo_name&gt;
 * </pre>
 *
 * <p>If no arguments are given, the program defaults to:
 * <ul>
 *   <li>Training file: {@code input.txt}</li>
 *   <li>Model save path: {@code model.bin}</li>
 *   <li>Steps: {@link GptConfig#DEFAULT_NUM_STEPS}</li>
 *   <li>Samples after training: {@link GptConfig#DEFAULT_NUM_SAMPLES}</li>
 *   <li>Temperature: {@link GptConfig#DEFAULT_TEMPERATURE}</li>
 * </ul>
 */
public class Main {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            // ----------------------------------------------------------------
            //  Default mode: train on input.txt, save to model.bin, then infer
            // ----------------------------------------------------------------
            trainAndInfer(
                Paths.get("input.txt"),
                Paths.get("model.bin"),
                GptConfig.DEFAULT_NUM_STEPS,
                GptConfig.DEFAULT_NUM_SAMPLES,
                GptConfig.DEFAULT_TEMPERATURE
            );

        } else {
            String mode = args[0].toLowerCase();
            switch (mode) {
                case "train" -> {
                    // java -jar java-gpt.jar train <inputFile> [modelFile] [steps]
                    Path inputFile = Paths.get(args.length > 1 ? args[1] : "input.txt");
                    Path modelFile = Paths.get(args.length > 2 ? args[2] : "model.bin");
                    int  steps     = args.length > 3 ? Integer.parseInt(args[3])
                                                      : GptConfig.DEFAULT_NUM_STEPS;
                    trainAndInfer(inputFile, modelFile, steps,
                                  GptConfig.DEFAULT_NUM_SAMPLES,
                                  GptConfig.DEFAULT_TEMPERATURE);
                }

                case "infer" -> {
                    // java -jar java-gpt.jar infer <modelFile> [numSamples] [temperature]
                    Path  modelFile  = Paths.get(args.length > 1 ? args[1] : "model.bin");
                    int   numSamples = args.length > 2 ? Integer.parseInt(args[2])
                                                       : GptConfig.DEFAULT_NUM_SAMPLES;
                    float temp       = args.length > 3 ? Float.parseFloat(args[3])
                                                       : GptConfig.DEFAULT_TEMPERATURE;
                    inferOnly(modelFile, numSamples, temp);
                }

                case "upload" -> {
                    // java -jar java-gpt.jar upload <modelFile> <hfToken> <owner> <repoName>
                    if (args.length < 5) {
                        System.err.println(
                            "Usage: upload <modelFile> <hfToken> <owner> <repoName>");
                        System.exit(1);
                    }
                    Path   modelFile = Paths.get(args[1]);
                    String hfToken   = args[2];
                    String owner     = args[3];
                    String repoName  = args[4];
                    uploadToHuggingFace(modelFile, hfToken, owner, repoName);
                }

                default -> {
                    System.err.println("Unknown mode: " + mode);
                    System.err.println("Modes: train | infer | upload");
                    System.exit(1);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Mode implementations
    // -----------------------------------------------------------------------

    /**
     * Trains the model from scratch, then immediately generates sample text.
     *
     * @param inputFile  path to the training text file (one document per line)
     * @param modelFile  path where the trained model will be saved
     * @param numSteps   number of training steps
     * @param numSamples number of sample sequences to generate after training
     * @param temp       sampling temperature
     */
    private static void trainAndInfer(Path inputFile, Path modelFile,
                                      int numSteps, int numSamples, float temp)
            throws IOException {

        RandomUtils randomUtils = new RandomUtils(42L);

        // ---- Load dataset ----
        Dataset dataset = new Dataset();
        dataset.load(inputFile);
        System.out.printf("num docs: %d%n", dataset.size());

        // Shuffle documents for training here one line from input text consider
        // as one doc. hence shuffler will prevetns bias
        dataset.shuffle(randomUtils);

        // ---- Build tokeniser ----
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.build(dataset.getLines());
        System.out.printf("vocab size: %d%n", tokenizer.getVocabSize());

        // ---- Initialise model parameters ----
        ModelParams params = new ModelParams();
        params.init(tokenizer.getVocabSize(), randomUtils);

        // ---- Set up training components ----
        GptModel      model     = new GptModel();
        AdamOptimizer optimizer = new AdamOptimizer(
            GptConfig.DEFAULT_LR, GptConfig.ADAM_BETA1,
            GptConfig.ADAM_BETA2, GptConfig.ADAM_EPS, numSteps);
        GptTrainer trainer = new GptTrainer(model, optimizer, dataset, tokenizer, randomUtils);

        // ---- Train ----
        trainer.train(params, numSteps);

        // ---- Save model ----
        ModelSerializer.save(params, tokenizer, modelFile);

        // ---- Generate samples ----
        trainer.runInference(params, tokenizer, numSamples, temp);
    }

    /**
     * Loads a saved model and generates text samples without any training.
     *
     * @param modelFile  path to the saved binary model file
     * @param numSamples number of sample sequences to generate
     * @param temp       sampling temperature
     */
    private static void inferOnly(Path modelFile, int numSamples, float temp)
            throws IOException {

        RandomUtils rng       = new RandomUtils(42L);
        Tokenizer   tokenizer = new Tokenizer();

        // Load model (also restores the tokeniser vocabulary)
        ModelParams params = ModelSerializer.load(modelFile, tokenizer);

        // Create dummy dataset + trainer just for the inference helper
        GptModel      model  = new GptModel();
        GptTrainer    trainer = new GptTrainer(model, new AdamOptimizer(),
                                               new Dataset(), tokenizer, rng);
        trainer.runInference(params, tokenizer, numSamples, temp);
    }

    /**
     * Uploads a saved model binary and a generated model card to Hugging Face Hub.
     *
     * @param modelFile  path to the saved binary model file
     * @param hfToken    Hugging Face write-access API token
     * @param owner      HF username or organisation that owns the target repo
     * @param repoName   name of the HF model repository
     */
    private static void uploadToHuggingFace(Path modelFile, String hfToken,
                                            String owner, String repoName)
            throws IOException {

        HuggingFaceUploader uploader = new HuggingFaceUploader(hfToken);

        // Create repository (no-op if it already exists)
        uploader.createRepository(owner, repoName,
            "A character-level GPT trained in pure Java (java-gpt)", false);

        // Upload model weights
        uploader.uploadFile(owner, repoName, modelFile, "model.bin");

        // Upload a basic model card (README.md)
        uploader.uploadModelCard(owner, repoName,
            "Java GPT – " + repoName,
            "A minimal character-level GPT trained from scratch in Java, "
            + "following the architecture of microgpt-c.");

        System.out.printf("Upload complete: https://huggingface.co/%s/%s%n",
                          owner, repoName);
    }
}
