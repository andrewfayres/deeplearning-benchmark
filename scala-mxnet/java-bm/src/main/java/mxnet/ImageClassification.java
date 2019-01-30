package mxnet;

import org.apache.mxnet.infer.javaapi.*;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.ResourceScope;
import org.kohsuke.args4j.Option;

import java.awt.image.BufferedImage;
import org.kohsuke.args4j.CmdLineParser;

import java.util.ArrayList;
import java.util.List;

public class ImageClassification {
    @Option(name = "--modelPathPrefix", usage = "The model to benchmark")
    private static String modelPathPrefix = "/tmp/resnet50_ssd/resnet50_ssd_model";
    @Option(name = "--inputImagePath", usage = "Input image path for single inference")
    private static String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";
    @Option(name = "--batchSize", usage = "Batchsize of the model")
    private static int batchSize = 1;
    @Option(name = "--times", usage = "Number of times to run the benchmark")
    private static int times = 1;
    @Option(name = "--context", usage = "Context to run on")
    private static String ctx = "cpu";
    
    Predictor loadModel(String modelPathPrefix, List<Context> context, int batchSize) {
        Shape inputShape = new Shape(new int[] {batchSize, 3, 224, 224});
        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        return new Predictor(modelPathPrefix, inputDescriptors, context,0);
    }
    
    NDArray loadSingleImage(String inputImagePath, List<Context> context){
        BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
        BufferedImage reShapedImg = ObjectDetector.reshapeImage(img, 224, 224);
        NDArray imgND = ObjectDetector.bufferedImageToPixels(reShapedImg, new Shape(new int[] {1, 3, 224, 224}));
        return imgND.copyTo(context.get(0));
    }
    
    List<NDArray> loadBatchImage(String inputImagePath, int batchSize, List<Context> context) {
        NDArray imgND = loadSingleImage(inputImagePath, context);
        List<NDArray> nd = new ArrayList<>();
        NDArray[] temp = new NDArray[batchSize];
        for (int i = 0; i < batchSize; i++) temp[i] = imgND.copy();
        NDArray batched = NDArray.concat(temp, batchSize, 0, null)[0];
        nd.add(batched);
        return nd;
    }
    
    double[] runSingleInference(String modelPathPrefix, String inputImagePath, List<Context> context, int times) {
        Predictor loadedModel = loadModel(modelPathPrefix, context, 1);
        List<Double> inferenceTimes = new ArrayList<Double>();
        List<NDArray> dataset = loadBatchImage(inputImagePath, 1, context);
    
    
        // Warm up intervals
        // println("Warming up the system")
        for (int i = 0; i < 5; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                loadedModel.predictWithNDArray(dataset).get(0).waitToRead();
            }
        }
        // println("Warm up done")
        
        double[] result = new double[times];
        for (int i = 0; i < times; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                long startTime = System.nanoTime();
                loadedModel.predictWithNDArray(dataset).get(0).waitToRead();
                result[i] = (System.nanoTime() - startTime) / (1e6 * 1.0);
                System.out.printf("Inference time at iteration: %d is : %f \n", i, result[i]);
            }
        }
        return result;
    }
    
    double[] runBatchInference(String modelPathPrefix, String inputImagePath, List<Context> context, int batchSize, int times) {
        Predictor loadedModel = loadModel(modelPathPrefix, context, batchSize);
        List<Double> inferenceTimes = new ArrayList<Double>();
        List<NDArray> dataset = loadBatchImage(inputImagePath, batchSize, context);
        
        // Warm up intervals
        // println("Warming up the system")
        for (int i = 0; i < 5; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                loadedModel.predictWithNDArray(dataset).get(0).waitToRead();
            }
        }
        // println("Warm up done")
        
        double[] result = new double[times];
        for (int i = 0; i < times; i++) {
            try(ResourceScope scope = new ResourceScope()) {
                long startTime = System.nanoTime();
                loadedModel.predictWithNDArray(dataset).get(0).waitToRead();
                result[i] = (System.nanoTime() - startTime) / (1e6 * 1.0);
                System.out.printf("Inference time at iteration: %d is : %f \n", i, result[i]);
            }
        }
        return result;
    }
    
    public static void main(String[] args) {
        ImageClassification inst = new ImageClassification();
        CmdLineParser parser = Utils.parse(inst, args);
        
        List<Context> context = Utils.getContext(ctx);
        
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running single inference");
            double[] inferenceTimes = inst.runSingleInference(modelPathPrefix, inputImagePath, context, times);
            Utils.printStatistics(inferenceTimes, "single_inference");
        }
        
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + batchSize);
            double[] inferenceTimes = inst.runBatchInference(modelPathPrefix, inputImagePath, context, batchSize, times);
            Utils.printStatistics(inferenceTimes, "batch_inference_1x");
        }
        
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + 2 * batchSize);
            double[] inferenceTimes = inst.runBatchInference(modelPathPrefix, inputImagePath, context, 2 * batchSize, times);
            Utils.printStatistics(inferenceTimes, "batch_inference_2x");
        }
        
        try(ResourceScope scope = new ResourceScope()) {
            System.out.println("Running batch inference with batsize : " + 4 * batchSize);
            double[] inferenceTimes = inst.runBatchInference(modelPathPrefix, inputImagePath, context, 4 * batchSize, times);
            Utils.printStatistics(inferenceTimes, "batch_inference_4x");
        }
    }
}