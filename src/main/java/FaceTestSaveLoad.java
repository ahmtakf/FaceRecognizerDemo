import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.solve;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

public class FaceTestSaveLoad {

    private static int width = 200;
    private static int height = 200;
    private static String trainingDir = "D:\\DEV\\HUAWEI\\FaceTest\\src\\main\\resources\\images";
    private static String testDir = "D:\\DEV\\HUAWEI\\FaceTest\\src\\main\\resources\\test\\13.jpg";

    public static void main(String[] args) {
        long startTime = System.nanoTime();

        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".xml");
            }
        };

        File[] trainFiles = root.listFiles(imgFilter);
        String[] fileNames = new String[trainFiles.length];

        for (int i = 0; i < fileNames.length; i++) {
            fileNames[i] = trainFiles[i].getName();
        }

        File testFile = new File(testDir);

        BufferedImage image = null;
        try {
            image = ImageIO.read(testFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        BufferedImage temp = FaceTestMain.resize(image, width, height);

        try {
            ImageIO.write(temp, FilenameUtils.getExtension(testFile.getAbsolutePath()), new File(testFile.getAbsolutePath()));
        } catch (IOException e) {
            e.printStackTrace();
        }

        Mat testImage = imread(testDir, CV_LOAD_IMAGE_GRAYSCALE);

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);

        //Three different algorithms
        //FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
        //FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();

        Double min = Double.MAX_VALUE;
        int resultId = 0;
        for (String file : fileNames) {
            FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

            faceRecognizer.read(trainingDir + "\\" + file);
            System.out.println(((LBPHFaceRecognizer) faceRecognizer).getLabels().getIntBuffer().get());
            faceRecognizer.predict(testImage, label, confidence);
            int predictedLabel = label.get(0);

            if (confidence.get() < min) {
                min = confidence.get();
                resultId = predictedLabel;
            }
            System.out.println("Predicted label: " + predictedLabel + " Confidence: " + confidence.get());
        }

        System.out.println("Training is done! " + resultId);
        long endTime = System.nanoTime();
        System.out.println("Execution time: " + (endTime - startTime));

    }

}
