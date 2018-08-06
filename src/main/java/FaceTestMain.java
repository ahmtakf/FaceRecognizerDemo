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
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

public class FaceTestMain {

    private static int width = 200;
    private static int height = 200;
    private static String trainingDir = "D:\\DEV\\HUAWEI\\FaceTest\\src\\main\\resources\\images";
    private static String testDir = "D:\\DEV\\HUAWEI\\FaceTest\\src\\main\\resources\\test\\13.jpg";

    public static void main(String[] args) {
        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        for (int i = 0; i < imageFiles.length; i++) {
            BufferedImage image = null;
            try {
                image = ImageIO.read(imageFiles[i]);
            } catch (IOException e) {
                e.printStackTrace();
            }

            BufferedImage temp = resize(image, width, height);

            try {
                ImageIO.write(temp, FilenameUtils.getExtension(imageFiles[i].getAbsolutePath()), new File(imageFiles[i].getAbsolutePath()));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        imageFiles = root.listFiles(imgFilter);

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image: imageFiles ) {

            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

            int label = Integer.parseInt(image.getName().split("_")[0]);
            images.put(counter, img);

            System.out.println("Image : " + img.toString());

            labelsBuf.put(counter, label);

            System.out.println("Label : " + label);

            counter++;
        }

        //Three different algorithms
        //FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
        //FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(images, labels);

        System.out.println("Training is done!");

        faceRecognizer.save(trainingDir + "\\TrainingFiles.xml");

        File testFile = new File(testDir);

        BufferedImage image = null;
        try {
            image = ImageIO.read(testFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        BufferedImage temp = resize(image, width, height);

        try {
            ImageIO.write(temp, FilenameUtils.getExtension(testFile.getAbsolutePath()), new File(testFile.getAbsolutePath()));
        } catch (IOException e) {
            e.printStackTrace();
        }

        Mat testImage = imread(testDir, CV_LOAD_IMAGE_GRAYSCALE);

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognizer.predict(testImage, label, confidence);
        int predictedLabel = label.get(0);

        System.out.println("Predicted label: " + predictedLabel);
    }

    public static BufferedImage resize(final Image image, int width, int height) {
        final BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        final Graphics2D graphics2D = bufferedImage.createGraphics();
        graphics2D.setComposite(AlphaComposite.Src);
        //below three lines are for RenderingHints for better image quality at cost of higher processing time
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING,RenderingHints.VALUE_RENDER_QUALITY);
        graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
        graphics2D.drawImage(image, 0, 0, width, height, null);
        graphics2D.dispose();
        return bufferedImage;
    }

}
