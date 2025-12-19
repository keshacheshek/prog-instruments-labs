import java.io.FileWriter;
import java.util.Random;
import java.io.IOException;

public class java_PRNG {
    public static void main(String[] args) throws Exception {
        Random rand = new Random();
        try (FileWriter writer = new FileWriter("java_Sequence.txt")) {

            for (int i = 0; i < 128; i++) {
                int number = rand.nextInt(2); // только числа 0 и 1
                System.out.print(number);
                writer.write(Integer.toString(number));
            }

            writer.close();
            System.out.println("\nSaved to java_Sequence.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}