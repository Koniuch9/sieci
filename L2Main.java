import java.io.IOException;
import Jama.*;

public class L2Main {
	public static final double learningFactor = .2;
	public static final double startingWeights = 1.;
	private static final int hiddenLayerSize = 100;
	private static final double minError = 0.01;
	private static final int batchSize = 50;
	private static final int maxIterations = 5;
	private static final double momentum = .5;
	
	public static final int testSize = 1000;
	public static final int trainSize = 60000;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			TrainingData x = Reader.read("mnist_train.csv",trainSize);
			TrainingData t = Reader.read("mnist_test.csv",testSize);
			Network n = new Network(learningFactor,startingWeights,hiddenLayerSize,minError,batchSize,maxIterations);			
			//n.printWeights();			
			//x.print(2,2);			
			//n.propagate(x).print(2,2);
			//n.(x);
			n.train(x.inputs,x.outputs);
			double good = n.test(t.inputs,t.outputs);
			System.out.println(good);
		}catch(IOException e) {e.printStackTrace();}
		
		/*Network n = new Network(3,3,3,3,3,3);
		n.printWeights();
		Matrix m = Matrix.random(6,6).timesEquals(10);
		m.print(2,2);
		n.sigmoid(m).print(2,2);*/
		
	}

}
