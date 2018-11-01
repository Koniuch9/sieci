import Jama.Matrix;

public class TrainingData {

	public Matrix inputs;
	public Matrix outputs;
	
	public TrainingData(Matrix i, Matrix o) 
	{
		inputs = i;
		outputs = o;
	}
}
