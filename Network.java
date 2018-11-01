import Jama.*;

public class Network {
	public static final int inputs = 784;
	public static final int outputs = 10;
	private double learningFactor;
	private double startingWeights;
	private int hiddenLayerSize;
	private double minError;
	private int batchSize;
	private int maxIterations;
	private double momentum;
	
	private Matrix weightsInputHidden;
	private Matrix weightsHiddenOutput;
	private Matrix hidden;
	
	public Network(double learningFactor, double startingWeights, int hiddenLayerSize, double minError, int batchSize,
			int maxIterations) {
		this.learningFactor = learningFactor;
		this.startingWeights = startingWeights;
		this.hiddenLayerSize = hiddenLayerSize;
		this.minError = minError;
		this.batchSize = batchSize;
		this.maxIterations = maxIterations;
		
		this.weightsInputHidden = Matrix.random(hiddenLayerSize+1,inputs+1).times(2*startingWeights).minus(new Matrix(hiddenLayerSize+1,inputs+1,startingWeights));		
		this.weightsHiddenOutput = Matrix.random(outputs,hiddenLayerSize+1).times(2*startingWeights).minus(new Matrix(outputs,hiddenLayerSize+1,startingWeights));		
		
	}	
	
	/**
	 * Uczy sieć aż nie osiągnie błędu zadanego albo przekroczy dozwoloną liczbę iteracji
	 * @param inputs [inputs+1 X trainSize]
	 * @param results [outputs X trainSize]
	 */
	public void train(Matrix inputs, Matrix results) 
	{	
		Matrix inputBatch = inputs.getMatrix(0,Network.inputs,0,batchSize-1);
		inputBatch.print(2,2);
		Matrix resultBatch = results.getMatrix(0,Network.outputs-1,0,batchSize-1);
		resultBatch.print(2,2);
		for(int i=0;i<maxIterations;i++) {
			/*Matrix inputBatch = inputs.getMatrix(0,Network.inputs,i*batchSize,(i+1)*batchSize-1);
			Matrix resultBatch = results.getMatrix(0,Network.outputs-1,i*batchSize,(i+1)*batchSize-1);*/
			Matrix outputs = propagateForward(inputBatch);
			Matrix cost = costFunction(outputs,resultBatch);
			Matrix errors = errors(cost);
			propagateBackward(errors,inputBatch,hidden,outputs);
			System.out.println("Outputs:");
			outputs.print(2, 2);
			System.out.println("Desired outputs");
			resultBatch.print(2,2);
			System.out.println("Cost");
			cost.print(2, 2);
			System.out.println("Errors");
			errors.print(2,2);
		}
	}
	
	/** 
	 * @param inputBatch [inputs+1 X batchSize] : inputy 
	 * @return [outputs X batchSize] : outputy dla batcha
	 */
	public Matrix propagateForward(Matrix inputBatch) 
	{		
		Matrix firstLayer = sigmoid(weightsInputHidden.times(inputBatch));
		for(int i = 0;i < batchSize;i++)firstLayer.set(hiddenLayerSize,i,1.);	
		hidden = firstLayer;		
		return sigmoid(weightsHiddenOutput.times(firstLayer));
	}	
	
	/**
	 * backprop alg
	 * @param errors [outputs X 1] : średni błąd dla onputa z batcha
	 * @param activationsInput [inputs+1 X batchSize] : aktywacje na inpucie
	 * @param activationsHidden [hiddenLayerSize+1 X batchSize] : aktywacje na hiddenie
	 * @param activationsOutput [outputs X batchSize] : aktywacje na outpucie
	 */
	public void propagateBackward(Matrix errors,Matrix activationsInput, Matrix activationsHidden, Matrix activationsOutput) 
	{
		Matrix deltaO = deltaOutput(activationsOutput, errors);
		System.out.println("Delta Output");
		deltaO.print(2,2);
		Matrix deltaH = deltaHidden(activationsHidden, deltaO);
		System.out.println("Delta Hidden");
		deltaH.print(2,2);
		updateInputHidden(deltaH,activationsInput);
		updateHiddenOutput(deltaO,activationsHidden);
	}
	
	/** 
	 * @param outputs [outputs X batchSize] : outputy dla batcha
	 * @param desiredOutputs [outputs X batchSize] : pożądane outputy ze zbioru testowego
	 * @return [outputs X batchSize] : różnica pożądanych outputów i tych co były
	 */
	public Matrix costFunction(Matrix outputs, Matrix desiredOutputs)
	{
		return desiredOutputs.minus(outputs);
	}
	
	/** 
	 * @param cost [outputs X batchSize] : różnica dOutów i outów
	 * ones [batchSize X 1] : wypełniona 1  
	 * @return [outputs X 1] : cost * ones : średni błąd dla każdego outputu 
	 */
	public Matrix errors(Matrix cost) 
	{
		Matrix ones = new Matrix(batchSize,1,1.);
		return cost.times(ones).times(1./(2.*batchSize));
	}
	
	/** 
	 * @param activationsOutput [outputs X batchSize] 
	 * @param errors [outputs X 1]
	 * @return [1 X outputs] : średnia delta dla outputu
	 */
	public Matrix deltaOutput(Matrix activationsOutput, Matrix errors) 
	{
		Matrix ones = new Matrix(batchSize,10,1.);
		return errors.transpose().times(dSigmoid(activationsOutput.times(ones).times(1./(2.*batchSize))));
	}
	
	/** 
	 * @param activationsHidden [hiddenLayerSize+1 X batchSize]
	 * @param weightsHidden [outputs X hiddenLayerSize+1] 
	 * @param deltaO [1 X outputs] 
	 * @return [1 X hiddenLayerSize+1]
	 */
	public Matrix deltaHidden(Matrix activationsHidden, Matrix deltaO) 
	{
		Matrix ones = new Matrix(batchSize,hiddenLayerSize+1,1.);
		return deltaO.times(weightsHiddenOutput).times(dSigmoid(activationsHidden.times(ones).times(1./(2.*batchSize))));
	}
	
	/**
	 * Updatuje wagi od inputów do hidden layera
	 * @param deltaH [1 X hiddenLayerSize+1]
	 * @param activationsInput [inputs+1 X batchSize]
	 * weightsInputHidden [hiddenLaywerSize+1 X inputs+1]
	 */
	public void updateInputHidden(Matrix deltaH, Matrix activationsInput) 
	{
		Matrix ones = new Matrix(batchSize,1,1.);
		weightsInputHidden.plusEquals((activationsInput.times(ones).times(1./(2.*batchSize)).times(deltaH)).transpose().times(learningFactor));
	}
	
	/**
	 * Update wagi od hidden do outputów
	 * @param deltaO : [1 X outputs]
	 * @param activationsHidden : [hiddenLayerSize+1 X batchSize]
	 * weightsHiddenOutput [outputs X hiddenLayerSize+1]
	 */
	public void updateHiddenOutput(Matrix deltaO, Matrix activationsHidden) 
	{
		Matrix ones = new Matrix(batchSize,1,1.);
		weightsHiddenOutput.plusEquals((activationsHidden.times(ones).times(1./(2.*batchSize)).times(deltaO)).transpose().times(learningFactor));
	}
	
	public double sigmoid(double x) 
	{
		return 1./(1. + Math.exp(-x));
	}
	
	public Matrix sigmoid(Matrix m) 
	{
		int x,y;
		Matrix ret = new Matrix(x=m.getRowDimension(),y=m.getColumnDimension());
		for(int i=0;i<x;i++) 
		{
			for(int j=0;j<y;j++) 
			{
				ret.set(i,j,sigmoid(m.get(i,j)));
			}
		}
		return ret;
	}
	
	public double dSigmoid(double x) 
	{
		return x * (1. - x);
	}
	
	public Matrix dSigmoid(Matrix m) 
	{
		int x,y;
		Matrix ret = new Matrix(x=m.getRowDimension(),y=m.getColumnDimension());
		for(int i=0;i<x;i++) 
		{
			for(int j=0;j<y;j++) 
			{
				ret.set(i,j,dSigmoid(m.get(i,j)));
			}
		}
		return ret;
	}
	
	public void printWeights() 
	{
		System.out.print("Weights input-hidden: " + weightsInputHidden.getRowDimension() + " x " + weightsInputHidden.getColumnDimension());
		weightsInputHidden.print(2,2);
		System.out.print("Weights hidden-output: " + weightsHiddenOutput.getRowDimension() + " x " + weightsHiddenOutput.getColumnDimension());
		weightsHiddenOutput.print(2,2);
	}
	
	public double getLearningFactor() {
		return learningFactor;
	}

	public void setLearningFactor(double learningFactor) {
		this.learningFactor = learningFactor;
	}

	public double getStartingWeights() {
		return startingWeights;
	}

	public void setStartingWeights(double startingWeights) {
		this.startingWeights = startingWeights;
	}

	public int getHiddenLayerSize() {
		return hiddenLayerSize;
	}

	public void setHiddenLayerSize(int hiddenLayerSize) {
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public double getMinError() {
		return minError;
	}

	public void setMinError(double minError) {
		this.minError = minError;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}
}
