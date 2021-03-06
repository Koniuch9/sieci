import java.util.Random;

import Jama.*;

public class Network {
	public static final int inputs = 784;
	public static final int outputs = 10;
	private final double meanFactor = 1.;
	private double learningFactor;
	private double startingWeights;
	private int hiddenLayerSize;
	private double minError;
	private int batchSize;
	private int maxIterations;
	private double momentum;
	
	private Matrix weightsInputHidden;
	private Matrix weightsHiddenOutput;
	private Matrix netHidden;
	private Matrix netOutput;
	private Matrix prevHidden;
	private Matrix prevOutput;
	
	public Network(double learningFactor, double startingWeights, int hiddenLayerSize, double minError, int batchSize,
			int maxIterations, double momentum) {
		this.learningFactor = learningFactor;
		this.startingWeights = startingWeights;
		this.hiddenLayerSize = hiddenLayerSize;
		this.minError = minError;
		this.batchSize = batchSize;
		this.maxIterations = maxIterations;
		this.momentum = momentum;
		this.weightsInputHidden = Matrix.random(hiddenLayerSize+1,inputs+1).times(2*startingWeights).minus(new Matrix(hiddenLayerSize+1,inputs+1,startingWeights));		
		this.weightsHiddenOutput = Matrix.random(outputs,hiddenLayerSize+1).times(2*startingWeights).minus(new Matrix(outputs,hiddenLayerSize+1,startingWeights));
		this.prevHidden = new Matrix(hiddenLayerSize+1,inputs+1,0);
		this.prevOutput = new Matrix(outputs,hiddenLayerSize+1,0);		
	}	
	
	/**
	 * Uczy sieć aż nie osiągnie błędu zadanego albo przekroczy dozwoloną liczbę iteracji
	 * @param inputs [inputs+1 X trainSize]
	 * @param results [outputs X trainSize]
	 */
	public void train(Matrix inputs, Matrix results, Matrix testIn, Matrix testOut) 
	{	
		/*Matrix inputBatch = inputs.getMatrix(0,Network.inputs,0,batchSize-1);
		Matrix resultBatch = results.getMatrix(0,Network.outputs-1,0,batchSize-1);	*/
		//System.out.print("Input batch");
		//inputBatch.print(2,2);
		int i = 0;
		int iter = 0;
		double err = .0;
		Random r = new Random();
		//for(int i=0;i<maxIterations && error();i++)
		Matrix cost = null;
		do {
			Matrix inputBatch = inputs.getMatrix(0,Network.inputs,i*batchSize,(i+1)*batchSize-1);
			Matrix resultBatch = results.getMatrix(0,Network.outputs-1,i*batchSize,(i+1)*batchSize-1);
			Matrix outputs = propagateForward(inputBatch);
			cost = costFunction(outputs,resultBatch);						
			propagateBackward(cost,inputBatch,netHidden,outputs);			
			i++;
			err += errors(cost);
			if(batchSize*(1+i) > L2Main.trainSize) {
				err /= i;
				i = r.nextInt(batchSize);
				iter++;
				learningFactor*=.97;
				System.out.format("Epoka %d, error: %.5f, wsp_uczenia: %.4f, test: %.2f%%\n",iter,err,learningFactor,test(testIn,testOut)/L2Main.testSize*100);
			}
		}while(iter < maxIterations && err > minError);
		System.out.println(iter);
	}
	
	public double test(Matrix input, Matrix output) 
	{
		Matrix results = propagateForward(input);
		return classified(results).transpose().times(output).trace();
	}	
	
	public Matrix classified(Matrix result) 
	{
		Matrix ret = new Matrix(result.getRowDimension(),result.getColumnDimension(),0);
		for(int i=0;i < result.getColumnDimension();i++) 
		{
			double max = -1.,pom;
			int index = 0;
			for(int j=0;j<result.getRowDimension();j++) 
			{
				if((pom=result.get(j,i)) > max) 
				{
					max = pom;
					index = j;
				}
			}
			ret.set(index,i,1.);
		}
		return ret;
	}
	
	/** 
	 * @param inputBatch [inputs+1 X batchSize] : inputy 
	 * @return [outputs X batchSize] : outputy dla batcha
	 */
	public Matrix propagateForward(Matrix inputBatch) 
	{		
		Matrix firstLayer = sigmoid(weightsInputHidden.times(inputBatch));
		netHidden = firstLayer;	
		for(int i = 0;i < batchSize;i++)firstLayer.set(hiddenLayerSize,i,1.);			
		return netOutput=sigmoid(weightsHiddenOutput.times(firstLayer));
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
		Matrix deltaO = deltaOutput(errors);
		/*System.out.println("Delta Output");
		deltaO.print(2,2);*/
		Matrix deltaH = deltaHidden(deltaO);
		/*System.out.println("Delta Hidden");
		deltaH.print(2,2);*/
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
	public double errors(Matrix cost) 
	{		
		Matrix ones = new Matrix(batchSize,1,1./(2.*batchSize));
		return (new Matrix(1,10,1./10.)).times(cost.arrayTimes(cost).times(ones)).get(0,0);
	}
	
	/** 
	 * @param netOutput [outputs X batchSize] 
	 * @param errors [outputs X batchSize]
	 * @return [batchSize X outputs] : średnia delta dla outputu
	 */
	public Matrix deltaOutput(Matrix errors) 
	{		
		return (errors.arrayTimes(dSigmoid(netOutput))).transpose();
	}
	
	/** 
	 * @param netHidden [hiddenLayerSize+1 X batchSize]
	 * @param weightsHidden [outputs X hiddenLayerSize+1] 
	 * @param deltaO [batchSize X outputs] 
	 * @return [batchSize X hiddenLayerSize+1]
	 */
	public Matrix deltaHidden(Matrix deltaO) 
	{		
		return deltaO.times(weightsHiddenOutput).arrayTimes(dSigmoid(netHidden).transpose());
	}
	
	/**
	 * Updatuje wagi od inputów do hidden layera
	 * @param deltaH [batchSize X hiddenLayerSize+1]
	 * @param activationsInput [inputs+1 X batchSize]
	 * weightsInputHidden [hiddenLaywerSize+1 X inputs+1]
	 */
	public void updateInputHidden(Matrix deltaH, Matrix activationsInput) 
	{		
		
		weightsInputHidden.plusEquals(prevHidden = (prevHidden.times(momentum).plus((meanBatch(activationsInput.times(deltaH))).transpose()).times(learningFactor)));
	}
	
	/**
	 * Update wagi od hidden do outputów
	 * @param deltaO : [batchSize X outputs]
	 * @param activationsHidden : [hiddenLayerSize+1 X batchSize]
	 * weightsHiddenOutput [outputs X hiddenLayerSize+1]
	 */
	public void updateHiddenOutput(Matrix deltaO, Matrix activationsHidden) 
	{		
		weightsHiddenOutput.plusEquals(prevOutput = (prevOutput.times(momentum).plus(prevOutput=(meanBatch(activationsHidden.times(deltaO))).transpose())).times(learningFactor));
	}
	
	public Matrix meanBatch(Matrix batch)
	{		
		return batch.times(1./(meanFactor*batchSize));
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
	
	public void showBatch(Matrix inputs, Matrix results) 
	{
		Random r = new Random();
		int x = r.nextInt(L2Main.testSize/batchSize);
		Matrix inputBatch = inputs.getMatrix(0,Network.inputs,x*batchSize,(x+1)*batchSize-1);
		Matrix resultBatch = results.getMatrix(0,Network.outputs-1,x*batchSize,(x+1)*batchSize-1);
		Matrix outs = propagateForward(inputBatch);
		System.out.println("Outputs");
		outs.print(2, 2);
		System.out.println("Desireed outputs");
		resultBatch.print(2,2);
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
