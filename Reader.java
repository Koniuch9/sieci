import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import Jama.*;

public class Reader {
	
	public static TrainingData read(String filename, int lineNo)throws IOException
	{
		BufferedReader bR = new BufferedReader(new FileReader(filename));		
		String line = null;
		String[] lines;
		int j=0;
		double[][] vals = new double[Network.inputs+1][lineNo]; 
		double[][] outs = new double[Network.outputs][lineNo];
		while((line = bR.readLine()) != null && j<lineNo) 
		{
			lines = line.split(",");			
			for(int i = 1;i<Network.inputs+1;i++) 
			{
				vals[i][j] = Double.parseDouble(lines[i])/255;
				
			}
			vals[Network.inputs][j]=1.;			
			for(int i=0;i<Network.outputs;i++)outs[i][j] = .0;
			outs[Integer.parseInt(lines[0])][j] = 1.;
			j++;
		}	
		
		bR.close();
		return new TrainingData(new Matrix(vals),new Matrix(outs));
	}
	
	

}
