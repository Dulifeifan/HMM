import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
import Jama.Matrix;
import com.sun.jndi.toolkit.ctx.AtomicDirContext;

class HMM {
	/* Section for variables regarding the data */
	//
	private static ArrayList<Sentence> labeled_corpus;
	//
	private static ArrayList<Sentence> unlabeled_corpus;
	// number of pos tags
	static int num_postags;
	// mapping POS tags in String to their indices
	static Hashtable<String, Integer> pos_tags;
	// inverse of pos_tags: mapping POS tag indices to their String format
	static Hashtable<Integer, String> inv_pos_tags;
	// vocabulary size
	static int num_words;
	static Hashtable<String, Integer> vocabulary;
	private int max_sentence_length;
	/* Section for variables in HMM */
	// transition matrix
	private static Matrix A;
	private static Matrix AL;
	// emission matrix
	private static Matrix B;
	private static Matrix BL;
	// prior of pos tags
	private static Matrix pi;
	private static Matrix PL;
	// store the scaled alpha and beta
	private static Matrix alpha;
	private static Matrix beta;
	// scales to prevent alpha and beta from underflowing
	private static Matrix scales;
	// logged v for Viterbi
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;
	// \xi_t(i): expected frequency of pos tag i at position t. Use as an accumulator.
	private static Matrix gamma;
	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at position t.  Use as an accumulator.
	private static Matrix digamma;
	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private static Matrix gamma_w;
	// \xi_0(i): expected frequency of pos tag i at position 0.
	private static Matrix gamma_0;
	/* Section of parameters for running the algorithms */
	// smoothing epsilon for the B matrix (since there are likely to be unseen words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.1;
	// number of iterations of EM
	private int max_iters = 10;
	// \mu: a value in [0,1] to balance estimations from MLE and EM
	// \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM totally.
	private static double mu = 0.8;
	/* Section of variables monitoring training */
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	private double[][][] xi;
	static double[][] aijn;
	static double[] aijd;
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
	}
	/**
	 * Set the semi-supervised parameter \mu
	 */
	public void setMu(double _mu) {
		if (_mu < 0) {
			this.mu = 0.0;
		} else if (_mu > 1) {
			this.mu = 1.0;
		}
		this.mu = _mu;
	}
	private static void scale(double[] scaling, Matrix alpha, int t) {
		double[] table = alpha.getArray()[t];

		double sum = 0.0;
		for (int i = 0; i < table.length; i++) {
			sum += table[i];
		}

		scaling[t] = sum;
		for (int i = 0; i < table.length; i++) {
			table[i] /= sum;
		}
	}
	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
		A = new Matrix(num_postags, num_postags, 0.000001);
		B = new Matrix(num_postags, num_words, 0.000001);
		pi = new Matrix(num_postags, 1, 0.000001);
		AL = new Matrix(num_postags, num_postags, 0.000001);
		BL = new Matrix(num_postags, num_words, 0.000001);
		PL = new Matrix(num_postags, 1, 0.000001);
	}
	/**
	 * MLE A, B and pi on a labeled corpus
	 * used as initialization of the parameters.
	 */
	public void mle() {
		double sumpi = 0;
		for (int i = 0; i < num_postags; i++) {
			double sumA = 0;
			double sumB = 0;
			for (int j = 0; j < num_postags; j++) {
				sumA = sumA + A.get(i, j);
			}
			for (int j = 0; j < num_postags; j++) {
				A.set(i, j, A.get(i, j) / sumA);
			}
			for (int j = 0; j < num_words; j++) {
				sumB = sumB + B.get(i, j);
			}
			for (int j = 0; j < num_words; j++) {
				B.set(i, j, B.get(i, j) / sumB);
			}
			sumpi = sumpi + pi.get(i, 0);
		}
		for (int i = 0; i < num_postags; i++) {
			pi.set(i, 0, pi.get(i, 0) / sumpi);
		}
	}
	/**
	 * Main EM algorithm.
	 */
	public void em() {
		double gammat[][][]=new double[unlabeled_corpus.size()][][];
		aijn=new double[num_postags][num_postags];
		aijd= new double[num_postags];
		for(int k=0;k<unlabeled_corpus.size();k++)
		{
			double scaling[]=new double[unlabeled_corpus.get(k).length()];
				forward(unlabeled_corpus.get(k),scaling);
				backward(unlabeled_corpus.get(k),scaling);
				gammat[k]=expectation(unlabeled_corpus.get(k));
		}
		maximization(gammat);
		for(int i=0;i<num_postags;i++)
		{
			for(int j=0;j<num_postags;j++)
			{
				A.set(i,j,AL.get(i,j)*mu+A.get(i,j)*(1-mu));
			}
			for(int j=0;j<num_words;j++)
			{
				B.set(i,j,BL.get(i,j)*mu+B.get(i,j)*(1-mu));
			}
			pi.set(i,0,PL.get(i,0)*mu+pi.get(i,0)*(1-mu));
		}
	}
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
		double logp=0;
		for(int i=0; i<unlabeled_corpus.size();i++)
		{
			viterbi(unlabeled_corpus.get(i));
			for (int j=0; j<unlabeled_corpus.get(i).length();j++)
			{
				unlabeled_corpus.get(i).getWordAt(j).setPosTag(inv_pos_tags.get((int)pred_seq.get(j,0)));
			}
		}
	}
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		for(int i=0;i<unlabeled_corpus.size();i++)
		{
			for(int j=0;j<unlabeled_corpus.get(i).length();j++)
			{
				bw.write(unlabeled_corpus.get(i).getWordAt(j).getLemme()+" "+
						unlabeled_corpus.get(i).getWordAt(j).getPosTag()+"\n");
			}
			bw.write("\n");
			bw.flush();
		}
		bw.close();
	}
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		double sum=0;
		for(int i=0;i<unlabeled_corpus.size();i++)
		{
	//		sum=sum+forward(unlabeled_corpus.get(i));
		}
		bw.write(""+sum);
		bw.close();
	}
	/**
	 * Expectation step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double[][] expectation(Sentence s) {
		ArrayList<Integer> index=new ArrayList<>();
		for(int i=0;i<s.length();i++)
		{
			index.add(i,vocabulary.get(s.getWordAt(i).getLemme()));
		}

		if(s.length()>2) {
			//System.out.println(s.length()+" "+num_postags);
			xi = new double[s.length() - 1][num_postags][num_postags];
			for (int t = 0; t < s.length() - 1; t++) {
				for (int i = 0; i < num_postags; i++) {
					for (int j = 0; j < num_postags; j++) {
						xi[t][i][j] = alpha.get(t, i) *
								A.get(i, j) *
								B.get(j, index.get(t+1)) *
								beta.get(t + 1, j);
					}
				}
			}
//		}
		gamma=new Matrix(xi.length+1,num_postags);
			for (int t = 0; t < xi.length; t++) {
				for (int i = 0; i < num_postags; i++) {
					for (int j = 0; j < num_postags; j++) {
						gamma.set(t,i,gamma.get(t,i)+xi[t][i][j]) ;
					}
				}
			}
			for (int j = 0; j < num_postags; j++) {
				for (int i = 0; i < num_postags; i++) {
					gamma.set(xi.length,j,gamma.get(xi.length,j)+xi[xi.length-1][i][j]);
				}
			}
		for (int i = 0; i < num_postags; i++) {
			double sum1 = 0;
			for (int j = 0; j < num_postags; j++) {
				double sum = 0;
				for (int t = 0; t < s.length()-1; t++) {
					sum += xi[t][i][j];
					sum1 += xi[t][i][j];
				}
				aijn[i][j]+=sum;
			}
			aijd[i] += sum1;
		}}
		index.clear();
		return gamma.getArray();
		}
	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private static void maximization(double gammat[][][]) {
		for(int i=0;i<num_postags;i++)
		{
			if(aijd[i]==0)
			{

			}
			else
				{
					for(int j=0; j<num_postags;j++)
					{
						A.set(i,j,aijn[i][j]/aijd[i]);
					}
				}
		}
		for(int j=0;j<unlabeled_corpus.size();j++)
		{
			for(int i=0;i<num_postags;i++)
			{
				pi.set(i,0,pi.get(i,0)+gammat[j][0][i]);
			}
		}
		for(int i=0;i<num_postags;i++)
		{
			pi.set(i,0,pi.get(i,0)/unlabeled_corpus.size());
		}
		for(int i=0; i<num_postags;i++)
		{
			double sum=0.0;
			for(int j=0; j<unlabeled_corpus.size();j++)
			{
				for(int t=0; t<unlabeled_corpus.get(j).length();t++)
				{
					B.set(i,vocabulary.get(unlabeled_corpus.get(j).getWordAt(t).getLemme()),
							B.get(i,vocabulary.get(unlabeled_corpus.get(j).getWordAt(t).getLemme()))+gammat[j][t][i]);
					sum=sum+gammat[j][t][i];
				}
			}
			for(int j=0;j<num_words;j++)
			{
				B.set(i,j,B.get(i,j)/sum);
			}
		}
	}
	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)
	 * return: log P(O|\lambda)
	 */
	private static double forward(Sentence s,double scaling[]) {
		ArrayList<Integer> index=new ArrayList<>();
		for(int i=0;i<s.length();i++)
		{
			index.add(i,vocabulary.get(s.getWordAt(i).getLemme()));
		}
		alpha=new Matrix(s.length(),num_postags,0.0);
        double p=0;
		for(int i=0; i<num_postags; i++)
		{
			alpha.set(0,i,
					pi.get(i,0)*B.get(i,index.get(0)));
		}
		scale(scaling,alpha,0);
		for(int i=1; i<s.length();i++)
		{
			for (int j=0; j<num_postags;j++)
			{
				double sum=0.0;
				for(int n=0; n<num_postags;n++)
				{
					sum=sum+alpha.get(i-1,n)*A.get(n,j);
				}
				alpha.set(i,j,sum*B.get(j,index.get(i)));
			}
			scale(scaling,alpha,i);
		}
		for(int i=0; i<s.length();i++)
		{
			p=p+Math.log(scaling[i]);
		}
		index.clear();
		return p;
	}
	/**
	 * Backward algorithm for one sentence
	 *
	 * return: log P(O|\lambda)
	 */
	private static double backward(Sentence s,double scaling[])
	{
		ArrayList<Integer> index=new ArrayList<>();
		for(int i=0;i<s.length();i++)
		{
			index.add(i,vocabulary.get(s.getWordAt(i).getLemme()));
		}
		beta = new Matrix(s.length(), num_postags, 0.0);
		for (int i = 0; i < num_postags; i++)
		{
			beta.set(s.length() - 1, i, 1.0/scaling[s.length()-1]);
		}
		for (int t = s.length() - 2; t >= 0; t--)
		{
			for (int j = 0; j < num_postags; j++)
			{
				double sum = 0;
				for (int i = 0; i < num_postags; i++)
				{
					sum = sum + A.get(j, i) * B.get(i, index.get(t+1)) * beta.get(t + 1, i);
				}
				beta.set(t, j, sum/scaling[t]);
			}
		}
		double p = 0;
		for (int i = 0; i < num_postags; i++)
		{
			p = p + beta.get(0, i) * pi.get(i, 0) * B.get(i, index.get(0));
		}
		double logp = Math.log(p);
		index.clear();
		return logp;
	}
	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s) {
		ArrayList<Integer> index=new ArrayList<>();
		for(int i=0;i<s.length();i++)
		{
			index.add(i,vocabulary.get(s.getWordAt(i).getLemme()));
		}
		v= new Matrix(s.length(),num_postags,0);
		back_pointer=new Matrix(s.length(),num_postags,0);
		for(int j=0; j<num_postags;j++)
		{
			v.set(0,j,
					Math.log(pi.get(j,0))+Math.log(B.get(j,index.get(0))));
		}
		for(int i=1;i<s.length();i++)
		{
			for(int j=0;j<num_postags;j++)
			{
				double max=v.get(i-1,0)+Math.log(A.get(0,j));
				int maxi=0;
				for (int n=0;n<num_postags; n++)
				{
					double cv=v.get(i-1,n)+Math.log(A.get(n,j));
					if(cv>max)
					{
						max=cv;
						maxi=n;
					}
				}
				v.set(i,j,max+Math.log(B.get(j,index.get(i))));
				back_pointer.set(i,j,maxi);
			}
		}
		pred_seq= new Matrix(s.length(),1,0);
		double p=v.get(s.length()-1,0);
		pred_seq.set(s.length()-1,0,0);
		for(int i=1;i<num_postags;i++)
		{
			if(v.get(s.length()-1,i)>p)
			{
				p=v.get(s.length()-1,i);
				pred_seq.set(s.length()-1,0,i);
			}
		}
		for(int i=s.length()-2;i>=0;i--)
		{
			pred_seq.set(i,0,back_pointer.get(i+1,(int)pred_seq.get(i+1,0)));
		}
		index.clear();
		return p;
	}
	public static void main(String[] args) throws IOException {
/*		args[0]="./data/p1/train.txt";
		args[1]="./data/p1/test.txt";
		args[2]="./results/p1/prediction.txt";
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}


		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
*/
		String labeledFileName = "./data/p1/train.txt";
		String unlabeledFileName = "./data/p3/concatenated.txt";
		String predictionFileName = "./data/p3/predictions_concatenated";
		String trainingLogFileName = "./results/p3/log_concatenated";
//		String unlabeledFileName = "./data/p1/test.txt";
//		String predictionFileName = "./results/p3/predictions";
//		String trainingLogFileName = "./results/p3/log";
		if (args.length > 3) {
			trainingLogFileName = args[3];
		}
		//double mu = 0.0;
		if (args.length > 4) {
			mu = Double.parseDouble(args[4]);
		}
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		labeled_corpus = fh.readTaggedSentences(labeledFileName);
		unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);
		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		vocabulary=new Hashtable<>();
		pos_tags=new Hashtable<>();
		inv_pos_tags=new Hashtable<>();
		int kvw=0;
		int kvp=0;
		for (int i=0; i<labeled_corpus.size();i++)
		{
			for(int j=0; j<labeled_corpus.get(i).length();j++)
			{
				if(!vocabulary.containsKey(labeled_corpus.get(i).getWordAt(j).getLemme()))
				{
					if (labeled_corpus.get(i).getWordAt(j).getLemme() == null) {
						throw new NullPointerException();
					}
					else
					{
						vocabulary.put(labeled_corpus.get(i).getWordAt(j).getLemme(),kvw);
						kvw++;
					}
				}
				if(!pos_tags.containsKey(labeled_corpus.get(i).getWordAt(j).getPosTag())) {
					pos_tags.put(labeled_corpus.get(i).getWordAt(j).getPosTag(), kvp);
					inv_pos_tags.put(kvp, labeled_corpus.get(i).getWordAt(j).getPosTag());
					kvp++;
				}
			}
		}
		for(int i=0;i<unlabeled_corpus.size();i++)
		{
			for(int j=0; j<unlabeled_corpus.get(i).length();j++)
			{
				if(!vocabulary.containsKey(unlabeled_corpus.get(i).getWordAt(j).getLemme()))
				{
					if (unlabeled_corpus.get(i).getWordAt(j).getLemme() == null) {
						throw new NullPointerException();
					}
					else
					{
						vocabulary.put(unlabeled_corpus.get(i).getWordAt(j).getLemme(),kvw);
						kvw++;
					}
				}
			}
		}
		num_postags=pos_tags.size();
		num_words=vocabulary.size();
		model.prepareMatrices();
		for (int i=0; i<labeled_corpus.size();i++)
		{
			for(int j=0; j<labeled_corpus.get(i).length()-1;j++)
			{
				A.set(pos_tags.get(labeled_corpus.get(i).getWordAt(j).getPosTag()),
						pos_tags.get(labeled_corpus.get(i).getWordAt(j+1).getPosTag()),
						A.get(pos_tags.get(labeled_corpus.get(i).getWordAt(j).getPosTag()),
								pos_tags.get(labeled_corpus.get(i).getWordAt(j+1).getPosTag()))+1);
			}
			for(int j=0;j<labeled_corpus.get(i).length();j++)
			{
				B.set(pos_tags.get(labeled_corpus.get(i).getWordAt(j).getPosTag()),
						vocabulary.get(labeled_corpus.get(i).getWordAt(j).getLemme()),
						B.get(pos_tags.get(labeled_corpus.get(i).getWordAt(j).getPosTag()),
								vocabulary.get(labeled_corpus.get(i).getWordAt(j).getLemme()))+1);
			}
			pi.set(pos_tags.get(labeled_corpus.get(i).getWordAt(0).getPosTag()),
					0,
					pi.get(pos_tags.get(labeled_corpus.get(i).getWordAt(0).getPosTag()),
							0)+1);
		}
		model.mle();
		AL=A.copy();
		BL=B.copy();
		PL=pi.copy();

		for(mu=0.0;mu<=1.0;mu=mu+0.1) {
			A=AL.copy();
			B=BL.copy();
			pi=PL.copy();
			FileWriter fw = new FileWriter(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int j = 0; j < 30; j++) {
				model.em();
				double sumf = 0;
				double sumb = 0;
				for (int i = 0; i < unlabeled_corpus.size(); i++) {
					double scaling[] = new double[unlabeled_corpus.get(i).length()];
					sumf = sumf + forward(unlabeled_corpus.get(i), scaling);
				}
				System.out.println(j + "	" + sumf + "	" + sumb);
				bw.write(sumf + "\n");
				//bw.write(""+sumb+"\n");
				bw.flush();

			}
			bw.close();
			model.predict();
			model.outputPredictions(predictionFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
/*		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
*/
/*
		model.setMu(mu);
*/
	}
}
