import java.util.Hashtable;

class Word {
	private String word;

	private String pos_tag;

	private Hashtable otherFeatures;

	public Word() {}

	public Word(String lemme) {
		word = lemme;
		pos_tag = null;
		otherFeatures = null;
	}
	public Word(String lemme, String tag) {
		word = lemme;
		pos_tag = tag;
		otherFeatures = null;
	}

	public String getLemme() {
		return word;
	}

	public String getPosTag() {
		return pos_tag;
	}
	
	public void setPosTag(String tag) {
		pos_tag = tag;
	}

	public Hashtable getOtherFeatures() {
		return otherFeatures;
	}
}
